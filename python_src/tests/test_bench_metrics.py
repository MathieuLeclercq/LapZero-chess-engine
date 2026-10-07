"""Tests du noyau de mesure du banc.

Aucun modele, aucun processus : les deux acces au reseau sont injectes, sur le
modele du fetcher de lichess_games.fetch_games. Le plateau, lui, est le vrai
Chessboard : il ne coute que 2 Mio a l'import et le simuler reviendrait a
tester une fiction.
"""
import os
import sys

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from evaluation.bench_metrics import BenchPuzzle, parse_bench_line

LIGNE = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    "|e2e4 e7e5 g1f3 b8c6 f1b5"
    "|a7a6 b5c6"
    "|1615"
    "|crushing discoveredAttack middlegame short"
)


def test_parse_bench_line_splits_the_five_fields():
    puzzle = parse_bench_line(7, LIGNE + "\n")

    assert puzzle.ligne == 7
    assert puzzle.fen_initiale.startswith("rnbqkbnr/")
    assert puzzle.coups_uci == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    assert puzzle.solution_uci == ["a7a6", "b5c6"]
    assert puzzle.rating == 1615
    assert puzzle.themes == "crushing discoveredAttack middlegame short"


def test_parse_bench_line_is_the_inverse_of_format_line():
    """Verrouille les deux modules ensemble : si build_puzzle_dataset change
    l'ordre des champs, ce test tombe au lieu de laisser le banc lire de
    travers."""
    import chess
    from evaluation.build_puzzle_dataset import MatchResult, PuzzleRow, format_line

    row = PuzzleRow(
        puzzle_id="abc12",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        moves=["f1b5", "a7a6", "b5c6"],
        rating=1500,
        themes="mateIn2 short",
        game_url="https://lichess.org/testtest#4",
    )
    match = MatchResult(start_fen=chess.STARTING_FEN,
                        moves_uci=["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])

    puzzle = parse_bench_line(0, format_line(row, match))

    assert puzzle.fen_initiale == chess.STARTING_FEN
    assert puzzle.coups_uci == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    assert puzzle.solution_uci == ["a7a6", "b5c6"]
    assert puzzle.rating == 1500
    assert puzzle.themes == "mateIn2 short"


def test_parse_bench_line_rejects_a_wrong_field_count():
    with pytest.raises(ValueError):
        parse_bench_line(0, "un|deux|trois\n")


import chess_engine
from evaluation.bench_metrics import (
    DonneesInvalides,
    charger_position,
    measure_puzzle,
    uci_to_index,
)

DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _puzzle(coups, solution, fen=DEPART, rating=1500, themes="fork short"):
    return BenchPuzzle(
        ligne=0, fen_initiale=fen, coups_uci=coups, solution_uci=solution,
        rating=rating, themes=themes,
    )


def _faux_reseau(coup_prefere, p=0.7):
    """policy_fn factice : met p sur coup_prefere, repartit le reste."""
    def policy_fn(board):
        indices = board.get_legal_move_indices()
        cible = uci_to_index(board, coup_prefere)
        reste = (1.0 - p) / max(1, len(indices) - 1)
        return {i: (p if i == cible else reste) for i in indices}, 0.25
    return policy_fn


def _visites_sur(board, coup):
    pi = [0.0] * 4672
    pi[uci_to_index(board, coup)] = 1.0
    return pi


def _recherche_sequentielle(coups):
    """search_fn factice : renvoie les coups fournis, dans l'ordre des appels.

    Plus explicite que d'inspecter des cases du plateau pour deviner a quel
    coup de la ligne on se trouve.
    """
    restants = list(coups)

    def search_fn(board):
        return _visites_sur(board, restants.pop(0))
    return search_fn


def test_charger_position_rejoue_l_historique():
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    board = charger_position(puzzle)

    assert board.turn == chess_engine.Color.BLACK
    # Le cavalier est arrive en f3, soit file 5 rang 2.
    assert board.get_square(5, 2).get_piece().get_type() == chess_engine.PieceType.KNIGHT


def test_charger_position_refuse_un_historique_illegal():
    puzzle = _puzzle(["e2e4", "e2e4"], ["b8c6"])

    with pytest.raises(DonneesInvalides) as info:
        charger_position(puzzle)

    assert info.value.cause == "historique_illegal"


def test_charger_position_sans_historique_donne_la_meme_position():
    """Le bras sans historique repart de la FEN atteinte, ce qui vide les plans
    d'historique du tenseur sans changer la position."""
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    avec = charger_position(puzzle)
    sans = charger_position(puzzle, sans_historique=True)

    assert avec.to_fen() == sans.to_fen()
    assert avec.get_legal_move_indices() == sans.get_legal_move_indices()
    assert not (avec.get_alphazero_tensor() == sans.get_alphazero_tensor()).all()


def test_measure_puzzle_compte_une_reussite_au_premier_coup():
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    mesure = measure_puzzle(puzzle, _faux_reseau("b8c6"),
                            _recherche_sequentielle(["b8c6"]))

    assert mesure.erreur == ""
    assert mesure.reussi_reseau is True
    assert mesure.reussi_recherche is True
    assert mesure.coup_reseau == "b8c6"
    assert mesure.coup_recherche == "b8c6"
    assert mesure.p_correct_reseau == pytest.approx(0.7)
    assert mesure.rang_correct_reseau == 1
    assert mesure.part_visites_correct == pytest.approx(1.0)
    assert mesure.plies_historique == 3
    assert mesure.nb_coups_legaux > 0


def test_measure_puzzle_accepte_le_retour_du_mode_temps():
    """En mode temps, search_fn renvoie aussi le bilan de la fenetre : le
    nombre de simulations terminees et le depassement du dernier appel."""
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    def search_fn(board):
        return (_visites_sur(board, "b8c6"),
                {"simulations": 24, "depassement_s": 0.012})

    mesure = measure_puzzle(puzzle, _faux_reseau("b8c6"), search_fn)

    assert mesure.reussi_recherche is True
    assert mesure.simulations_recherche == 24
    assert mesure.depassement_s == pytest.approx(0.012)


def test_measure_puzzle_compte_un_echec_et_le_rang_du_bon_coup():
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    mesure = measure_puzzle(puzzle, _faux_reseau("g8f6"),
                            _recherche_sequentielle(["g8f6"]))

    assert mesure.reussi_reseau is False
    assert mesure.reussi_recherche is False
    assert mesure.coup_recherche == "g8f6"
    assert mesure.rang_correct_reseau > 1
    assert mesure.part_visites_correct == pytest.approx(0.0)


def test_measure_puzzle_ne_lance_qu_une_recherche_sur_une_longue_solution():
    """Le banc ne suit plus la ligne : le self-play ne traite specialement que
    le premier coup, donc c'est lui seul qui est mesure. Le faux ne fournit
    qu'un coup et leverait IndexError si une seconde recherche etait lancee.
    """
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3", "b8c6"],
                     ["f1b5", "a7a6", "b5c6"])

    mesure = measure_puzzle(puzzle, _faux_reseau("f1b5"),
                            _recherche_sequentielle(["f1b5"]))

    assert mesure.erreur == ""
    assert mesure.reussi_recherche is True


def test_measure_puzzle_mesure_bien_la_position_du_puzzle():
    """Aucun coup de la solution n'est joue avant la recherche : la position
    vue par search_fn est celle du puzzle, sans quoi part_visites porterait sur
    le mauvais coup."""
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3", "b8c6"],
                     ["f1b5", "a7a6", "b5c6"])
    vu = {}

    def search_fn(board):
        vu["fen"] = board.to_fen()
        return _visites_sur(board, "f1b5")

    measure_puzzle(puzzle, _faux_reseau("f1b5"), search_fn)

    assert vu["fen"] == charger_position(puzzle).to_fen()


def test_measure_puzzle_gere_une_solution_d_un_seul_coup():
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["b8c6"])

    mesure = measure_puzzle(puzzle, _faux_reseau("b8c6"),
                            _recherche_sequentielle(["b8c6"]))

    assert mesure.reussi_recherche is True


def test_measure_puzzle_signale_une_solution_illegale():
    """Second controle, independant des tests de build_puzzle_dataset : si
    solution[0] n'est pas legal, on compte au lieu de planter."""
    puzzle = _puzzle(["e2e4", "e7e5", "g1f3"], ["e2e4"])

    mesure = measure_puzzle(puzzle, _faux_reseau("b8c6"),
                            _recherche_sequentielle(["b8c6"]))

    assert mesure.erreur == "solution_illegale"
    assert mesure.reussi_recherche is False


def test_measure_puzzle_signale_un_historique_illegal():
    puzzle = _puzzle(["e2e4", "e2e4"], ["b8c6"])

    mesure = measure_puzzle(puzzle, _faux_reseau("b8c6"),
                            _recherche_sequentielle(["b8c6"]))

    assert mesure.erreur == "historique_illegal"
    assert mesure.reussi_recherche is False


def test_measure_puzzle_compare_des_index_et_pas_des_chaines():
    """Une promotion dame s'ecrit 'a7a8q' cote Lichess et peut se decoder sans
    suffixe : comparer les chaines produirait un faux echec."""
    puzzle = _puzzle(["a2a4"], ["a7a8q"],
                     fen="7k/P7/8/8/8/8/8/7K w - - 0 1")
    # a2a4 n'existe pas dans cette position : on verifie d'abord le rejet,
    # puis on mesure la vraie position sans historique.
    assert measure_puzzle(puzzle, _faux_reseau("a7a8q"),
                          _recherche_sequentielle(["a7a8q"])).erreur == "historique_illegal"

    puzzle = _puzzle([], ["a7a8q"], fen="7k/P7/8/8/8/8/8/7K w - - 0 1")

    mesure = measure_puzzle(puzzle, _faux_reseau("a7a8q"),
                            _recherche_sequentielle(["a7a8q"]))

    assert mesure.erreur == ""
    assert mesure.reussi_recherche is True


from evaluation.bench_metrics import PuzzleMeasure, aggregate, mcnemar, wilson


def test_wilson_encadre_la_proportion():
    bas, haut = wilson(50, 100)

    assert bas < 0.5 < haut
    # Valeurs connues pour 50/100 a 95 pour cent.
    assert 0.40 < bas < 0.41
    assert 0.59 < haut < 0.60


def test_wilson_ne_sort_jamais_de_zero_un():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_se_resserre_quand_l_effectif_grandit():
    """C'est tout l'interet de l'afficher : distinguer un ecart d'un bruit."""
    petit = wilson(50, 100)
    grand = wilson(500, 1000)

    assert (grand[1] - grand[0]) < (petit[1] - petit[0])


def test_mcnemar_sur_une_table_connue():
    chi2, p = mcnemar(10, 40)

    assert chi2 == pytest.approx((abs(10 - 40) - 1) ** 2 / 50)
    assert p < 0.001


def test_mcnemar_sans_discordance_ne_signale_rien():
    assert mcnemar(0, 0) == (0.0, 1.0)

    chi2, p = mcnemar(20, 20)
    assert chi2 == 0.0
    assert p == pytest.approx(1.0)


def test_mcnemar_est_symetrique():
    """Le test dit s'il y a un ecart, pas dans quel sens : ce sont b et c,
    rapportes separement, qui portent le sens."""
    assert mcnemar(10, 40) == mcnemar(40, 10)


def _m(ligne, rating, reseau, recherche, p=0.5, part=0.5,
       themes="fork short", erreur="", nb_legaux=30):
    return PuzzleMeasure(
        ligne=ligne, rating=rating, themes=themes,
        plies_historique=20, nb_coups_legaux=nb_legaux,
        coup_reseau="e2e4", reussi_reseau=reseau, p_correct_reseau=p,
        rang_correct_reseau=1, value_reseau=0.1,
        coup_recherche="e2e4", reussi_recherche=recherche,
        part_visites_correct=part,
        duree_s=0.5, erreur=erreur,
    )


def test_aggregate_compte_les_taux_globaux():
    mesures = [
        _m(0, 1100, True, True),
        _m(1, 1500, False, True),
        _m(2, 2000, True, False),
        _m(3, 2400, False, False),
    ]

    stats = aggregate(mesures)

    assert stats.global_.total == 4
    assert stats.global_.reseau == 2
    assert stats.global_.recherche == 2


def test_aggregate_ventile_par_tranche_de_rating():
    mesures = [
        _m(0, 1100, True, True),
        _m(1, 1500, True, True),
        _m(2, 1500, False, False),
    ]

    stats = aggregate(mesures)

    assert stats.par_tranche["1000-1449"].total == 1
    assert stats.par_tranche["1450-1899"].total == 2
    assert stats.par_tranche["1450-1899"].reseau == 1
    assert stats.par_tranche["1900-2349"].total == 0


def test_aggregate_compte_un_puzzle_dans_chacun_de_ses_themes():
    mesures = [_m(0, 1500, True, True, themes="fork pin short")]

    stats = aggregate(mesures)

    assert stats.par_theme["fork"].total == 1
    assert stats.par_theme["pin"].total == 1
    # 'short' decrit la longueur, pas un motif : il ne doit pas apparaitre.
    assert "short" not in stats.par_theme


def test_aggregate_remplit_les_cases_discordantes_de_mcnemar():
    mesures = [
        _m(0, 1500, True, False),    # reseau bon, recherche mauvaise
        _m(1, 1500, True, False),
        _m(2, 1500, False, True),    # l'inverse
        _m(3, 1500, True, True),      # concordant
    ]

    stats = aggregate(mesures)

    assert stats.mcnemar_b == 2
    assert stats.mcnemar_c == 1


def test_aggregate_exclut_les_erreurs_des_taux_et_les_compte_a_part():
    """Les inclure ferait passer un defaut de donnees pour une faiblesse du
    modele."""
    mesures = [
        _m(0, 1500, True, True),
        _m(1, 1500, False, False, erreur="solution_illegale"),
    ]

    stats = aggregate(mesures)

    assert stats.global_.total == 1
    assert stats.global_.reseau == 1
    assert stats.erreurs == {"solution_illegale": 1}


def test_aggregate_signale_les_positions_au_dela_de_la_limite_de_tt():
    """TT_MAX_MOVES tronque a 128 : au dela, des coups legaux ne peuvent jamais
    etre joues."""
    mesures = [
        _m(0, 1500, True, True, nb_legaux=52),
        _m(1, 1500, True, True, nb_legaux=140),
    ]

    assert aggregate(mesures).au_dela_128 == 1


def test_aggregate_calcule_les_medianes():
    mesures = [
        _m(0, 1500, True, True, p=0.1, part=0.2),
        _m(1, 1500, True, True, p=0.5, part=0.6),
        _m(2, 1500, True, True, p=0.9, part=1.0),
    ]

    stats = aggregate(mesures)

    assert stats.global_.p_correct_median == pytest.approx(0.5)
    assert stats.global_.part_visites_median == pytest.approx(0.6)


def test_aggregate_supporte_une_liste_vide():
    stats = aggregate([])

    assert stats.global_.total == 0
    assert stats.mcnemar_b == 0
    assert stats.erreurs == {}


from evaluation.bench_metrics import format_report

CONTEXTE = {
    "modele": "iter316_dynamic.onnx",
    "iteration": 316,
    "global_step": 19415,
    "simulations": 800,
    "c_puct": 1.4,
    "batch_size": 8,
    "fichier_banc": "data/puzzles_bench.txt",
    "sans_historique": False,
    "duree_totale_s": 3600.0,
    "travailleurs": 16,
}


def _stats_exemple():
    return aggregate([
        _m(0, 1100, True, True, p=0.8, part=0.9),
        _m(1, 1500, False, True, p=0.1, part=0.6),
        _m(2, 2000, True, False, p=0.7, part=0.2, themes="pin short"),
    ])


def test_format_report_contient_le_contexte_et_les_taux():
    texte = format_report(_stats_exemple(), CONTEXTE)

    assert "iter316_dynamic.onnx" in texte
    assert "19415" in texte
    assert "800" in texte
    assert "batch 8" in texte
    assert "McNemar" in texte
    assert "1000-1449" in texte
    assert "avec historique" in texte


def test_format_report_annonce_le_bras_sans_historique():
    contexte = dict(CONTEXTE, sans_historique=True)

    assert "sans historique" in format_report(_stats_exemple(), contexte)


def test_format_report_ne_contient_pas_de_tiret_cadratin():
    """Regle de redaction du projet."""
    assert "—" not in format_report(_stats_exemple(), CONTEXTE)


def test_format_report_signale_les_erreurs_quand_il_y_en_a():
    stats = aggregate([
        _m(0, 1500, True, True),
        _m(1, 1500, False, False, erreur="solution_illegale"),
    ])

    assert "solution_illegale" in format_report(stats, CONTEXTE)


def test_format_report_reste_muet_sur_les_erreurs_quand_il_n_y_en_a_pas():
    assert "solution_illegale" not in format_report(_stats_exemple(), CONTEXTE)


def test_format_report_avertit_au_dela_de_la_limite_de_tt():
    stats = aggregate([_m(0, 1500, True, True, nb_legaux=140)])

    assert "128" in format_report(stats, CONTEXTE)


def test_format_report_supporte_des_stats_vides():
    """Le rapport d'un essai a zero puzzle ne doit pas lever."""
    texte = format_report(aggregate([]), CONTEXTE)

    assert "Banc de puzzles" in texte


def test_parse_bench_line_reads_the_real_bench_file():
    """La lecture doit tenir sur les donnees reelles, pas seulement sur une
    ligne forgee."""
    chemin = os.path.join(RACINE, "..", "data", "puzzles_bench.txt")
    if not os.path.exists(chemin):
        pytest.skip("fichier de banc absent")

    with open(chemin, encoding="utf-8") as f:
        puzzles = [parse_bench_line(i, ligne)
                   for i, ligne in zip(range(20), f)]

    assert len(puzzles) == 20
    assert all(p.coups_uci for p in puzzles)
    assert all(p.solution_uci for p in puzzles)
    assert all(1000 <= p.rating <= 2800 for p in puzzles)
    assert [p.ligne for p in puzzles] == list(range(20))


def test_sous_echantillon_etale_sur_tout_le_fichier():
    """Prendre les premieres lignes biaiserait l'echantillon : le banc est
    ecrit tranche de rating par tranche de rating."""
    import evaluation.puzzle_bench as puzzle_bench

    lignes = list(enumerate(f"l{i}" for i in range(5000)))

    echantillon = puzzle_bench.sous_echantillon(lignes, 4)

    assert len(echantillon) == 4
    index = [i for i, _ in echantillon]
    assert index == sorted(index)
    assert index[0] == 0
    assert index[-1] >= 3750, index
    assert len(set(index)) == 4


def test_sous_echantillon_rend_tout_si_la_limite_depasse():
    import evaluation.puzzle_bench as puzzle_bench

    lignes = list(enumerate("abc"))

    assert puzzle_bench.sous_echantillon(lignes, 10) == lignes
    assert puzzle_bench.sous_echantillon(lignes, 0) == lignes


def test_sous_echantillon_ne_sort_jamais_des_bornes():
    import evaluation.puzzle_bench as puzzle_bench

    lignes = list(enumerate(range(7)))

    for combien in range(1, 8):
        echantillon = puzzle_bench.sous_echantillon(lignes, combien)
        assert len(echantillon) == combien
        assert all(0 <= i < 7 for i, _ in echantillon)


def test_format_report_produit_des_tables_markdown_valides():
    """Garde fou : en retirant la colonne ligne complete j'avais laisse un
    separateur a 5 cellules pour un en-tete a 6, ce qui casse le rendu sans
    qu'aucun autre test ne le voie."""
    texte = format_report(_stats_exemple(), CONTEXTE)

    lignes = texte.split("\n")
    separateurs = [i for i, l in enumerate(lignes) if set(l) <= set("|-") and "-" in l]

    assert separateurs, "aucune table trouvee"
    for i in separateurs:
        cols_sep = lignes[i].count("|") - 1
        cols_tete = lignes[i - 1].count("|") - 1
        assert cols_sep == cols_tete, (
            f"table ligne {i} : separateur {cols_sep} cellules, "
            f"en-tete {cols_tete}")
        # Les lignes de donnees qui suivent doivent avoir le meme compte.
        j = i + 1
        while j < len(lignes) and lignes[j].startswith("|"):
            assert lignes[j].count("|") - 1 == cols_tete, lignes[j]
            j += 1


from evaluation.bench_metrics import (  # noqa: E402
    PuzzleMeasure,
    comparer_historique,
    format_comparaison,
)


def _mesure(ligne, p_correct, value=0.0, reussi=False, coup="a2a3",
            part_visites=0.0, erreur="", reussi_reseau=False):
    return PuzzleMeasure(
        ligne=ligne, rating=1500, themes="fork", plies_historique=10,
        nb_coups_legaux=30, coup_reseau=coup, reussi_reseau=reussi_reseau,
        p_correct_reseau=p_correct, rang_correct_reseau=1, value_reseau=value,
        coup_recherche=coup, reussi_recherche=reussi,
        part_visites_correct=part_visites, duree_s=0.1, erreur=erreur)


def test_comparer_historique_apparie_par_ligne_et_mesure_l_ecart():
    """L'ecart apparie est la grandeur du confondant : negatif tant que la
    presentation sans historique gonfle le prior du coup solution."""
    avec = [
        _mesure(0, 0.05, value=-0.8, reussi=True, coup="a2a3",
                part_visites=0.9),
        _mesure(1, 0.30, value=0.1, reussi=False, coup="b2b3",
                part_visites=0.4),
    ]
    sans = [
        _mesure(0, 0.39, value=0.1, reussi=False, coup="c2c3",
                part_visites=0.2),
        _mesure(1, 0.50, value=0.2, reussi=True, coup="b2b3",
                part_visites=0.6),
    ]

    c = comparer_historique(avec, sans)

    assert c.n == 2
    assert c.non_apparies == 0
    assert c.p_correct_median_avec == pytest.approx(0.175)
    assert c.p_correct_median_sans == pytest.approx(0.445)
    assert c.delta_p_correct_median == pytest.approx(-0.27)
    assert c.delta_part_visites_median == pytest.approx(0.25)
    assert c.delta_value_median == pytest.approx(-0.5)
    assert c.recherche_avec == 1
    assert c.recherche_sans == 1
    assert c.mcnemar_b == 1
    assert c.mcnemar_c == 1
    assert c.accord_coups == pytest.approx(0.5)


def test_comparer_historique_ecarte_les_paires_en_erreur():
    avec = [_mesure(0, 0.1), _mesure(1, 0.2, erreur="solution_illegale")]
    sans = [_mesure(0, 0.2), _mesure(1, 0.3)]

    c = comparer_historique(avec, sans)

    assert c.n == 1
    assert c.non_apparies == 1


def test_comparer_historique_ecarte_une_ligne_absente():
    c = comparer_historique([_mesure(0, 0.1), _mesure(1, 0.2)],
                            [_mesure(0, 0.2)])

    assert c.n == 1
    assert c.non_apparies == 1


def test_comparer_historique_supporte_les_listes_vides():
    c = comparer_historique([], [])

    assert c.n == 0
    assert c.mcnemar_p == 1.0
    assert c.delta_p_correct_median == 0.0


def test_format_comparaison_nomme_le_chiffre_a_suivre():
    c = comparer_historique([_mesure(0, 0.05)], [_mesure(0, 0.39)])

    texte = format_comparaison(c, {})

    assert "Comparaison avec / sans historique" in texte
    assert "ecart median" in texte
    assert "-0.3400" in texte
    assert "Puzzles apparies : **1**" in texte
    assert "—" not in texte


def test_traiter_lot_en_comparaison_mesure_les_deux_passes(monkeypatch):
    """Le mode comparaison mesure chaque puzzle deux fois, avec puis sans
    historique, dans le meme processus travailleur."""
    import evaluation.bench_metrics as bench_metrics
    import evaluation.puzzle_bench as puzzle_bench

    appels = []

    def fausse_mesure(puzzle, policy_fn, search_fn, sans_historique=False,
                      horloge=None):
        appels.append(sans_historique)
        return ("mesure", puzzle.ligne, sans_historique)

    monkeypatch.setattr(bench_metrics, "measure_puzzle", fausse_mesure)
    monkeypatch.setitem(puzzle_bench._ETAT, "policy_fn", object())
    monkeypatch.setitem(puzzle_bench._ETAT, "search_fn", object())
    monkeypatch.setitem(puzzle_bench._ETAT, "sans_historique", False)
    monkeypatch.setitem(puzzle_bench._ETAT, "comparer_historique", True)

    resultat = puzzle_bench.traiter_lot([(3, LIGNE + "\n")])

    assert appels == [False, True]
    assert resultat == [(("mesure", 3, False), ("mesure", 3, True))]


def test_traiter_lot_sans_comparaison_renvoie_des_mesures_simples(monkeypatch):
    import evaluation.bench_metrics as bench_metrics
    import evaluation.puzzle_bench as puzzle_bench

    monkeypatch.setattr(bench_metrics, "measure_puzzle",
                        lambda puzzle, policy_fn, search_fn,
                        sans_historique=False, horloge=None:
                        ("mesure", puzzle.ligne))
    monkeypatch.setitem(puzzle_bench._ETAT, "policy_fn", object())
    monkeypatch.setitem(puzzle_bench._ETAT, "search_fn", object())
    monkeypatch.setitem(puzzle_bench._ETAT, "sans_historique", False)
    monkeypatch.setitem(puzzle_bench._ETAT, "comparer_historique", False)

    assert puzzle_bench.traiter_lot([(3, LIGNE + "\n")]) == [("mesure", 3)]
