"""Noyau de mesure du banc de puzzles.

Toute la logique de scoring vit ici et recoit ses deux acces au reseau par
injection, policy_fn et search_fn, sur le modele du fetcher de
lichess_games.fetch_games. Ce module n'importe donc ni torch ni onnxruntime, et
se teste entierement avec des faux.

Voir docs/etudes/evaluation-puzzles/protocole/2026-08-14-puzzle-bench-design.md
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import collections
import math
import statistics
import time
from dataclasses import dataclass

import chess_engine
from core.move_coding import (
    coords_to_uci,
    decode_move_index,
    encode_move,
    gestion_promo_dame,
    parse_uci_to_coords,
)

TAILLE_POLICY = 4672

# Reprises de build_puzzle_dataset.BENCH_BUCKETS, avec leur etiquette.
TRANCHES = (
    ("1000-1449", 1000, 1449),
    ("1450-1899", 1450, 1899),
    ("1900-2349", 1900, 2349),
    ("2350-2800", 2350, 2800),
)

# TT_MAX_MOVES cote C++ (evaluation_cache.hpp) : au dela, les coups legaux sont tronques
# et ne peuvent jamais etre joues.
LIMITE_TT_MAX_MOVES = 128


@dataclass(frozen=True)
class BenchPuzzle:
    ligne: int
    fen_initiale: str
    coups_uci: list[str]
    solution_uci: list[str]
    rating: int
    themes: str


def parse_bench_line(index: int, ligne: str) -> BenchPuzzle:
    """Inverse de build_puzzle_dataset.format_line.

    Le fichier du banc ne contient pas le PuzzleId : l'index de ligne fait donc
    office d'identifiant, ce qui interdit de reordonner le fichier.
    """
    champs = ligne.rstrip("\n").split("|")
    if len(champs) != 5:
        raise ValueError(f"ligne {index} : {len(champs)} champs au lieu de 5")

    fen, coups, solution, rating, themes = champs
    return BenchPuzzle(
        ligne=index,
        fen_initiale=fen,
        coups_uci=coups.split(),
        solution_uci=solution.split(),
        rating=int(rating),
        themes=themes,
    )


class DonneesInvalides(Exception):
    """La ligne du banc ne decrit pas une position jouable."""

    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


def uci_to_index(board, uci: str) -> int:
    """Index de policy d'un coup UCI dans la position courante.

    La promotion dame est implicite dans la convention AlphaZero. Lichess ecrit
    bien le suffixe, mais gestion_promo_dame couvre le cas ou il manque.
    """
    o_f, o_r, d_f, d_r, promo = parse_uci_to_coords(uci)
    promo = gestion_promo_dame(board, o_f, o_r, d_r, promo)
    is_black = board.turn == chess_engine.Color.BLACK
    return encode_move(o_f, o_r, d_f, d_r, promo, is_black)


def index_to_uci(board, index: int) -> str:
    is_black = board.turn == chess_engine.Color.BLACK
    return coords_to_uci(*decode_move_index(board, index, is_black))


def charger_position(puzzle: BenchPuzzle, sans_historique: bool = False):
    """Rejoue l'historique cote C++, ce qui construit les plans d'historique.

    On ne reconstruit jamais le tenseur en Python : le banc reste ainsi
    insensible a un changement de representation.
    """
    board = chess_engine.Chessboard()
    board.load_fen(puzzle.fen_initiale)
    for coup in puzzle.coups_uci:
        if not board.move_piece_uci(coup):
            raise DonneesInvalides("historique_illegal")

    if sans_historique:
        # Recharger la FEN atteinte donne la meme position, sans le passe.
        atteinte = board.to_fen()
        board = chess_engine.Chessboard()
        board.load_fen(atteinte)
    return board


@dataclass(frozen=True)
class PuzzleMeasure:
    ligne: int
    rating: int
    themes: str
    plies_historique: int
    nb_coups_legaux: int
    coup_reseau: str
    reussi_reseau: bool
    p_correct_reseau: float
    rang_correct_reseau: int
    value_reseau: float
    coup_recherche: str
    reussi_recherche: bool
    part_visites_correct: float
    duree_s: float
    erreur: str
    simulations_recherche: int = 0
    depassement_s: float = 0.0


def _mesure(puzzle: BenchPuzzle, duree: float, erreur: str = "", *,
            nb_coups_legaux: int = 0,
            coup_reseau: str = "", reussi_reseau: bool = False,
            p_correct: float = 0.0, rang: int = 0, value: float = 0.0,
            coup_recherche: str = "", reussi_recherche: bool = False,
            part_visites: float = 0.0, simulations_recherche: int = 0,
            depassement_s: float = 0.0) -> PuzzleMeasure:
    return PuzzleMeasure(
        ligne=puzzle.ligne, rating=puzzle.rating, themes=puzzle.themes,
        plies_historique=len(puzzle.coups_uci),
        nb_coups_legaux=nb_coups_legaux,
        coup_reseau=coup_reseau, reussi_reseau=reussi_reseau,
        p_correct_reseau=float(p_correct), rang_correct_reseau=rang,
        value_reseau=float(value),
        coup_recherche=coup_recherche, reussi_recherche=reussi_recherche,
        part_visites_correct=float(part_visites),
        duree_s=duree, simulations_recherche=simulations_recherche,
        depassement_s=float(depassement_s), erreur=erreur,
    )


def measure_puzzle(puzzle: BenchPuzzle, policy_fn, search_fn,
                   sans_historique: bool = False,
                   horloge=time.perf_counter) -> PuzzleMeasure:
    """Mesure un puzzle : reseau seul, puis recherche, sur le PREMIER coup.

    Seul le premier coup est score, et non la ligne complete. Deux raisons : le
    self-play ne traite specialement que le premier coup d'une position de
    puzzle (TACTICAL_FIRST_MOVE_SIMS), donc le banc mesure exactement ce que
    l'entrainement optimise ; et cela divise le cout par 2,17, le nombre moyen
    de coups solveur par ligne.

    policy_fn(board) -> (probabilites sur les index legaux, value)
    search_fn(board) -> sequence de 4672 flottants, visites normalisees

    Les comparaisons portent sur des index de policy et jamais sur des chaines
    UCI : l'encodage est injectif sur les coups legaux, alors que deux mises en
    forme UCI du meme coup peuvent differer.
    """
    debut = horloge()

    if not puzzle.solution_uci:
        return _mesure(puzzle, horloge() - debut, "solution_vide")

    try:
        board = charger_position(puzzle, sans_historique)
    except DonneesInvalides as exc:
        return _mesure(puzzle, horloge() - debut, exc.cause)

    legaux = set(board.get_legal_move_indices())
    nb_coups_legaux = len(legaux)
    idx_solution = uci_to_index(board, puzzle.solution_uci[0])
    if idx_solution not in legaux:
        return _mesure(puzzle, horloge() - debut, "solution_illegale",
                       nb_coups_legaux=nb_coups_legaux)

    # --- Reseau seul : une inference, aucune recherche ---
    probs, value = policy_fn(board)
    p_correct = probs.get(idx_solution, 0.0)
    rang = 1 + sum(1 for p in probs.values() if p > p_correct)
    idx_reseau = max(probs, key=probs.get)

    # --- Recherche : le meme reseau avec le MCTS ---
    resultat = search_fn(board)
    # Le mode temps renvoie aussi le bilan de la fenetre (simulations
    # terminees, depassement du dernier appel complet).
    if isinstance(resultat, tuple):
        pi, bilan = resultat
    else:
        pi, bilan = resultat, {}
    idx_recherche = max(range(TAILLE_POLICY), key=lambda i: pi[i])

    return _mesure(
        puzzle, horloge() - debut, "",
        nb_coups_legaux=nb_coups_legaux,
        coup_reseau=index_to_uci(board, idx_reseau),
        reussi_reseau=(idx_reseau == idx_solution),
        p_correct=p_correct, rang=rang, value=value,
        coup_recherche=index_to_uci(board, idx_recherche),
        reussi_recherche=(idx_recherche == idx_solution),
        part_visites=float(pi[idx_solution]),
        simulations_recherche=int(bilan.get("simulations", 0)),
        depassement_s=float(bilan.get("depassement_s", 0.0)))


def wilson(succes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson.

    Preferable a l'intervalle normal sur des taux proches de 0 ou de 1, et sur
    de petits effectifs, ou l'intervalle normal peut sortir de [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)

    p = succes / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    demi = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Test de McNemar avec correction de continuite.

    b : le reseau seul avait trouve, la recherche a perdu.
    c : le reseau seul avait manque, la recherche a trouve.

    Le test est symetrique en b et c : il dit s'il y a un ecart, pas dans quel
    sens. Ce sont b et c, rapportes separement, qui portent le sens.

    La p-valeur bilaterale d'un chi2 a un degre de liberte vaut
    erfc(sqrt(chi2 / 2)), ce qui evite une dependance a scipy.
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)

    chi2 = max(0.0, abs(b - c) - 1.0) ** 2 / n
    return (chi2, math.erfc(math.sqrt(chi2 / 2.0)))


@dataclass(frozen=True)
class Taux:
    total: int
    reseau: int
    recherche: int
    p_correct_median: float
    part_visites_median: float


@dataclass(frozen=True)
class BenchStats:
    global_: Taux
    par_tranche: dict
    par_theme: dict
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_chi2: float
    mcnemar_p: float
    erreurs: dict
    au_dela_128: int


def _taux(mesures: list) -> Taux:
    if not mesures:
        return Taux(0, 0, 0, 0.0, 0.0)

    return Taux(
        total=len(mesures),
        reseau=sum(1 for m in mesures if m.reussi_reseau),
        recherche=sum(1 for m in mesures if m.reussi_recherche),
        p_correct_median=statistics.median(m.p_correct_reseau for m in mesures),
        part_visites_median=statistics.median(
            m.part_visites_correct for m in mesures),
    )


def aggregate(mesures: list) -> BenchStats:
    """Agrege les mesures : global, par tranche de rating, par theme.

    Les puzzles en erreur sont exclus des taux et comptes a part : les inclure
    ferait passer un defaut de donnees pour une faiblesse du modele.
    """
    from evaluation.build_puzzle_dataset import TACTICAL_THEMES

    erreurs: collections.Counter = collections.Counter(
        m.erreur for m in mesures if m.erreur)
    valides = [m for m in mesures if not m.erreur]

    par_tranche = {
        etiquette: _taux([m for m in valides if bas <= m.rating <= haut])
        for etiquette, bas, haut in TRANCHES
    }

    par_theme = {}
    for theme in sorted(TACTICAL_THEMES):
        lot = [m for m in valides if theme in m.themes.split()]
        if lot:
            par_theme[theme] = _taux(lot)

    b = sum(1 for m in valides if m.reussi_reseau and not m.reussi_recherche)
    c = sum(1 for m in valides if not m.reussi_reseau and m.reussi_recherche)
    chi2, p = mcnemar(b, c)

    return BenchStats(
        global_=_taux(valides),
        par_tranche=par_tranche,
        par_theme=par_theme,
        mcnemar_b=b, mcnemar_c=c, mcnemar_chi2=chi2, mcnemar_p=p,
        erreurs=dict(erreurs),
        au_dela_128=sum(1 for m in valides
                        if m.nb_coups_legaux > LIMITE_TT_MAX_MOVES),
    )


_EN_TETE_TABLE = (
    "| | n | Reseau seul % | Recherche % "
    "| p med. du bon coup | part de visites med. |\n"
    "|---|---|---|---|---|---|"
)


def _ligne_taux(etiquette: str, t: Taux) -> str:
    """Une ligne de table, taux suivis de leur intervalle de Wilson."""
    if t.total == 0:
        return f"| {etiquette} | 0 | | | | |"

    br, hr = wilson(t.reseau, t.total)
    bs, hs = wilson(t.recherche, t.total)
    return (
        f"| {etiquette} | {t.total} "
        f"| {100.0 * t.reseau / t.total:.1f} "
        f"({100 * br:.1f} a {100 * hr:.1f}) "
        f"| {100.0 * t.recherche / t.total:.1f} "
        f"({100 * bs:.1f} a {100 * hs:.1f}) "
        f"| {t.p_correct_median:.3f} | {t.part_visites_median:.3f} |"
    )


def format_report(stats: BenchStats, contexte: dict) -> str:
    """Rapport markdown.

    Les taux sont suivis de leur intervalle de Wilson a 95 pour cent, sans quoi
    deux points d'ecart se liraient comme un ecart.
    """
    bras = "sans historique" if contexte["sans_historique"] else "avec historique"
    budget = contexte.get("budget_label") or (
        f"{contexte.get('simulations')} simulations")

    lignes = [
        "# Banc de puzzles : resultats",
        "",
        f"Modele : `{contexte['modele']}`, iteration {contexte['iteration']}, "
        f"global_step {contexte['global_step']}",
        f"Banc : `{contexte['fichier_banc']}`, bras {bras}",
        f"Recherche : {budget}, "
        f"c_puct {contexte['c_puct']}, batch {contexte.get('batch_size', 0)}, "
        f"TT {('legacy' if contexte.get('cache_history_depth', 0) == -1 else 'h' + str(contexte.get('cache_history_depth', 0)))}, "
        f"{contexte['travailleurs']} travailleurs, "
        f"{contexte.get('search_workers', 1)} workers de recherche, "
        f"{contexte.get('accelerateur', 'CPU')}",
        f"Duree : {contexte['duree_totale_s'] / 60.0:.1f} min",
        "",
        "Seul le PREMIER coup est score, celui ou il y a une tactique a trouver.",
        "C'est aussi le seul coup que le self-play traite specialement, donc le",
        "banc mesure ce que l'entrainement optimise.",
        "",
        "La colonne reseau seul est une inference sans aucune recherche : c'est la",
        "policy brute, la grandeur qui s'effondrait sur les puzzles prives",
        "d'historique. La colonne recherche est le meme reseau avec le MCTS.",
        "",
        "## Global",
        "",
        _EN_TETE_TABLE,
        _ligne_taux("global", stats.global_),
        "",
        "## Par tranche de rating",
        "",
        _EN_TETE_TABLE,
    ]
    lignes += [_ligne_taux(etiquette, stats.par_tranche[etiquette])
               for etiquette, _, _ in TRANCHES]

    lignes += [
        "",
        "## Par theme tactique",
        "",
        "Un puzzle portant plusieurs themes compte dans chacun : la ventilation",
        "est multi-etiquettes et ne somme pas au total.",
        "",
        _EN_TETE_TABLE,
    ]
    lignes += [_ligne_taux(theme, t) for theme, t in
               sorted(stats.par_theme.items(), key=lambda kv: -kv[1].total)]

    lignes += [
        "",
        "## Reseau seul contre recherche, comparaison appariee",
        "",
        "Les deux colonnes portent sur les memes puzzles, donc McNemar",
        "s'applique. La case qui compte est la premiere : si elle est grosse, la",
        "recherche detruit des tactiques que la policy voyait deja, ce qui est un",
        "diagnostic tout autre qu'un reseau faible.",
        "",
        f"- Reseau bon, recherche mauvaise : **{stats.mcnemar_b}**",
        f"- Reseau mauvais, recherche bonne : **{stats.mcnemar_c}**",
        f"- McNemar : chi2 = {stats.mcnemar_chi2:.2f}, p = {stats.mcnemar_p:.2e}",
    ]

    if stats.erreurs:
        lignes += ["", "## Erreurs de donnees", ""]
        lignes += [f"- `{cause}` : {n}"
                   for cause, n in sorted(stats.erreurs.items())]
        lignes += [
            "",
            "Ces puzzles sont exclus des taux ci-dessus. Les compter dedans "
            "ferait passer un defaut de donnees pour une faiblesse du modele.",
        ]

    if stats.au_dela_128:
        lignes += [
            "",
            f"Attention : {stats.au_dela_128} positions depassent "
            f"{LIMITE_TT_MAX_MOVES} coups legaux, or TT_MAX_MOVES tronque a "
            f"{LIMITE_TT_MAX_MOVES} et les coups au dela ne peuvent jamais "
            "etre joues.",
        ]

    return "\n".join(lignes) + "\n"


@dataclass(frozen=True)
class ComparaisonHistorique:
    """Ecart apparie entre les deux presentations des memes puzzles."""
    n: int
    p_correct_median_avec: float
    p_correct_median_sans: float
    delta_p_correct_median: float
    delta_part_visites_median: float
    value_median_avec: float
    value_median_sans: float
    delta_value_median: float
    recherche_avec: int
    recherche_sans: int
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_chi2: float
    mcnemar_p: float
    accord_coups: float
    non_apparies: int


def comparer_historique(mesures_avec: list,
                        mesures_sans: list) -> ComparaisonHistorique:
    """Ecart entre les passes avec et sans historique, puzzle par puzzle.

    Le confondant « pas d'historique = position tactique » gonfle le prior du
    coup solution et la value quand l'historique est vide. L'ecart apparie
    `avec - sans` est donc fortement negatif tant que le raccourci est present,
    et doit converger vers zero quand il est desappris. C'est le chiffre a
    suivre d'une campagne a l'autre ; les taux de resolution, la part de
    visites, la value et l'accord des coups de recherche l'accompagnent pour
    eviter de conclure sur le seul prior.

    Les puzzles en erreur d'un cote ou de l'autre sont ecartes et comptes.
    """
    par_ligne_sans = {m.ligne: m for m in mesures_sans}
    paires = []
    non_apparies = 0
    for avec in mesures_avec:
        sans = par_ligne_sans.get(avec.ligne)
        if sans is None or avec.erreur or sans.erreur:
            non_apparies += 1
            continue
        paires.append((avec, sans))

    if not paires:
        return ComparaisonHistorique(
            n=0, p_correct_median_avec=0.0, p_correct_median_sans=0.0,
            delta_p_correct_median=0.0, delta_part_visites_median=0.0,
            value_median_avec=0.0, value_median_sans=0.0,
            delta_value_median=0.0, recherche_avec=0, recherche_sans=0,
            mcnemar_b=0, mcnemar_c=0, mcnemar_chi2=0.0, mcnemar_p=1.0,
            accord_coups=0.0, non_apparies=non_apparies)

    b = sum(1 for a, s in paires
            if a.reussi_recherche and not s.reussi_recherche)
    c = sum(1 for a, s in paires
            if not a.reussi_recherche and s.reussi_recherche)
    chi2, p = mcnemar(b, c)

    return ComparaisonHistorique(
        n=len(paires),
        p_correct_median_avec=statistics.median(
            a.p_correct_reseau for a, _ in paires),
        p_correct_median_sans=statistics.median(
            s.p_correct_reseau for _, s in paires),
        delta_p_correct_median=statistics.median(
            a.p_correct_reseau - s.p_correct_reseau for a, s in paires),
        delta_part_visites_median=statistics.median(
            a.part_visites_correct - s.part_visites_correct for a, s in paires),
        value_median_avec=statistics.median(a.value_reseau for a, _ in paires),
        value_median_sans=statistics.median(s.value_reseau for _, s in paires),
        delta_value_median=statistics.median(
            a.value_reseau - s.value_reseau for a, s in paires),
        recherche_avec=sum(1 for a, _ in paires if a.reussi_recherche),
        recherche_sans=sum(1 for _, s in paires if s.reussi_recherche),
        mcnemar_b=b, mcnemar_c=c, mcnemar_chi2=chi2, mcnemar_p=p,
        accord_coups=sum(1 for a, s in paires
                         if a.coup_recherche == s.coup_recherche) / len(paires),
        non_apparies=non_apparies,
    )


def format_comparaison(comparaison: ComparaisonHistorique,
                       contexte: dict) -> str:
    """Section de comparaison, avec le chiffre a suivre en tete."""
    ecartes = (f", {comparaison.non_apparies} ecartes"
               if comparaison.non_apparies else "")
    lignes = [
        "",
        "## Comparaison avec / sans historique (passe appariee)",
        "",
        "Meme modele, memes puzzles, meme budget : la seule difference est la",
        "presentation. Tant que le raccourci « pas d'historique = position",
        "tactique » est present, la passe sans historique gonfle le prior du",
        "coup solution et la value. Le chiffre a suivre d'une campagne a",
        "l'autre est l'ecart median du prior : il doit converger vers zero",
        "quand le raccourci est desappris. Les autres lignes evitent de",
        "conclure sur le seul prior.",
        "",
        f"- Puzzles apparies : **{comparaison.n}**{ecartes}",
        f"- Prior median du coup solution : avec "
        f"{comparaison.p_correct_median_avec:.3f}, sans "
        f"{comparaison.p_correct_median_sans:.3f}, **ecart median "
        f"{comparaison.delta_p_correct_median:+.4f}**",
        f"- Part de visites mediane du coup solution : ecart median "
        f"{comparaison.delta_part_visites_median:+.4f}",
        f"- Value du reseau : avec {comparaison.value_median_avec:+.3f}, sans "
        f"{comparaison.value_median_sans:+.3f}, ecart median "
        f"{comparaison.delta_value_median:+.3f}",
        f"- Recherche : avec {comparaison.recherche_avec}/{comparaison.n}, "
        f"sans {comparaison.recherche_sans}/{comparaison.n}",
        f"- McNemar apparie : reussi avec seul {comparaison.mcnemar_b}, reussi "
        f"sans seul {comparaison.mcnemar_c}, chi2 = "
        f"{comparaison.mcnemar_chi2:.2f}, p = {comparaison.mcnemar_p:.2e}",
        f"- Accord des coups de recherche entre les deux passes : "
        f"{100.0 * comparaison.accord_coups:.1f} %",
        "",
        "Les deux rapports complets suivent : bras avec historique, puis bras",
        "sans historique.",
        "",
    ]
    return "\n".join(lignes) + "\n"
