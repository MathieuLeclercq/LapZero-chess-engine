"""Tests couples au moteur : ils ont besoin du module compile et d'un modele.

Le test d'accord du softmax est le seul garde-fou du choix d'implementation
retenu, a savoir recalculer le prior en Python plutot que d'ajouter un binding
C++. Il compare la meme grandeur des deux cotes, pas un substitut.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine
from evaluation.bench_metrics import charger_position, parse_bench_line

BANC = RACINE / "data" / "puzzles_bench.txt"
CHECKPOINT = (RACINE / "data" / "checkpoints"
              / "2026_04_23_23h25_iter316_unsupervised.pt")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists() or not BANC.exists(),
    reason="checkpoint ou fichier de banc absent")


@pytest.fixture(scope="module")
def modele(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    dossier = tmp_path_factory.mktemp("onnx")
    return puzzle_bench.resoudre_modele(CHECKPOINT, dossier)


@pytest.fixture(scope="module")
def session(modele):
    import onnxruntime as ort

    chemin, _ = modele
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(chemin), options,
                                providers=["CPUExecutionProvider"])


def _premier_puzzle():
    with open(BANC, encoding="utf-8") as f:
        return parse_bench_line(0, next(f))


def test_export_recupere_les_metadonnees_du_checkpoint(modele):
    _, meta = modele

    assert meta["iteration"] == 316
    assert meta["global_step"] == 19415


def test_export_deduit_l_architecture_du_checkpoint(modele):
    """La coder en dur casserait silencieusement sur un modele de taille
    differente."""
    _, meta = modele

    assert meta["num_res_blocks"] == 10
    assert meta["num_filters"] == 128


def test_la_somme_des_probabilites_vaut_un(session):
    import evaluation.puzzle_bench as puzzle_bench

    board = charger_position(_premier_puzzle())
    probs, value = puzzle_bench.faire_policy_fn(session)(board)

    assert set(probs) == set(board.get_legal_move_indices())
    assert sum(probs.values()) == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0


def test_le_softmax_python_egale_le_prior_du_cpp(modele, session):
    """Le C++ applique un softmax sur les 4672 sorties (onnx_evaluator.cpp)
    puis renormalise sur les coups legaux (mcts.cpp), ce qui est identique a un
    softmax sur les seuls coups legaux. L'egalite est mathematique, on la
    verifie quand meme : c'est le seul risque de l'approche retenue.
    """
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = modele
    board = charger_position(_premier_puzzle())
    probs, _ = puzzle_bench.faire_policy_fn(session)(board)

    # Cote C++ : step_analysis renseigne m_analysis_root, et
    # get_analysis_results expose le prior des enfants visites.
    evaluateur = chess_engine.ONNXEvaluator(str(chemin), False)
    mcts = chess_engine.MCTS(evaluateur, puzzle_bench.TAILLE_TT)
    mcts.step_analysis(board, 400, 1.4)
    stats = mcts.get_analysis_results()

    assert stats, "aucun coup visite, le test ne verifierait rien"

    for s in stats:
        assert s.move_idx in probs, f"index {s.move_idx} absent du softmax Python"
        assert s.prior == pytest.approx(probs[s.move_idx], rel=1e-4, abs=1e-6)

    assert len(stats) >= 5, f"seulement {len(stats)} priors compares"


def test_la_recherche_renvoie_une_distribution_de_visites(modele):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = modele
    evaluateur = chess_engine.ONNXEvaluator(str(chemin), False)
    search_fn = puzzle_bench.faire_search_fn(evaluateur, 64, 1.4)

    board = charger_position(_premier_puzzle())
    pi = search_fn(board)

    assert len(pi) == 4672
    assert sum(pi) == pytest.approx(1.0)
    legaux = set(board.get_legal_move_indices())
    assert all(pi[i] == 0.0 for i in range(4672) if i not in legaux)


def test_la_recherche_transmet_batch_et_workers(monkeypatch):
    """Le banc doit exercer le chemin batche et multicœur, pas seulement les
    exposer en CLI."""
    import evaluation.puzzle_bench as puzzle_bench

    appels = []

    class FauxMCTS:
        def __init__(self, evaluateur, taille_tt, cache_history_depth):
            assert taille_tt == puzzle_bench.TAILLE_TT
            appels.append(("constructeur", cache_history_depth))

        def set_fixed_batch(self, enabled):
            appels.append(("fixed", enabled))

        def mcts_search(self, board, simulations, c_puct, bruit,
                        batch_size, worker_count):
            appels.append((board, simulations, c_puct, bruit, batch_size,
                           worker_count))
            return [1.0]

    monkeypatch.setattr(puzzle_bench.chess_engine, "MCTS", FauxMCTS)
    plateau = object()

    assert puzzle_bench.faire_search_fn(
        object(), 700, 1.4, 8, cache_history_depth=0,
        worker_count=4)(plateau) == [1.0]
    assert appels == [
        ("constructeur", 0),
        ("fixed", True),
        (plateau, 700, 1.4, False, 8, 4),
    ]


def test_la_recherche_transmet_le_reglage_de_divergence(monkeypatch):
    import evaluation.puzzle_bench as puzzle_bench

    appels = []

    class FauxMCTS:
        def __init__(self, *args):
            pass

        def set_fixed_batch(self, enabled):
            appels.append(("fixed", enabled))

        def set_tuning(self, virtual_loss, fpu_reduction,
                       collision_attempt_factor):
            appels.append(("tuning", virtual_loss, fpu_reduction,
                           collision_attempt_factor))

        def mcts_search(self, *args):
            return [1.0]

    monkeypatch.setattr(puzzle_bench.chess_engine, "MCTS", FauxMCTS)

    puzzle_bench.faire_search_fn(
        object(), 700, 1.4, 8, worker_count=4, virtual_loss=2,
        fpu_reduction=0.45)(object())

    assert appels == [("fixed", True), ("tuning", 2, 0.45, 4)]


def test_la_taille_de_tt_reste_petite():
    """Au defaut de 2 097 143 entrees a 1040 octets, chaque MCTS reserverait
    2,03 Gio, soit 32,5 Gio a 16 travailleurs pour 31,4 Gio de RAM. Un MCTS
    neuf est cree a chaque recherche."""
    import evaluation.puzzle_bench as puzzle_bench

    assert puzzle_bench.TAILLE_TT <= 65536


def test_main_de_bout_en_bout_sur_un_petit_echantillon(tmp_path, monkeypatch):
    """Execute main() sur 6 puzzles et 8 simulations : verifie l'assemblage,
    pas la qualite du modele."""
    import csv as csv_mod
    import json
    import evaluation.puzzle_bench as puzzle_bench
    from evaluation.puzzle_bench import chemin_sidecar

    sortie_csv = tmp_path / "res.csv"
    sortie_rapport = tmp_path / "rapport.md"

    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py",
        "--model", str(CHECKPOINT),
        "--banc", str(BANC),
        "--limite", "6",
        "--simulations", "8",
        "--travailleurs", "2",
        "--cache-history-depth", "3",
        "--dossier-onnx", str(tmp_path / "onnx"),
        "--out-csv", str(sortie_csv),
        "--out-rapport", str(sortie_rapport),
    ])

    assert puzzle_bench.main() == 0

    with open(sortie_csv, encoding="utf-8", newline="") as f:
        lignes = list(csv_mod.DictReader(f))

    assert len(lignes) == 6
    assert list(lignes[0]) == list(puzzle_bench.CHAMPS_CSV)
    # Le sous-echantillon est a pas regulier, donc etale sur tout le fichier,
    # et le CSV reste trie par index de ligne.
    index = [int(l["ligne"]) for l in lignes]
    assert index == sorted(index)
    assert index[-1] > 100, "l'echantillon ne couvre qu'un debut de fichier"
    # Aucune erreur de donnees : second controle de la correction du pipeline,
    # independant des tests de build_puzzle_dataset.
    assert all(l["erreur"] == "" for l in lignes), [
        l["erreur"] for l in lignes if l["erreur"]]
    # Le premier coup est toujours renseigne quand il n'y a pas d'erreur.
    for l in lignes:
        assert l["coup_recherche"], l
        assert l["reussi_recherche"] in ("True", "False")

    rapport = sortie_rapport.read_text(encoding="utf-8")
    assert "Banc de puzzles" in rapport
    assert "McNemar" in rapport
    assert "h3" in rapport

    # Le sidecar porte le contexte complet, sans quoi deux campagnes ne
    # seraient pas comparables sans ambiguite.
    sidecar = json.loads(
        chemin_sidecar(sortie_csv).read_text(encoding="utf-8"))
    assert sidecar["search_workers"] == 1
    assert sidecar["accelerateur"] == "CPU"
    assert sidecar["critere"] == "premier_coup_recherche"
    assert len(sidecar["modele_sha256"]) == 64
    assert len(sidecar["banc_sha256"]) == 64
    assert sidecar["simulations"] == 8


def test_le_binding_self_play_avec_compteurs_ne_plante_pas(modele):
    """La variante de diagnostic renvoie un tuple, ses compteurs, et l'acces
    aux tenseurs reste valide apres le retour.

    Regression : construire un py::tuple dans une lambda dont le GIL est
    relache plantait en access violation juste apres la generation.
    """
    chemin, _ = modele
    evaluateur = chess_engine.ONNXEvaluator(str(chemin), False)
    parties, stats = chess_engine.generate_self_play_games_with_stats(
        evaluateur, 4, 20, 5, 4, 0.25, 8192,
        str(RACINE / "training_data" / "puzzles_train.txt"))

    assert len(parties) == 4
    assert stats.games_started == 4
    assert stats.games_completed == 4
    assert stats.active_slots == 0
    assert stats.new_plies > 0
    assert stats.replayed_plies >= 0
    assert parties[0].state_tensors.shape[1:] == (119, 8, 8)


def _horloge_factice(pas_s: float = 0.02):
    temps = [0.0]

    def horloge():
        valeur = temps[0]
        temps[0] += pas_s
        return valeur

    return horloge


def test_faire_search_fn_en_mode_temps_rapporte_le_bilan(monkeypatch):
    """Le mode temps avance par tranches, garde le meme MCTS entre deux
    puzzles, et rapporte les simulations terminees et le depassement du
    dernier appel complet."""
    import evaluation.puzzle_bench as puzzle_bench

    instances = []

    class FauxMCTS:
        def __init__(self, evaluateur, taille_tt, depth):
            instances.append(self)
            self.sims = 0
            self.clears = 0

        def set_fixed_batch(self, enabled):
            pass

        def reset_analysis(self):
            self.sims = 0

        def clear_evaluation_cache(self):
            self.clears += 1

        def reset_counters(self):
            self.sims = 0

        def step_analysis(self, board, simulations, c_puct, batch_size,
                          worker_count):
            self.sims += simulations

        def get_analysis_results(self):
            return [type("S", (), dict(move_idx=1, visits=self.sims))()]

        def get_counters(self):
            return type("C", (), dict(completed_simulations=self.sims))()

    monkeypatch.setattr(puzzle_bench.chess_engine, "MCTS", FauxMCTS)
    search_fn = puzzle_bench.faire_search_fn(
        object(), 100, 1.4, 8, cache_history_depth=0, worker_count=4,
        search_seconds=0.05, horloge=_horloge_factice())

    pi, bilan = search_fn(object())

    assert bilan["simulations"] == 24
    assert bilan["depassement_s"] == pytest.approx(0.01)
    assert sum(pi) == pytest.approx(1.0)
    assert pi[1] == pytest.approx(1.0)

    # Deuxieme puzzle dans le meme processus : l'objet persiste, arbre et TT
    # sont remis a froid hors fenetre.
    search_fn(object())
    assert len(instances) == 1
    assert instances[0].clears == 2


def test_main_transmet_search_workers_sans_processus_supplementaires(
        tmp_path, monkeypatch):
    """Huit workers de recherche sont des threads C++ d'un seul processus :
    ils ne doivent pas devenir huit processus Python."""
    import json
    import multiprocessing
    import evaluation.puzzle_bench as puzzle_bench

    onnx = tmp_path / "modele.onnx"
    onnx.write_bytes(b"onnx factice")
    csv = tmp_path / "res.csv"
    rapport = tmp_path / "rapport.md"
    processus = []

    class FauxPool:
        def __init__(self, processes, initializer=None, initargs=()):
            processus.append(processes)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, fonction, lots):
            return iter(())

    monkeypatch.setattr(multiprocessing, "Pool", FauxPool)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (onnx, {}))
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(onnx), "--banc", str(BANC),
        "--limite", "2", "--simulations", "8", "--travailleurs", "1",
        "--batch-size", "8", "--search-workers", "8",
        "--out-csv", str(csv), "--out-rapport", str(rapport),
    ])

    assert puzzle_bench.main() == 0
    assert processus == [1]
    assert "8 workers de recherche" in rapport.read_text(encoding="utf-8")
    from evaluation.puzzle_bench import chemin_sidecar
    sidecar = json.loads(chemin_sidecar(csv).read_text(encoding="utf-8"))
    assert sidecar["search_workers"] == 8
    assert sidecar["cache_history_depth"] == 0


def test_main_en_gpu_reste_dans_un_seul_processus(tmp_path, monkeypatch):
    """La jambe GPU ne cree aucun Pool et passe les workers de recherche a
    l'initialiseur unique."""
    import multiprocessing
    import evaluation.puzzle_bench as puzzle_bench

    onnx = tmp_path / "modele.onnx"
    onnx.write_bytes(b"onnx factice")
    rapport = tmp_path / "rapport.md"
    appels = []

    def pool_interdit(*args, **kwargs):
        raise AssertionError("aucun Pool ne doit etre cree en mode GPU")

    def faux_initialiser_gpu(*args, **kwargs):
        appels.append((args, kwargs))

    monkeypatch.setattr(multiprocessing, "Pool", pool_interdit)
    monkeypatch.setattr(puzzle_bench, "initialiser_gpu", faux_initialiser_gpu)
    monkeypatch.setattr(puzzle_bench, "traiter_lot", lambda lot: [])
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (onnx, {}))
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(onnx), "--banc", str(BANC),
        "--limite", "2", "--simulations", "8", "--travailleurs", "1",
        "--batch-size", "8", "--search-workers", "4", "--gpu",
        "--out-csv", str(tmp_path / "r.csv"), "--out-rapport", str(rapport),
    ])

    assert puzzle_bench.main() == 0
    assert len(appels) == 1
    args, _kwargs = appels[0]
    assert args[0] == str(onnx)
    assert args[6] == 4          # search_workers
    assert args[7] is None       # mode simulations fixes
    assert "GPU" in rapport.read_text(encoding="utf-8")


def test_main_refuse_le_gpu_avec_plusieurs_processus(tmp_path, monkeypatch):
    import evaluation.puzzle_bench as puzzle_bench

    onnx = tmp_path / "modele.onnx"
    onnx.write_bytes(b"onnx factice")
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (onnx, {}))
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(onnx), "--banc", str(BANC),
        "--gpu", "--travailleurs", "2",
        "--out-csv", str(tmp_path / "r.csv"),
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    with pytest.raises(SystemExit):
        puzzle_bench.main()


def test_main_refuse_simulations_explicites_et_mode_temps(
        tmp_path, monkeypatch):
    import evaluation.puzzle_bench as puzzle_bench

    onnx = tmp_path / "modele.onnx"
    onnx.write_bytes(b"onnx factice")
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (onnx, {}))
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(onnx), "--banc", str(BANC),
        "--simulations", "32", "--search-seconds", "1.5",
        "--out-csv", str(tmp_path / "r.csv"),
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    with pytest.raises(SystemExit):
        puzzle_bench.main()


def test_main_retourne_un_echec_de_campagne_si_un_puzzle_est_en_erreur(
        tmp_path, monkeypatch, modele):
    """Une campagne avec des lignes en erreur doit sortir en echec, pour
    qu'aucun CSV partiellement exploitable ne soit compare en silence."""
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = modele
    banc = tmp_path / "banc_erreur.txt"
    banc.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        "|e2e4 e2e4|b8c6|1500|fork short\n",
        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(CHECKPOINT),
        "--dossier-onnx", str(chemin.parent),
        "--banc", str(banc), "--limite", "1", "--simulations", "8",
        "--travailleurs", "1",
        "--out-csv", str(tmp_path / "res.csv"),
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert puzzle_bench.main() == 1


def test_main_refuse_un_fichier_de_banc_absent(tmp_path, monkeypatch):
    import evaluation.puzzle_bench as puzzle_bench

    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py",
        "--model", str(CHECKPOINT),
        "--banc", str(tmp_path / "absent.txt"),
        "--out-csv", str(tmp_path / "res.csv"),
        "--out-rapport", str(tmp_path / "rapport.md"),
    ])

    assert puzzle_bench.main() == 2


def test_main_en_comparaison_ecrit_les_deux_csv_et_l_ecart(
        tmp_path, monkeypatch):
    """Le mode comparaison produit les deux passes et la section d'ecart, sur
    les memes puzzles dans le meme ordre."""
    import csv as csv_mod
    import json
    import evaluation.puzzle_bench as puzzle_bench
    from evaluation.puzzle_bench import chemin_sidecar

    sortie_csv = tmp_path / "res.csv"
    sortie_rapport = tmp_path / "rapport.md"

    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py",
        "--model", str(CHECKPOINT),
        "--banc", str(BANC),
        "--limite", "2",
        "--simulations", "8",
        "--travailleurs", "1",
        "--comparer-historique",
        "--dossier-onnx", str(tmp_path / "onnx"),
        "--out-csv", str(sortie_csv),
        "--out-rapport", str(sortie_rapport),
    ])

    assert puzzle_bench.main() == 0

    with open(sortie_csv, encoding="utf-8", newline="") as f:
        avec = list(csv_mod.DictReader(f))
    sans_csv = tmp_path / "res_sans.csv"
    with open(sans_csv, encoding="utf-8", newline="") as f:
        sans = list(csv_mod.DictReader(f))

    assert len(avec) == 2
    assert len(sans) == 2
    assert [l["ligne"] for l in avec] == [l["ligne"] for l in sans]

    sidecar_avec = json.loads(
        chemin_sidecar(sortie_csv).read_text(encoding="utf-8"))
    sidecar_sans = json.loads(
        chemin_sidecar(sans_csv).read_text(encoding="utf-8"))
    assert sidecar_avec["comparer_historique"] is True
    assert sidecar_avec["sans_historique"] is False
    assert sidecar_sans["sans_historique"] is True

    rapport = sortie_rapport.read_text(encoding="utf-8")
    assert "Comparaison avec / sans historique" in rapport
    assert "ecart median" in rapport
    assert "bras sans historique" in rapport


def test_main_refuse_les_deux_modes_d_historique(tmp_path, monkeypatch):
    import evaluation.puzzle_bench as puzzle_bench

    onnx = tmp_path / "modele.onnx"
    onnx.write_bytes(b"onnx factice")
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (onnx, {}))
    monkeypatch.setattr(sys, "argv", [
        "puzzle_bench.py", "--model", str(onnx), "--banc", str(BANC),
        "--sans-historique", "--comparer-historique",
        "--out-csv", str(tmp_path / "r.csv"),
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    with pytest.raises(SystemExit):
        puzzle_bench.main()
