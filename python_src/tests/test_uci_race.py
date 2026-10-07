"""La race entre parse_position et le fil de recherche.

parse_position appelle update_root et reconstruit self.board. Tant que le fil
de recherche tourne, il descend dans un arbre qu'update_root est en train de
detruire. Aujourd'hui c'est un bug latent ; avec le batching la boucle
detiendra des MCTSNode* bruts pendant plusieurs millisecondes d'appel GPU, et
cela deviendra une ecriture en memoire liberee.
"""
import os
import sys
from pathlib import Path
import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

import chess_engine
import core.uci as uci
from core.uci import UCIEngine


RACINE_PROJET = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("nom", [
    "core/uci.py", "archive/pygame/play_against_bot.py", "evaluation/tournament_elo.py",
])
def test_les_trois_lanceurs_partagent_le_batch_et_le_virtual_loss(nom):
    """Le batch 8 et le virtual loss 2 sont les reglages valides par les bancs."""
    path = RACINE_PROJET / "python_src" / nom
    if nom.startswith("archive/") and not path.exists():
        pytest.skip("Interface Pygame archivee non distribuee dans cette copie.")
    source = path.read_text(encoding="utf-8")
    assert "MCTS_BATCH_SIZE = 8" in source, nom
    assert "MCTS_VIRTUAL_LOSS = 2" in source, nom


def test_le_modele_uci_configure_est_resolu_depuis_le_depot():
    attendu = RACINE_PROJET / "data" / "checkpoints" / "onnx"

    assert Path(uci.MODEL_PATH).parent == attendu


class _FauxMCTS:
    """Enregistre les appels, sans moteur ni modele."""

    def __init__(self):
        self.appels = []

    def update_root(self, move_idx):
        self.appels.append(("update_root", move_idx))

    def reset_analysis(self):
        self.appels.append(("reset_analysis",))


def _moteur():
    """UCIEngine sans modele : c'est la raison d'etre de l'injection.

    MODEL_PATH (uci.py) pointe vers une autre machine, donc UCIEngine() n'est
    pas constructible ici.
    """
    return UCIEngine(evaluator=object(), mcts=_FauxMCTS())


def test_parse_position_arrete_la_recherche_avant_de_toucher_a_l_arbre():
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.update_root = lambda idx: ordre.append("update_root")

    moteur.parse_position(["startpos", "moves", "e2e4"])

    assert "stop_search" in ordre, "parse_position n'arrete pas la recherche"
    assert ordre[0] == "stop_search", f"stop_search doit venir en premier, ordre={ordre}"


def test_parse_position_arrete_la_recherche_aussi_sur_le_chemin_de_reconstruction():
    """Le chemin 3, nouvelle partie ou ponder miss, reassigne self.board."""
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.reset_analysis = lambda: ordre.append("reset_analysis")

    moteur.parse_position(["startpos"])

    assert ordre[0] == "stop_search", f"ordre={ordre}"
    assert "reset_analysis" in ordre


def test_le_constructeur_accepte_des_dependances_injectees():
    faux = _FauxMCTS()
    moteur = UCIEngine(evaluator=object(), mcts=faux)

    assert moteur.mcts is faux
    assert isinstance(moteur.board, chess_engine.Chessboard)


def test_le_constructeur_active_le_lot_fixe_et_le_tuning(monkeypatch):
    """Sur le chemin de production, le MCTS recoit le lot de forme fixe et le
    virtual loss valide par le rebalayage."""
    import core.uci as uci

    appels = []

    class FauxMCTS:
        def __init__(self, *args, **kwargs):
            pass

        def set_fixed_batch(self, enabled):
            appels.append(("fixed_batch", enabled))

        def set_tuning(self, virtual_loss, fpu_reduction=0.30,
                       collision_attempt_factor=4):
            appels.append(("tuning", virtual_loss))

    monkeypatch.setattr(uci.chess_engine, "MCTS", FauxMCTS)
    uci.UCIEngine(evaluator=object())

    assert appels == [("fixed_batch", uci.MCTS_FIXED_BATCH),
                      ("tuning", uci.MCTS_VIRTUAL_LOSS)]
    assert uci.MCTS_FIXED_BATCH is True
    assert uci.MCTS_VIRTUAL_LOSS == 2


def test_la_boucle_uci_transmet_le_nombre_de_workers():
    """Le cinquieme argument reel de step_analysis doit etre le reglage
    MCTS_WORKER_COUNT, pas une valeur par defaut implicite."""
    import threading

    appels = []
    premier = threading.Event()

    class FauxMCTSTransmission:
        def step_analysis(self, board, simulations, c_puct, batch_size,
                          worker_count):
            appels.append((simulations, batch_size, worker_count))
            premier.set()

        def get_analysis_results(self):
            return []

        def reset_analysis(self):
            pass

        def update_root(self, move_idx):
            pass

    moteur = UCIEngine(evaluator=object(), mcts=FauxMCTSTransmission())
    moteur.start_search(["go", "movetime", "10000"])
    try:
        assert premier.wait(timeout=5), "step_analysis n'a jamais ete appele"
    finally:
        moteur.stop_search()

    assert appels
    _simulations, batch_size, worker_count = appels[0]
    assert batch_size == uci.MCTS_BATCH_SIZE
    assert worker_count == uci.MCTS_WORKER_COUNT == 8


def test_la_recherche_journalise_le_bilan(capsys, tmp_path, monkeypatch):
    """Le bilan par coup sort sur stderr et dans un journal dedie, pour
    diagnostiquer un coup faible a posteriori sans dependre de lichess-bot."""
    import threading
    from types import SimpleNamespace

    journal = tmp_path / "uci_stats.log"
    monkeypatch.setattr(uci, "STATS_LOG", journal)
    premier = threading.Event()

    class FauxMCTSRecherche:
        indices = ()

        def step_analysis(self, board, simulations, c_puct, batch_size,
                          worker_count):
            premier.set()

        def get_analysis_results(self):
            return [SimpleNamespace(move_idx=index, visits=7 - rang,
                                    q_value=0.25, prior=0.5, v_value=0.2)
                    for rang, index in enumerate(FauxMCTSRecherche.indices)]

        def reset_analysis(self):
            pass

        def update_root(self, move_idx):
            pass

    moteur = UCIEngine(evaluator=object(), mcts=FauxMCTSRecherche())
    moteur.board.set_startup_pieces()
    FauxMCTSRecherche.indices = moteur.board.get_legal_move_indices()[:2]
    moteur.start_search(["go", "movetime", "10000"])
    try:
        assert premier.wait(timeout=5), "step_analysis n'a jamais ete appele"
    finally:
        moteur.stop_search()

    err = capsys.readouterr().err
    assert "recherche :" in err

    contenu = journal.read_text(encoding="utf-8")
    assert "recherche :" in contenu
    assert "coup 1 blancs" in contenu
    assert "2e" in contenu
    assert "workers" in contenu
    assert str(uci.MCTS_WORKER_COUNT) in contenu


def test_le_journal_tourne_au_dela_de_la_taille_max(tmp_path, monkeypatch):
    """Le journal ne doit pas grossir sans fin sur le disque."""
    journal = tmp_path / "uci_stats.log"
    monkeypatch.setattr(uci, "STATS_LOG", journal)
    monkeypatch.setattr(uci, "STATS_LOG_MAX_BYTES", 200)
    journal.write_text("x" * 400, encoding="utf-8")

    uci._journaliser("ligne suivante")

    ancien = tmp_path / "uci_stats.log.1"
    assert ancien.exists()
    assert ancien.stat().st_size == 400
    assert "ligne suivante" in journal.read_text(encoding="utf-8")
