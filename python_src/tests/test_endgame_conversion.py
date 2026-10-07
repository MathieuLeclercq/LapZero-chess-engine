import os
import sys
from pathlib import Path
from types import SimpleNamespace

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
sys.path.insert(0, str(RACINE / "python_src" / "dev_tools"))
os.add_dll_directory(str(RACINE / "python_src"))

import endgame_conversion


def test_suite_standard_couvre_les_finales_retenues():
    comptes = {}
    for position in endgame_conversion.POSITIONS.values():
        comptes[position.famille] = comptes.get(position.famille, 0) + 1

        board = endgame_conversion.chess.Board(position.fen)
        assert board.is_valid(), position.nom
        assert not board.is_game_over(claim_draw=True), position.nom

    assert comptes == {
        "dame_contre_roi": 3,
        "dame_pion_contre_fou": 2,
        "deux_tours_contre_roi": 2,
        "deux_fous_contre_roi": 1,
        "tour_contre_roi": 4,
    }
    assert len(endgame_conversion.POSITIONS) == 12


def test_resume_agrege_par_politique_et_famille():
    resultats = [
        endgame_conversion.Resultat(
            position="tour_a", famille="tour_contre_roi", batch_size=8,
            cache_history_depth=0, plies=21, issue="mat", duree_s=10.0,
            max_halfmove=21, coups=[], tree_mode="reuse", tt_hits=5,
            tt_misses=10, tt_rule50_rejects=1, tt_context_rejects=2,
            tt_history_rejects=3),
        endgame_conversion.Resultat(
            position="tour_b", famille="tour_contre_roi", batch_size=8,
            cache_history_depth=0, plies=100, issue="nulle_50_coups",
            duree_s=20.0, max_halfmove=100, coups=[], tree_mode="reuse",
            tt_hits=7, tt_misses=11, tt_rule50_rejects=4,
            tt_context_rejects=5, tt_history_rejects=6),
    ]

    resume = endgame_conversion.resumer_resultats(resultats)

    assert resume == [{
        "politique": "h0",
        "famille": "tour_contre_roi",
        "parties": 2,
        "mats": 1,
        "nulles": 1,
        "plies_median": 60.5,
        "plies_max": 100,
        "duree_s": 30.0,
        "max_halfmove": 100,
        "tt_hits": 12,
        "tt_misses": 21,
        "tt_rule50_rejects": 5,
        "tt_context_rejects": 7,
        "tt_history_rejects": 9,
    }]


def test_mode_fresh_conserve_la_profondeur_tt(monkeypatch):
    constructions = []

    class FauxMCTS:
        def __init__(self, evaluator, taille_tt, cache_history_depth):
            constructions.append((taille_tt, cache_history_depth))

        def reset_counters(self):
            pass

        def step_analysis(self, board, simulations, c_puct, batch_size):
            pass

        def get_counters(self):
            return SimpleNamespace(
                tt_hits=0, tt_misses=1, tt_rule50_rejects=0,
                tt_context_rejects=0, tt_history_rejects=0)

        def get_analysis_results(self):
            return []

    monkeypatch.setattr(endgame_conversion.chess_engine, "MCTS", FauxMCTS)

    resultat = endgame_conversion.jouer(
        "tour", endgame_conversion.POSITIONS["tour_centre"].fen, object(),
        simulations=8, batch_size=0, max_plies=1, tree_mode="fresh",
        cache_history_depth=3)

    assert constructions == [(131071, 3), (131071, 3)]
    assert resultat.cache_history_depth == 3
    assert resultat.tt_misses == 1
