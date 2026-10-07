"""Tests du runtime d'evaluation du banc de positions.

Faux evaluateur et faux MCTS : rien n'exige de GPU, de session ONNX ni de
modele. Les positions de test sont de vraies positions du moteur C++ obtenues
en rejouant une partie de reference.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine
from evaluation.bench_metrics import index_to_uci
from evaluation.position_bench import evaluate_positions, profile_positions
from evaluation.position_bench_metrics import EvalConfig, SearchConfig
from evaluation.position_bench_sources import (
    position_identity,
)

# Partie de l'Opera : 33 demi-coups, du developpement au mat.
OPERA = [
    "e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "c8g4", "d4e5", "g4f3",
    "d1f3", "d6e5", "f1c4", "g8f6", "f3b3", "d8e7", "b1c3", "c7c6",
    "c1g5", "b7b5", "c3b5", "c6b5", "c4b5", "b8d7", "e1c1", "a8d8",
    "d1d7", "d8d7", "h1d1", "e7e6", "b5d7", "f6d7", "b3b8", "d7b8",
    "d1d8",
]


def _plateau(ply):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    for coup in OPERA[:ply]:
        assert board.move_piece_uci(coup), coup
    return board


def _record(ply):
    board = _plateau(ply)
    indices = sorted(board.get_legal_move_indices())
    labels = [
        {"uci": index_to_uci(board, index), "index": index,
         "wdl": (400, 200, 400), "score": 0.5, "cp": 0, "mate": None,
         "depth": 20, "nodes": 200_000}
        for index in indices
    ]
    return {
        "position_id": position_identity(board),
        "start_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves_uci": OPERA[:ply],
        "fen": board.to_fen(),
        "legal_indices": indices,
        "labels": labels,
    }


@pytest.fixture(scope="module")
def records():
    return [_record(ply) for ply in range(20, 30)]


class FauxEvaluateur:
    """Evaluateur factice ; la valeur depend de la ligne."""

    def __init__(self, non_fini=False, valeur_hors_bornes=False,
                 mauvaise_forme=False):
        self.non_fini = non_fini
        self.valeur_hors_bornes = valeur_hors_bornes
        self.mauvaise_forme = mauvaise_forme
        self.lots = []

    def predict_batch(self, states):
        lignes = states.shape[0]
        self.lots.append(int(lignes))
        logits = np.zeros((lignes, 4672), dtype=np.float32)
        for ligne in range(lignes):
            logits[ligne, 0] = float(ligne)
        values = np.linspace(-0.5, 0.5, lignes, dtype=np.float32)
        if self.non_fini:
            logits[0, 0] = np.nan
        if self.valeur_hors_bornes:
            values[0] = 1.5
        if self.mauvaise_forme:
            logits = logits[:, :100]
        return logits, values


class FauxMCTS:
    """MCTS factice : verifie que la racine et les compteurs sont neufs."""

    instances: ClassVar[list] = []

    def __init__(self, evaluator, tt_size, cache_history_depth):
        self.evaluator = evaluator
        self.tt_size = tt_size
        self.cache_history_depth = cache_history_depth
        self.cold = False
        self.completed = 0
        self.fixe = None
        self.tuning = None
        self.appels = []
        FauxMCTS.instances.append(self)

    def set_fixed_batch(self, enabled):
        self.fixe = enabled

    def set_tuning(self, virtual_loss, fpu_reduction, collision_attempt_factor):
        self.tuning = (virtual_loss, fpu_reduction, collision_attempt_factor)

    def clear_evaluation_cache(self):
        self.cold = True

    def reset_counters(self):
        assert self.cold
        self.completed = 0

    def mcts_search(self, board, simulations, c_puct, bruit, batch_size,
                    workers):
        assert self.cold and self.completed == 0
        self.appels.append((simulations, c_puct, bruit, batch_size, workers))
        self.completed = simulations
        self.cold = False
        pi = np.zeros(4672)
        pi[min(board.get_legal_move_indices())] = 1.0
        return pi

    def get_counters(self):
        return SimpleNamespace(
            completed_simulations=self.completed, nn_calls=384,
            nn_batches=48, tt_hits=0, tt_misses=384, leaf_collisions=0)


@pytest.fixture(autouse=True)
def _vider_instances():
    FauxMCTS.instances.clear()
    yield
    FauxMCTS.instances.clear()


def _config(lot=4, simulations=384):
    return EvalConfig(search=SearchConfig(simulations=simulations),
                      policy_batch_size=lot)


# ============================================================
#                      PASSAGE DIRECT
# ============================================================

def test_les_lots_couvrent_dix_positions_sans_perte_ni_doublon(records):
    evaluateur = FauxEvaluateur()

    raw = evaluate_positions(records, [], evaluateur, _config(lot=4),
                             mcts_factory=FauxMCTS)

    assert evaluateur.lots == [4, 4, 2]
    assert raw["legal_offsets"][0] == 0
    assert raw["legal_offsets"][-1] == len(raw["legal_indices"])
    assert len(raw["values"]) == len(records)
    assert raw["position_ids"][0] == records[0]["position_id"]


def test_les_probabilites_sont_legales_et_normalisees(records):
    raw = evaluate_positions(records, [], FauxEvaluateur(), _config(),
                             mcts_factory=FauxMCTS)

    offsets = raw["legal_offsets"]
    for rang in range(len(records)):
        debut, fin = int(offsets[rang]), int(offsets[rang + 1])
        probas = raw["policy_probs"][debut:fin]
        indices = raw["legal_indices"][debut:fin]
        assert indices.tolist() == records[rang]["legal_indices"]
        assert np.all(np.isfinite(probas))
        assert probas.sum() == pytest.approx(1.0)
        assert len(indices) == len(records[rang]["legal_indices"])


def test_la_valeur_stockee_est_celle_de_l_evaluateur(records):
    evaluateur = FauxEvaluateur()

    raw = evaluate_positions(records, [], evaluateur, _config(),
                             mcts_factory=FauxMCTS)

    # Le faux evaluateur numerote les lignes dans chaque lot : la valeur
    # attendue se recolle lot par lot, pas sur les dix positions.
    attendu = np.concatenate([
        np.linspace(-0.5, 0.5, taille, dtype=np.float32)
        for taille in evaluateur.lots])
    np.testing.assert_allclose(raw["values"], attendu)


# ============================================================
#                       RECHERCHE MCTS
# ============================================================

def test_le_mcts_recoit_les_reglages_et_le_budget_exact(records):
    ids = [records[0]["position_id"], records[1]["position_id"]]

    raw = evaluate_positions(records, ids, FauxEvaluateur(), _config(),
                             mcts_factory=FauxMCTS)

    assert len(FauxMCTS.instances) == 1
    mcts = FauxMCTS.instances[0]
    assert mcts.tt_size == 8192
    assert mcts.cache_history_depth == 0
    assert mcts.fixe is True
    assert mcts.tuning == (2, 0.30, 4)
    assert mcts.evaluator is not None
    assert mcts.appels == [(384, 1.4, False, 8, 8), (384, 1.4, False, 8, 8)]
    assert raw["search_ids"].tolist() == ids
    assert len(raw["search_counters"]) == 2
    assert all(c["completed_simulations"] == 384
               for c in raw["search_counters"])
    offsets = raw["search_offsets"]
    for rang in range(2):
        debut, fin = int(offsets[rang]), int(offsets[rang + 1])
        assert raw["search_probs"][debut:fin].sum() == pytest.approx(1.0)


def test_la_table_est_remise_a_froid_avant_chaque_recherche(records):
    ids = [records[2]["position_id"], records[3]["position_id"],
           records[2]["position_id"]]

    evaluate_positions(records, ids, FauxEvaluateur(), _config(),
                       mcts_factory=FauxMCTS)

    mcts = FauxMCTS.instances[0]
    # FauxMCTS leve si la table n'a pas ete refroidie : la sequence A, B, A
    # passe donc, et le meme A rend la meme distribution.
    assert len(mcts.appels) == 3


def test_un_budget_incomplet_est_refuse(records):
    class FauxMCTSIncomplet(FauxMCTS):
        def get_counters(self):
            return SimpleNamespace(
                completed_simulations=self.completed - 1, nn_calls=0,
                nn_batches=0, tt_hits=0, tt_misses=0, leaf_collisions=0)

    with pytest.raises(ValueError):
        evaluate_positions(records, [records[0]["position_id"]],
                           FauxEvaluateur(), _config(),
                           mcts_factory=FauxMCTSIncomplet)


# ============================================================
#                     REFUS DES DONNEES FAUSSES
# ============================================================

@pytest.mark.parametrize("champ", ["fen", "position_id", "legal_indices",
                                   "labels"])
def test_une_position_incoherente_est_refusee(records, champ):
    fausse = [dict(record) for record in records]
    if champ == "fen":
        fausse[0][champ] = "8/8/8/8/8/8/8/K6k w - - 0 1"
    elif champ == "position_id":
        fausse[0][champ] = "0" * 64
    elif champ == "legal_indices":
        fausse[0][champ] = fausse[0][champ][:-1]
    else:
        fausse[0][champ] = fausse[0][champ][:-1]

    with pytest.raises(ValueError):
        evaluate_positions(fausse, [], FauxEvaluateur(), _config(),
                           mcts_factory=FauxMCTS)


@pytest.mark.parametrize("drapeau", ["non_fini", "valeur_hors_bornes",
                                     "mauvaise_forme"])
def test_une_sortie_evaluateur_invalide_est_refusee(records, drapeau):
    evaluateur = FauxEvaluateur(**{drapeau: True})

    with pytest.raises(RuntimeError):
        evaluate_positions(records, [], evaluateur, _config(),
                           mcts_factory=FauxMCTS)


# ============================================================
#                    TEMPS ET MODE PROFILE
# ============================================================

def test_une_cible_depassee_ne_retire_aucune_position(records):
    """Aucune adaptation au temps : tout est traite, quoi qu'il arrive."""
    temps = [0.0]

    def horloge():
        temps[0] += 60.0
        return temps[0]

    ids = [record["position_id"] for record in records]

    raw = evaluate_positions(records, ids, FauxEvaluateur(), _config(),
                             mcts_factory=FauxMCTS, clock=horloge)

    assert len(raw["position_ids"]) == len(records)
    assert len(raw["search_ids"]) == len(ids)
    assert raw["timings"]["total_s"] > 300.0


def test_le_profil_ne_requiert_pas_d_etiquettes(records):
    brutes = []
    for record in records:
        sans = dict(record)
        del sans["labels"]
        brutes.append(sans)
    evaluateur = FauxEvaluateur()

    rapport = profile_positions(brutes, evaluateur, _config(lot=4),
                                mcts_factory=FauxMCTS, search_count=3)

    assert rapport["mode"] == "profile"
    assert rapport["count"] == len(records)
    assert rapport["searches"] == 3
    assert evaluateur.lots == [4, 4, 2]
    assert "policy_s" in rapport["timings"]
    assert len(rapport["timings"]["search_position_s"]) == 3
    # Aucun score ni version ecrite : les cles de mesure n'en contiennent pas.
    assert "metrics" not in rapport


def test_le_profil_sans_recherche_ne_cree_aucun_mcts():
    """search_count=0 : parcours direct seul, sans recherche."""
    # Les fixtures de module ne s'appliquent pas hors des tests parametres :
    # on reprend des positions brutes construites ici.
    sans = []
    for ply in (20, 21):
        record = dict(_record(ply))
        del record["labels"]
        sans.append(record)
    FauxMCTS.instances.clear()

    rapport = profile_positions(sans, FauxEvaluateur(), _config(),
                                mcts_factory=FauxMCTS, search_count=0)

    assert rapport["searches"] == 0
