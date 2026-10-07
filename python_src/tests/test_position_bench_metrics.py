"""Tests des contrats et metriques pures du banc de positions.

Aucun modele, aucun processus : le module ne manipule que des dictionnaires et
des tableaux numpy.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

from evaluation.position_bench_metrics import (
    TAILLE_POLICY,
    ContratInvalide,
    aggregate,
    score_distribution,
    validate_manifest,
    validate_position,
    value_metrics,
    wdl_category,
    wdl_score,
    wdl_value,
)

DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _label(index, score, wdl, cp: int | None = 100,
           mate: int | None = None, uci="a1a2"):
    return {"uci": uci, "index": index, "wdl": wdl, "score": score,
            "cp": cp, "mate": mate, "depth": 20, "nodes": 200_000}


def _position(legal_count=3, turn=0, labels=None, **extra):
    legal = list(range(1, legal_count + 1))
    if labels is None:
        labels = [_label(i, 0.5, (400, 200, 400)) for i in legal]
    s_best = max(label["score"] for label in labels)
    record = {
        "position_id": "p1", "game_id": "g1",
        "game_fingerprint": "a" * 64, "source_id": "s1",
        "event_id": "e1", "white_id": "w1", "black_id": "b1",
        "white_elo": 2200, "black_elo": 2100,
        "white_type": "human", "black_type": "bot",
        "game_date": "2026-08-01", "start_fen": DEPART,
        "moves_uci": ["e2e4"], "fen": DEPART,
        "ply": 10, "turn": turn, "halfmove_clock": 0, "repetition_count": 1,
        "phase": "milieu", "legal_indices": legal,
        "labels": labels, "s_best": s_best,
        "wdl_bucket": wdl_category(s_best),
    }
    record.update(extra)
    return record


# ============================================================
#              WDL, REGRETS ET DISTRIBUTIONS
# ============================================================

def test_echelle_wdl_et_value():
    assert wdl_score((300, 600, 100)) == pytest.approx(0.6)
    assert wdl_value((300, 600, 100)) == pytest.approx(0.2)


def test_categories_wdl_aux_frontieres():
    assert wdl_category(0.35) == "disputee"
    assert wdl_category(0.65) == "disputee"
    assert wdl_category(0.15) == "avantage"
    assert wdl_category(0.85) == "avantage"
    assert wdl_category(0.14) == "decisive"
    assert wdl_category(0.86) == "decisive"


def test_wdl_refuse_les_effectifs_invalides():
    with pytest.raises(ContratInvalide):
        wdl_score((1, 2))  # type: ignore[arg-type]
    with pytest.raises(ContratInvalide):
        wdl_score((0, 0, 0))
    with pytest.raises(ContratInvalide):
        wdl_score((1, -1, 2))
    with pytest.raises(ContratInvalide):
        wdl_value((1, 2.5, 0))  # type: ignore[arg-type]


def test_regret_pondere():
    labels = [
        _label(1, 0.8, (600, 400, 0), cp=120),
        _label(2, 0.6, (300, 600, 100), cp=40),
        _label(3, 0.2, (0, 400, 600), cp=-120),
    ]

    m = score_distribution(np.array([0.5, 0.3, 0.2]), labels)

    assert m["expected_regret"] == pytest.approx(0.18)
    assert m["argmax_regret"] == 0
    assert m["near_best_mass"] == pytest.approx(0.5)


def test_masses_aux_frontieres_inclusives():
    """Regret de 0.02 : masse proche ; regret de 0.20 : masse catastrophique."""
    labels = [
        _label(1, 0.8, (600, 400, 0), cp=100),
        _label(2, 0.78, (560, 440, 0), cp=80),
        _label(3, 0.6, (200, 800, 0), cp=0),
    ]

    m = score_distribution(np.array([0.0, 0.5, 0.5]), labels)

    assert m["argmax_regret"] == pytest.approx(0.02)
    assert m["near_best_mass"] == pytest.approx(0.5)
    assert m["catastrophic_mass"] == pytest.approx(0.5)


def test_score_distribution_refuse_une_distribution_invalide():
    labels = [
        _label(1, 0.8, (600, 400, 0)),
        _label(2, 0.6, (300, 600, 100)),
    ]

    with pytest.raises(ContratInvalide):
        score_distribution(np.array([0.5, 0.4]), labels)
    with pytest.raises(ContratInvalide):
        score_distribution(np.array([1.2, -0.2]), labels)
    with pytest.raises(ContratInvalide):
        score_distribution(np.array([1.0]), labels)
    with pytest.raises(ContratInvalide):
        score_distribution(np.array([np.nan, 1.0]), labels)


def test_score_distribution_refuse_des_etiquettes_non_triees():
    labels = [
        _label(2, 0.8, (600, 400, 0)),
        _label(1, 0.6, (300, 600, 100)),
    ]

    with pytest.raises(ContratInvalide):
        score_distribution(np.array([0.5, 0.5]), labels)


# ============================================================
#                  DIAGNOSTIC EN CENTIPIONS
# ============================================================

def test_cp_regret_choisit_le_meilleur_cp_malgre_la_wdl_saturee():
    """Deux coups a S = 1 et cp differents : la WDL ne les departage pas, le
    classement cp/mat si."""
    labels = [
        _label(1, 1.0, (1000, 0, 0), cp=900),
        _label(2, 1.0, (1000, 0, 0), cp=500),
    ]

    m = score_distribution(np.array([0.0, 1.0]), labels)

    assert m["cp_regret"] == pytest.approx(400)


def test_un_mat_perdant_passe_apres_un_cp_dans_le_classement():
    labels = [
        _label(1, 0.0, (0, 0, 1000), cp=None, mate=-2),
        _label(2, 0.0, (0, 0, 1000), cp=-100),
    ]

    m = score_distribution(np.array([0.0, 1.0]), labels)

    assert m["cp_regret"] == pytest.approx(0.0)


def test_cp_regret_null_des_qu_un_mat_est_en_jeu():
    labels = [
        _label(1, 1.0, (1000, 0, 0), cp=None, mate=3),
        _label(2, 1.0, (1000, 0, 0), cp=900),
    ]

    # Le meilleur est le mat : le coup choisi, un cp, sort du diagnostic.
    assert score_distribution(np.array([0.0, 1.0]), labels)["cp_regret"] is None
    # Et un coup choisi mat en sort aussi.
    assert score_distribution(np.array([1.0, 0.0]), labels)["cp_regret"] is None


# ============================================================
#                      VALUE SCALAIRE
# ============================================================

def test_value_metrics_sur_des_valeurs_synthetiques():
    m = value_metrics([0.1, 0.2, 0.3], [0.0, 0.2, 0.4])

    assert m["value_mae"] == pytest.approx(0.2 / 3)
    assert m["value_rmse"] == pytest.approx(math.sqrt(0.02 / 3))
    assert m["value_bias"] == pytest.approx(0.0)
    assert m["value_correlation"] == pytest.approx(1.0)


def test_value_metrics_correlation_null_si_variance_nulle():
    m = value_metrics([0.1, 0.2], [0.3, 0.3])

    assert m["value_correlation"] is None
    assert m["value_mae"] == pytest.approx(0.15)


def test_value_metrics_une_seule_valeur_n_a_pas_de_correlation():
    m = value_metrics([0.5], [0.25])

    assert m["value_mae"] == pytest.approx(0.25)
    assert m["value_correlation"] is None


def test_value_metrics_liste_vide():
    assert value_metrics([], []) == {
        "value_mae": None, "value_rmse": None,
        "value_bias": None, "value_correlation": None}


def test_value_metrics_refuse_des_entrees_invalides():
    with pytest.raises(ContratInvalide):
        value_metrics([0.1, 0.2], [0.1])
    with pytest.raises(ContratInvalide):
        value_metrics([0.1, np.inf], [0.1, 0.2])


# ============================================================
#                       AGREGATION
# ============================================================

def _row(phase="milieu", bucket="disputee", player="human", *,
         expected=0.1, argmax=0.1, near=0.5, catast=0.1,
         pred=0.0, target=0.0, cp: float | None = 0.0):
    return {"phase": phase, "wdl_bucket": bucket, "player_type": player,
            "policy_expected_regret": expected,
            "policy_argmax_regret": argmax,
            "near_best_mass": near, "catastrophic_mass": catast,
            "value_pred": pred, "value_target": target, "cp_regret": cp}


def test_aggregate_roupe_par_phase_wdl_et_type_avec_effectifs():
    rows = [
        _row(phase="ouverture", bucket="disputee", player="human"),
        _row(phase="finale", bucket="decisive", player="bot"),
        _row(phase="finale", bucket="decisive", player="bot"),
    ]

    a = aggregate(rows)

    assert a["count"] == 3
    assert a["phase"]["ouverture"]["count"] == 1
    assert a["phase"]["finale"]["count"] == 2
    assert a["phase"]["milieu"]["count"] == 0
    assert a["wdl"]["decisive"]["count"] == 2
    assert a["wdl"]["avantage"]["count"] == 0
    assert a["player"]["bot"]["count"] == 2
    assert a["player"]["unknown"]["count"] == 0


def test_aggregate_ne_moyenne_pas_les_correlations_des_sous_groupes():
    """La globale est recalculee sur les valeurs brutes : la moyenne des
    correlations de phase vaudrait zero ici."""
    rows = [
        _row(phase="ouverture", pred=0.0, target=0.0),
        _row(phase="ouverture", pred=1.0, target=1.0),
        _row(phase="ouverture", pred=2.0, target=2.0),
        _row(phase="finale", pred=0.0, target=1.0),
        _row(phase="finale", pred=1.0, target=0.0),
    ]

    a = aggregate(rows)

    assert a["phase"]["ouverture"]["value_correlation"] == pytest.approx(1.0)
    assert a["phase"]["finale"]["value_correlation"] == pytest.approx(-1.0)
    assert a["value_correlation"] == pytest.approx(1.8 / 2.8)


def test_aggregate_calcule_medianes_quantiles_et_seuils_cp():
    rows = [_row(cp=0.0), _row(cp=20.0), _row(cp=100.0)]

    a = aggregate(rows)

    assert a["cp_regret_median"] == pytest.approx(20.0)
    assert a["cp_regret_p90"] == pytest.approx(84.0)
    assert a["policy_within_20cp"] == pytest.approx(2 / 3)
    assert a["policy_within_50cp"] == pytest.approx(2 / 3)
    assert a["policy_within_100cp"] == pytest.approx(1.0)
    assert a["cp_coverage"] == pytest.approx(1.0)


def test_aggregate_sans_position_admissible_n_a_que_la_couverture():
    a = aggregate([_row(cp=None), _row(cp=None)])

    assert a["cp_regret_median"] is None
    assert a["cp_regret_p90"] is None
    assert a["cp_coverage"] == 0.0
    assert a["policy_within_20cp"] == 0.0


def test_aggregate_supporte_une_liste_vide():
    a = aggregate([])

    assert a["count"] == 0
    assert a["policy_expected_regret"] is None
    assert a["cp_coverage"] == 0.0
    assert a["phase"]["ouverture"]["count"] == 0


def test_aggregate_refuse_une_ligne_hors_contrat():
    with pytest.raises(ContratInvalide):
        aggregate([_row(phase="inconnue")])


# ============================================================
#                  VALIDATION DES POSITIONS
# ============================================================

def test_valide_une_position_annotee_complete():
    validate_position(_position())


def test_accepte_les_deux_couleurs():
    validate_position(_position(turn=0))
    validate_position(_position(turn=1))

    with pytest.raises(ContratInvalide):
        validate_position(_position(turn=2))


def test_rejette_un_index_de_policy_hors_bornes():
    record = _position(legal_count=1)
    record["legal_indices"] = [TAILLE_POLICY]
    record["labels"][0]["index"] = TAILLE_POLICY

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_rejette_une_annotation_manquante():
    record = _position(legal_count=3)
    record["labels"] = record["labels"][:2]

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_rejette_une_annotation_dupliquee():
    record = _position(legal_count=2)
    record["labels"][1] = dict(record["labels"][0])

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_accepte_plus_de_128_coups_legaux():
    """La limite TT_MAX_MOVES du moteur ne doit jamais filtrer le banc."""
    record = _position(legal_count=140)

    assert len(record["legal_indices"]) == 140
    validate_position(record)


def test_rejette_un_champ_obligatoire_manquant():
    record = _position()
    del record["fen"]

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_rejette_une_etiquette_incoherente_avec_sa_wdl():
    record = _position(legal_count=1)
    record["labels"][0]["score"] = 0.9

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_rejette_une_etiquette_sans_cp_ni_mat():
    record = _position(legal_count=1)
    record["labels"][0]["cp"] = None

    with pytest.raises(ContratInvalide):
        validate_position(record)


def test_rejette_une_etiquette_avec_cp_et_mat():
    record = _position(legal_count=1)
    record["labels"][0]["mate"] = 3

    with pytest.raises(ContratInvalide):
        validate_position(record)


# ============================================================
#                  VALIDATION DU MANIFESTE
# ============================================================

def _manifest(**extra):
    manifest = {
        "schema_version": 1,
        "dataset_version": "v1",
        "dataset_sha256": "b" * 64,
        "training_forbidden": True,
        "sources": [{
            "source_id": "s1",
            "url": "https://database.lichess.org/x.pgn.zst",
            "month": "2026-08",
            "archive_sha256": "c" * 64,
            "license": "CC BY-SA 4.0",
            "attribution": "Lichess",
        }],
        "counts": {"positions": 10000, "search": 256},
        "quotas": {},
        "stockfish": {},
        "audit": {},
        "search_ids": ["p1", "p2"],
        "protocol_defaults": {},
        "builder_revision": "rev1",
    }
    manifest.update(extra)
    return manifest


def test_valide_un_manifeste_complet():
    validate_manifest(_manifest())


def test_rejette_un_manifeste_qui_autorise_l_entrainement():
    with pytest.raises(ContratInvalide):
        validate_manifest(_manifest(training_forbidden=False))


def test_rejette_un_schema_inconnu():
    with pytest.raises(ContratInvalide):
        validate_manifest(_manifest(schema_version=2))


def test_rejette_des_identifiants_de_recherche_dupliques():
    with pytest.raises(ContratInvalide):
        validate_manifest(_manifest(search_ids=["p1", "p1"]))


def test_rejette_un_hash_de_dataset_mal_forme():
    with pytest.raises(ContratInvalide):
        validate_manifest(_manifest(dataset_sha256="abc"))


def test_rejette_un_champ_de_manifeste_manquant():
    manifest = _manifest()
    del manifest["builder_revision"]

    with pytest.raises(ContratInvalide):
        validate_manifest(manifest)
