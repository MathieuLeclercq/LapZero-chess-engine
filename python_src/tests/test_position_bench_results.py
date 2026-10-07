"""Tests du chargement, de la sauvegarde, de la comparaison et de W&B.

Les bancs de fixture sont de vrais enregistrements annotes, construits depuis
le PGN de test des sources. Aucun modele, aucun Stockfish.
"""
import json
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

from evaluation.bench_metrics import index_to_uci
from evaluation.build_position_bench import ecrire_jsonl_zst, write_dataset
from evaluation.position_bench_metrics import (
    EvalReport,
    RawResult,
    SourceInfo,
    aggregate,
)
from evaluation.position_bench_results import (
    agreger_resultat,
    compare_results,
    find_previous,
    load_dataset,
    load_result,
    paired_bootstrap,
    save_result,
    wandb_metrics,
)
from evaluation.position_bench_sources import (
    extract_candidates,
    read_games,
    replay_position,
)

BROADCASTS = RACINE / "tests" / "data" / "position_bench" / "broadcasts.pgn"
SOURCE: SourceInfo = {
    "source_id": "lichess-broadcasts-2026-08",
    "url": "https://database.lichess.org/broadcasts/x-2026-08.pgn.zst",
    "month": "2026-08",
    "archive_sha256": "c" * 64,
    "license": "CC BY-SA 4.0",
    "attribution": "Lichess",
}


@pytest.fixture(scope="module")
def annotees():
    candidats = extract_candidates(
        list(read_games(BROADCASTS, SOURCE)), SOURCE)
    records = []
    for candidat in candidats:
        board = replay_position(candidat)
        labels = [
            {"uci": index_to_uci(board, index), "index": index,
             "wdl": (400, 200, 400), "score": 0.5, "cp": 0,
             "mate": None, "depth": 20, "nodes": 200_000}
            for index in candidat["legal_indices"]
        ]
        records.append({**candidat, "labels": labels, "s_best": 0.5,
                        "wdl_bucket": "disputee"})
    return records


def _manifeste(records, search_ids, **extra):
    manifeste = {
        "schema_version": 1,
        "dataset_version": "v1",
        "dataset_sha256": "0" * 64,
        "training_forbidden": True,
        "sources": [SOURCE],
        "counts": {"positions": len(records), "search": len(search_ids)},
        "quotas": {"ouverture/disputee": len(records)},
        "stockfish": {"name": "fake", "binary_sha256": "a" * 64},
        "audit": {"passed": True, "count": len(records)},
        "search_ids": list(search_ids),
        "protocol_defaults": {},
        "builder_revision": "test",
    }
    manifeste.update(extra)
    return manifeste


def _banc(dossier, records, search_ids=None):
    ids = search_ids if search_ids is not None else [records[0]["position_id"]]
    write_dataset(records, _manifeste(records, ids), dossier)
    return dossier


def _distributions(tailles):
    return np.concatenate([np.full(taille, 1.0 / taille, dtype=np.float32)
                           for taille in tailles]) if tailles else np.array(
        [], dtype=np.float32)


def _offsets(tailles):
    offsets = [0]
    for taille in tailles:
        offsets.append(offsets[-1] + taille)
    return np.asarray(offsets, dtype=np.int64)


def _raw(ids, regrets, *, legaux=None, search_ids=None, search_legaux=None):
    tableau_ids = np.asarray(ids, dtype="<U64")
    nombre = len(tableau_ids)
    tailles = list(legaux) if legaux is not None else [1] * nombre
    ids_recherche = (list(search_ids) if search_ids is not None
                     else list(tableau_ids[:1]))
    tailles_recherche = (list(search_legaux) if search_legaux is not None
                         else [1] * len(ids_recherche))
    pas_recherche = len(tailles_recherche)
    return cast(RawResult, {
        "position_ids": tableau_ids,
        "legal_offsets": _offsets(tailles),
        "legal_indices": np.zeros(sum(tailles), dtype=np.int32),
        "policy_probs": _distributions(tailles),
        "values": np.zeros(nombre, dtype=np.float32),
        "policy_regrets": np.asarray(regrets, dtype=np.float32),
        "search_ids": np.asarray(ids_recherche, dtype="<U64"),
        "search_offsets": _offsets(tailles_recherche),
        "search_indices": np.zeros(sum(tailles_recherche), dtype=np.int32),
        "search_probs": _distributions(tailles_recherche),
        "search_regrets": np.zeros(pas_recherche, dtype=np.float32),
        "search_counters": [],
        "timings": {"total_s": 1.0, "search_position_s": []},
    })


def _rapport(dataset="d" * 64, protocole="p" * 64, iteration=None,
             status="ok"):
    return cast(EvalReport, {
        "status": status,
        "dataset_version": "v1",
        "dataset_sha256": dataset,
        "model_sha256": "m" * 64,
        "protocol_id": protocole,
        "completed_positions": 3,
        "completed_search_positions": 1,
        "duration_s": 12.5,
        "timings": {"total_s": 12.5},
        "metrics": {},
        "comparison": None,
        "result_path": "",
        "error": None,
        "target_s": 300.0,
        "over_target": False,
        "iteration": iteration,
    })


# ============================================================
#                      CHARGEMENT
# ============================================================

def test_load_dataset_relit_un_banc_de_fixture(tmp_path, annotees):
    dossier = _banc(tmp_path / "v1", annotees)

    manifeste, records = load_dataset(dossier, production=False)

    assert len(records) == len(annotees)
    assert manifeste["counts"]["positions"] == len(annotees)
    assert records[0]["labels"]


def test_load_dataset_refuse_un_banc_non_productif(tmp_path, annotees):
    dossier = _banc(tmp_path / "v1", annotees)

    with pytest.raises(ValueError):
        load_dataset(dossier, production=True)


def test_load_dataset_refuse_un_dataset_modifie(tmp_path, annotees):
    dossier = _banc(tmp_path / "v1", annotees)
    chemin = dossier / "positions.jsonl.zst"
    chemin.write_bytes(chemin.read_bytes() + b"x")

    with pytest.raises(ValueError):
        load_dataset(dossier, production=False)


def test_load_dataset_refuse_un_dataset_tronque(tmp_path, annotees):
    manifeste = _manifeste(annotees, [annotees[0]["position_id"]])
    manifeste["counts"]["positions"] = len(annotees) + 1
    dossier = tmp_path / "v1"
    write_dataset(annotees, manifeste, dossier)

    with pytest.raises(ValueError):
        load_dataset(dossier, production=False)


def test_load_dataset_refuse_un_identifiant_duplique(tmp_path, annotees):
    double = [annotees[0], annotees[0]]
    dossier = _banc(tmp_path / "v1", double)

    with pytest.raises(ValueError):
        load_dataset(dossier, production=False)


def test_load_dataset_refuse_une_annotation_manquante(tmp_path, annotees):
    ampute = dict(annotees[0])
    ampute["labels"] = ampute["labels"][:-1]
    dossier = tmp_path / "v1"
    write_dataset([ampute], _manifeste([ampute], [ampute["position_id"]]),
                  dossier)

    with pytest.raises(ValueError):
        load_dataset(dossier, production=False)


def test_load_dataset_refuse_un_schema_inconnu(tmp_path, annotees):
    dossier = tmp_path / "v1"
    dossier.mkdir()
    ecrire_jsonl_zst(annotees, dossier / "positions.jsonl.zst")
    from evaluation.position_bench_results import sha256_fichier

    manifeste = _manifeste(annotees, [annotees[0]["position_id"]],
                           schema_version=2)
    manifeste["dataset_sha256"] = sha256_fichier(
        dossier / "positions.jsonl.zst")
    (dossier / "manifest.json").write_text(
        json.dumps(manifeste), encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(dossier, production=False)


# ============================================================
#                   SAUVEGARDE ET RELECTURE
# ============================================================

def test_save_et_load_font_un_aller_retour(tmp_path):
    raw = _raw(["a", "b"], [0.1, 0.2])
    rapport = _rapport()

    chemin = save_result(raw, rapport, tmp_path, "iter1")
    relu, rapport_relu = load_result(chemin)

    assert chemin.name == "iter1.npz"
    np.testing.assert_array_equal(relu["policy_regrets"],
                                  raw["policy_regrets"])
    assert relu["search_counters"] == []
    assert relu["timings"]["total_s"] == 1.0
    assert rapport_relu["dataset_sha256"] == rapport["dataset_sha256"]


def test_load_result_refuse_un_npz_altere(tmp_path):
    save_result(_raw(["a"], [0.1]), _rapport(), tmp_path, "iter1")
    chemin = tmp_path / "iter1.npz"
    chemin.write_bytes(chemin.read_bytes() + b"x")

    with pytest.raises(ValueError):
        load_result(chemin)


def test_save_refuse_un_resultat_different(tmp_path):
    save_result(_raw(["a"], [0.1]), _rapport(), tmp_path, "iter1")
    # Meme contenu : idempotent.
    save_result(_raw(["a"], [0.1]), _rapport(), tmp_path, "iter1")

    with pytest.raises(FileExistsError):
        save_result(_raw(["a"], [0.9]), _rapport(), tmp_path, "iter1")


# ============================================================
#                      COMPARAISON
# ============================================================

def test_bootstrap_d_un_delta_constant():
    delta = np.full(100, -0.01)

    bas, haut = paired_bootstrap(delta)

    assert bas == pytest.approx(-0.01)
    assert haut == pytest.approx(-0.01)
    assert haut < 0


def test_bootstrap_est_deterministe():
    delta = np.linspace(-0.5, 0.5, 500)

    assert paired_bootstrap(delta) == paired_bootstrap(delta)


def test_compare_apparie_par_identifiant_et_ignore_l_ordre():
    courant = _raw(["a", "b", "c"], [0.10, 0.20, 0.30])
    precedent = _raw(["c", "a", "b"], [0.30, 0.15, 0.20])
    meta = {"dataset_sha256": "d" * 64, "protocol_id": "p" * 64}

    comparaison = compare_results(courant, precedent, meta, meta)

    # Ecarts apparies : a -0.05, b 0, c 0.
    assert comparaison["count"] == 3
    assert comparaison["delta_policy_regret"] == pytest.approx(-0.05 / 3)
    assert comparaison["improved"] == 1
    assert comparaison["unchanged"] == 2


def test_compare_declare_l_amelioration_quand_la_borne_haute_est_negative():
    ids = [f"p{i}" for i in range(100)]
    courant = _raw(ids, [0.05] * 100)
    precedent = _raw(ids, [0.15] * 100)
    meta = {"dataset_sha256": "d" * 64, "protocol_id": "p" * 64}

    comparaison = compare_results(courant, precedent, meta, meta)

    assert comparaison["delta_policy_regret"] == pytest.approx(-0.1)
    assert comparaison["delta_policy_regret_high95"] < 0
    assert comparaison["verdict"] == "ameliore"


def test_compare_refuse_un_dataset_ou_un_protocole_different():
    courant = _raw(["a"], [0.1])
    precedent = _raw(["a"], [0.2])

    with pytest.raises(ValueError):
        compare_results(courant, precedent,
                        {"dataset_sha256": "d1", "protocol_id": "p"},
                        {"dataset_sha256": "d2", "protocol_id": "p"})
    with pytest.raises(ValueError):
        compare_results(courant, precedent,
                        {"dataset_sha256": "d", "protocol_id": "p1"},
                        {"dataset_sha256": "d", "protocol_id": "p2"})


def test_compare_refuse_des_positions_differentes():
    courant = _raw(["a", "b"], [0.1, 0.2])
    precedent = _raw(["a"], [0.2])
    meta = {"dataset_sha256": "d", "protocol_id": "p"}

    with pytest.raises(ValueError):
        compare_results(courant, precedent, meta, meta)


# ============================================================
#                        AGREGATION
# ============================================================

def test_agreger_resultat_utilise_les_labels_et_la_configuration(annotees):
    records = annotees[:2]
    raw = _raw(
        [record["position_id"] for record in records], [0.0, 0.1],
        legaux=[len(record["legal_indices"]) for record in records],
        search_ids=[records[0]["position_id"]],
        search_legaux=[len(records[0]["legal_indices"])])
    # Les distributions sont uniformes sur les coups legaux : le regret
    # attendu vaut zero car tous les labels portent le meme score.
    manifeste = _manifeste(annotees, [annotees[0]["position_id"]])

    metriques = agreger_resultat(manifeste, annotees, raw)

    assert metriques["policy"]["count"] == 2
    assert metriques["policy"]["policy_expected_regret"] == pytest.approx(0.0)
    assert metriques["search"]["count"] == 1


# ============================================================
#                          W&B
# ============================================================

def _rapport_ok(duration=340.0, correlation=None):
    lignes = [
        {"phase": "ouverture", "wdl_bucket": "disputee",
         "player_type": "human", "policy_expected_regret": 0.1,
         "policy_argmax_regret": 0.1, "near_best_mass": 0.5,
         "catastrophic_mass": 0.0, "value_pred": 0.0, "value_target": 0.0,
         "cp_regret": 0.0},
        {"phase": "finale", "wdl_bucket": "avantage",
         "player_type": "bot", "policy_expected_regret": 0.2,
         "policy_argmax_regret": 0.2, "near_best_mass": 0.4,
         "catastrophic_mass": 0.1, "value_pred": 0.0, "value_target": 0.0,
         "cp_regret": 10.0},
    ]
    metriques = {"policy": aggregate(lignes), "search": aggregate(lignes),
                 "comparison": None}
    rapport = _rapport()
    rapport["duration_s"] = duration
    rapport["over_target"] = duration > 300.0
    rapport["metrics"] = metriques
    return rapport


def test_wandb_metrics_publie_les_tranches_et_le_depassement():
    rapport = _rapport_ok(duration=340.0)

    metriques = wandb_metrics(rapport)

    assert metriques["eval/position/over_target"] is True
    assert metriques["eval/position/policy_expected_regret"] == pytest.approx(
        0.15)
    assert metriques["eval/position/search_expected_regret"] == pytest.approx(
        0.15)
    assert metriques["eval/position/phase/ouverture/policy_expected_regret"] \
        == pytest.approx(0.1)
    assert metriques["eval/position/phase/ouverture/count"] == 1
    assert metriques["eval/position/player/bot/count"] == 1
    versionne = f"eval/position/v1/{rapport['protocol_id']}"
    assert f"{versionne}/policy_expected_regret" in metriques
    # Sans variance, la correlation est nulle : absente du payload numerique.
    assert "eval/position/value_correlation" not in metriques


def test_wandb_metrics_statut_erreur_ne_publie_aucun_score():
    rapport = _rapport(status="error")
    rapport["metrics"] = {}

    metriques = wandb_metrics(rapport)

    assert metriques["eval/position/status"] == "error"
    assert "eval/position/duration_s" in metriques
    assert not any("expected_regret" in cle for cle in metriques)


# ============================================================
#                     RESULTAT PRECEDENT
# ============================================================

def test_find_previous_choisit_la_derniere_iteration_compatible(tmp_path):
    meta = {"dataset_sha256": "d" * 64, "protocol_id": "p" * 64}
    for iteration in (3, 5, 7):
        rapport = _rapport(iteration=iteration)
        save_result(_raw([f"p{iteration}"], [0.1]), rapport, tmp_path,
                    f"iter{iteration}")
    # Incompatible : meme iteration, autre dataset.
    rapport = _rapport(dataset="e" * 64, iteration=9)
    save_result(_raw(["autre"], [0.1]), rapport, tmp_path / "autre", "iter9")

    trouve = find_previous(tmp_path, meta["dataset_sha256"],
                           meta["protocol_id"], 6)

    assert trouve is not None
    assert trouve.name == "iter5.json"
    assert find_previous(tmp_path, meta["dataset_sha256"],
                         meta["protocol_id"], 3) is None
