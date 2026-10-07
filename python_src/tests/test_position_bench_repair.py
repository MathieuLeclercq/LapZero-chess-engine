"""Reparation et publication depuis l'audit en cache, sans recherche moteur."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.build_position_bench as builder
from evaluation.position_bench_teacher import AnnotationStore, cle_position


@pytest.fixture
def reserve(monkeypatch):
    monkeypatch.setattr(builder, "QUOTAS", {case: 2 for case in builder.CELLULES})
    monkeypatch.setattr(builder, "TAILLE_SOUS_BANC", 9)
    monkeypatch.setattr(builder, "AUDIT_COUNT", 18)
    records = []
    for phase, bucket in builder.CELLULES:
        score = {"disputee": 0.5, "avantage": 0.75, "decisive": 0.95}[bucket]
        for i in range(6):
            key = f"{phase}/{bucket}/{i}"
            records.append({
                "position_id": hashlib.sha256(key.encode()).hexdigest(),
                "game_fingerprint": key, "event_id": key,
                "white_id": "w" + key, "black_id": "b" + key,
                "phase": phase, "turn": i % 2, "wdl_bucket": bucket,
                "labels": [{"index": 1, "score": score, "nodes": 200_001},
                           {"index": 2, "score": score - 0.1, "nodes": 200_002}],
                "s_best": score,
            })
    base = builder.select_positions(records, builder.QUOTAS,
                                    salt=builder.SEL_FINAL,
                                    event_cap=100, player_cap=20)
    deeper = copy.deepcopy(base)
    for record in deeper:
        for label in record["labels"]:
            label["score"] += 0.03
            label["nodes"] = 400_001
        record["s_best"] += 0.03
    return records, deeper


def test_reparation_utilise_les_etiquettes_profondes_sans_falsifier_l_audit(reserve):
    records, deeper = reserve
    originaux = copy.deepcopy((records, deeper))

    selection, _, audit = builder.finaliser_depuis_audit(records, deeper)

    profonds = {r["position_id"]: r for r in deeper}
    assert all(r["labels"] == profonds[r["position_id"]]["labels"]
               for r in selection)
    assert all(r["annotation_budget_nodes"] == 400_000 for r in selection)
    assert audit["passed"] is False
    assert audit["accepted_with_warning"] is True
    assert audit["variation_moyenne"] == pytest.approx(0.03)
    assert audit["comparison"] == "original_labels_vs_cached_audit"
    assert audit["post_repair_stability_verified"] is False
    assert (records, deeper) == originaux


def test_mcts_est_stratifie_dans_les_positions_corrigees_et_deterministe(reserve):
    records, deeper = reserve

    selection, ids, audit = builder.finaliser_depuis_audit(records, deeper)
    inverse, ids_inverse, _ = builder.finaliser_depuis_audit(
        list(reversed(records)), list(reversed(deeper)))

    par_id = {r["position_id"]: r for r in selection}
    assert len(ids) == len(set(ids)) == 9
    assert set(ids).issubset({r["position_id"] for r in deeper})
    assert collections.Counter((par_id[i]["phase"], par_id[i]["wdl_bucket"])
                               for i in ids) == {case: 1 for case in builder.CELLULES}
    assert audit["search_reference_nodes"] == 400_000
    assert ids == ids_inverse
    assert selection == inverse


def test_reclassement_recalcule_les_quotas_sans_changer_les_seuils(reserve):
    records, deeper = reserve
    cible = next(r for r in deeper if r["wdl_bucket"] == "disputee")
    cible["labels"] = [{"index": 1, "score": 0.95, "nodes": 400_001},
                       {"index": 2, "score": 0.85, "nodes": 400_001}]
    cible["s_best"] = 0.95
    cible["wdl_bucket"] = "decisive"

    selection, ids, _ = builder.finaliser_depuis_audit(records, deeper)

    assert len(selection) == 18
    assert collections.Counter((r["phase"], r["wdl_bucket"])
                               for r in selection) == {case: 2 for case in builder.CELLULES}
    assert len(ids) == 9
    assert all(r["wdl_bucket"] == builder.wdl_category(r["s_best"])
               for r in selection)


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "history"])
def test_reparation_refuse_un_audit_incomplet_ou_incompatible(reserve, alteration):
    records, deeper = reserve
    if alteration == "missing":
        deeper.pop()
    elif alteration == "duplicate":
        deeper[-1] = copy.deepcopy(deeper[0])
    else:
        deeper[0]["game_fingerprint"] = "autre-partie"

    with pytest.raises(ValueError):
        builder.finaliser_depuis_audit(records, deeper)


def test_reparation_refuse_un_pool_mcts_insuffisant(reserve, monkeypatch):
    records, deeper = reserve
    monkeypatch.setattr(builder, "AUDIT_COUNT", 9)
    base = builder.select_positions(records, builder.QUOTAS, salt=builder.SEL_FINAL,
                                    event_cap=100, player_cap=20)
    ids = set(builder.select_audit_ids(base, 9))
    pool = [r for r in deeper if r["position_id"] in ids]

    with pytest.raises(builder.SelectionIncomplete):
        builder.finaliser_depuis_audit(records, pool)


def test_cli_repare_depuis_le_cache_sans_analyse_et_preserve_la_reserve(
        reserve, tmp_path, monkeypatch):
    records, deeper = reserve
    identite = {"name": "Stockfish test", "binary_sha256": "a" * 64}
    class Identification:
        def quit(self):
            pass
    monkeypatch.setattr(builder, "_ouvrir_stockfish",
                        lambda args: (Identification(), identite))
    def analyse_interdite(*args, **kwargs):
        pytest.fail("aucune recherche Stockfish ou extraction ne doit etre lancee")
    monkeypatch.setattr(builder, "annotate_position", analyse_interdite)
    monkeypatch.setattr(builder, "_ouvrir_moteurs", analyse_interdite)
    monkeypatch.setattr(builder, "download_archive", analyse_interdite)
    builder.ecrire_jsonl_zst(records, tmp_path / "annotated.jsonl.zst")
    avant = (tmp_path / "annotated.jsonl.zst").read_bytes()
    source = {"source_id": "test", "month": "2026-08", "url": "https://example.org/a",
              "archive_sha256": "c" * 64, "license": "CC BY-SA 4.0", "attribution": "Lichess"}
    (tmp_path / "sources.json").write_text(json.dumps([source]), encoding="utf-8")
    with AnnotationStore(tmp_path, builder._config_store(identite)) as store:
        for record in deeper:
            store.put(cle_position(record, 400_000), record)

    code = builder.main([
        "finalize", "--repair-from-cached-audit", "--stockfish", "faux",
        "--work-dir", str(tmp_path), "--output", str(tmp_path / "v1")])

    assert code == 0
    manifest = json.loads((tmp_path / "v1/manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"positions": 18, "search": 9}
    assert manifest["audit"]["passed"] is False
    assert manifest["audit"]["accepted_with_warning"] is True
    assert set(manifest["search_ids"]).issubset({r["position_id"] for r in deeper})
    assert (tmp_path / "annotated.jsonl.zst").read_bytes() == avant


def test_cli_refuse_une_analyse_en_cache_manquante_sans_la_recalculer(
        reserve, tmp_path, monkeypatch):
    records, deeper = reserve
    identite = {"name": "Stockfish test", "binary_sha256": "a" * 64}
    class Identification:
        def quit(self):
            pass
    monkeypatch.setattr(builder, "_ouvrir_stockfish",
                        lambda args: (Identification(), identite))
    monkeypatch.setattr(builder, "annotate_position",
                        lambda *args, **kwargs: pytest.fail("recalcul interdit"))
    builder.ecrire_jsonl_zst(records, tmp_path / "annotated.jsonl.zst")
    with AnnotationStore(tmp_path, builder._config_store(identite)) as store:
        for record in deeper[:-1]:
            store.put(cle_position(record, 400_000), record)

    assert builder.main([
        "finalize", "--repair-from-cached-audit", "--stockfish", "faux",
        "--work-dir", str(tmp_path), "--output", str(tmp_path / "v1")]) == 1
    assert not (tmp_path / "v1").exists()


@pytest.mark.parametrize("command", [["build"], ["extract"], ["finalize", "--stage", "reserve"]])
def test_option_de_reparation_est_reservee_a_la_finalisation(command, tmp_path):
    assert builder.main(command + ["--repair-from-cached-audit", "--work-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.iterdir())
