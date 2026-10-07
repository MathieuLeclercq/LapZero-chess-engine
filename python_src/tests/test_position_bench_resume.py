"""Reprise et extension du corpus, sans reseau ni processus Stockfish.

Les fichiers, le cache SQLite, les quotas, l'audit et la publication sont
reels. Seules les analyses du moteur sont remplacees par des scores controles.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.build_position_bench as builder
from evaluation.position_bench_teacher import AnnotationStore, cle_criblage


class Moteur:
    def quit(self):
        pass


@pytest.fixture
def campagne(tmp_path, monkeypatch):
    def reseau_interdit(*args, **kwargs):
        pytest.fail("les tests doivent reutiliser uniquement les archives synthetiques")

    monkeypatch.setattr(builder, "download_archive", reseau_interdit)
    monkeypatch.setattr(builder, "QUOTAS", {case: 2 for case in builder.CELLULES})
    monkeypatch.setattr(builder, "TAILLE_SOUS_BANC", 9)
    monkeypatch.setattr(builder, "FACTEUR_RESERVE", 2.0)
    identite = {"name": "Stockfish test", "binary_sha256": "a" * 64}
    monkeypatch.setattr(builder, "_ouvrir_moteurs",
                        lambda args: ([Moteur()], identite))
    analyses = collections.Counter()

    def cribler(moteur, position):
        analyses[position["position_id"]] += 1
        return position["test_score"]

    def annoter(moteur, position, nodes):
        score = position["test_score"]
        return dict(position, labels=[{"index": 1, "score": score}],
                    s_best=score, wdl_bucket=builder.wdl_category(score))

    monkeypatch.setattr(builder, "screen_position", cribler)
    monkeypatch.setattr(builder, "annotate_position", annoter)

    def preparer(mois, *, incomplet=False):
        archives = tmp_path / "archives"
        archives.mkdir(exist_ok=True)
        (archives / f"lichess_db_broadcast_{mois}.pgn.zst").write_bytes(b"archive")
        positions = []
        for phase, categorie in builder.CELLULES:
            if incomplet and (phase, categorie) == ("finale", "avantage"):
                continue
            for i in range(8):
                cle = f"{mois}/{phase}/{categorie}/{i}"
                positions.append({
                    "position_id": hashlib.sha256(cle.encode()).hexdigest(),
                    "phase": phase, "game_id": cle, "game_fingerprint": cle,
                    "source_id": f"lichess-broadcasts-{mois}", "ply": 20,
                    "event_id": cle, "white_id": "w" + cle,
                    "black_id": "b" + cle, "turn": i % 2,
                    "halfmove_clock": 0, "repetition_count": 1,
                    "test_score": {"disputee": 0.5, "avantage": 0.75,
                                   "decisive": 0.95}[categorie],
                })
        builder.ecrire_jsonl_zst(positions, tmp_path / f"candidates_{mois}.jsonl.zst")
        return positions

    def lancer(mois):
        return builder.main([
            "build", "--months", *mois, "--stockfish", "faux",
            "--work-dir", str(tmp_path), "--output", str(tmp_path / "v1"),
            "--resume",
        ])

    return tmp_path, preparer, lancer, analyses, identite


def test_build_etend_une_reserve_incomplete_sans_recribler(campagne):
    dossier, preparer, lancer, analyses, identite = campagne
    anciennes = preparer("2026-06", incomplet=True)
    nouvelles = preparer("2026-05")
    # Etat equivalent au laptop : tous les candidats initiaux deja cribles.
    with AnnotationStore(dossier, builder._config_store(identite)) as store:
        for position in anciennes:
            store.put(cle_criblage(position), position["test_score"])

    assert lancer(["2026-06"]) == 0

    manifeste = json.loads((dossier / "v1/manifest.json").read_text())
    assert [s["month"] for s in manifeste["sources"]] == ["2026-06", "2026-05"]
    assert manifeste["counts"] == {"positions": 18, "search": 9}
    positions = builder.lire_jsonl_zst(dossier / "v1/positions.jsonl.zst")
    assert collections.Counter((p["phase"], p["wdl_bucket"]) for p in positions) == {
        case: 2 for case in builder.CELLULES}
    assert all(analyses[p["position_id"]] == 0 for p in anciennes)
    assert all(analyses[p["position_id"]] == 1 for p in nouvelles)


def test_reprise_retrouve_les_mois_ajoutes_apres_interruption(
        campagne, monkeypatch):
    dossier, preparer, lancer, analyses, _ = campagne
    preparer("2026-06", incomplet=True)
    preparer("2026-05")
    annotation = builder._cmd_annotate

    def interrompre(args):
        raise RuntimeError("interruption simulee apres extension")

    monkeypatch.setattr(builder, "_cmd_annotate", interrompre)
    assert lancer(["2026-06"]) == 1
    monkeypatch.setattr(builder, "_cmd_annotate", annotation)
    analyses.clear()
    # Si la reprise repart de juin seul, la vraie selection leve de nouveau.
    selection = builder._finalize_reserve

    def verifier_corpus(dossier):
        sources = json.loads((dossier / "sources.json").read_text())
        assert [s["month"] for s in sources] == ["2026-06", "2026-05"]
        selection(dossier)

    monkeypatch.setattr(builder, "_finalize_reserve", verifier_corpus)
    assert lancer(["2026-06"]) == 0
    assert not analyses


def test_extension_s_arrete_avant_le_corpus_de_preentrainement(campagne, capsys):
    dossier, preparer, lancer, _, _ = campagne
    preparer("2026-03", incomplet=True)

    assert lancer(["2026-03"]) == 1

    erreur = capsys.readouterr().err
    assert "2026-03" in erreur and "preentrainement" in erreur
    assert not (dossier / "v1/manifest.json").exists()
    assert not (dossier / "archives/lichess_db_broadcast_2026-02.pgn.zst").exists()


@pytest.mark.parametrize("mois", ["2026-13", "2026-8", "2026-02"])
def test_build_refuse_les_mois_invalides_avant_extraction(tmp_path, monkeypatch, mois):
    def extraction_interdite(args):
        pytest.fail("ne doit pas lancer le telechargement d'un mois invalide")

    monkeypatch.setattr(builder, "_cmd_extract", extraction_interdite)
    assert builder.main(["build", "--months", mois,
                         "--work-dir", str(tmp_path)]) == 1


def test_build_ne_confond_pas_echec_audit_et_manque_de_positions(campagne, monkeypatch):
    dossier, preparer, lancer, _, _ = campagne
    preparer("2026-06")
    profonde = builder.annotate_position

    def fausse_reference(moteur, position, nodes):
        annotee = profonde(moteur, position, nodes)
        if nodes == builder.NODES_AUDIT:
            annotee.update(labels=[{"index": 1, "score": 0.0}], s_best=0.0)
        return annotee

    monkeypatch.setattr(builder, "annotate_position", fausse_reference)
    assert lancer(["2026-06"]) == 1
    assert not (dossier / "v1/manifest.json").exists()
    sources = json.loads((dossier / "sources.json").read_text())
    assert [s["month"] for s in sources] == ["2026-06"]


def test_build_etend_aussi_une_selection_finale_incomplete(campagne, monkeypatch):
    dossier, preparer, lancer, _, _ = campagne
    monkeypatch.setattr(builder, "FACTEUR_RESERVE", 4.0)
    preparer("2026-06")
    preparer("2026-05")
    annotation = builder.annotate_position
    appels = collections.Counter()

    def reference_plus_precise(moteur, position, nodes):
        appels[position["position_id"], nodes] += 1
        # Le criblage classe ces finales en avantage, mais l'analyse de tous
        # les coups les reclasse en positions disputees. Il faut plus de jeux.
        if (position["source_id"] == "lichess-broadcasts-2026-06"
                and position["phase"] == "finale"
                and position["test_score"] == 0.75):
            position = dict(position, test_score=0.5)
        return annotation(moteur, position, nodes)

    monkeypatch.setattr(builder, "annotate_position", reference_plus_precise)
    assert lancer(["2026-06"]) == 0
    manifeste = json.loads((dossier / "v1/manifest.json").read_text())
    assert [s["month"] for s in manifeste["sources"]] == ["2026-06", "2026-05"]
    assert manifeste["counts"]["positions"] == 18
    assert all(nombre == 1 for nombre in appels.values())


@pytest.mark.parametrize("initiaux, supplementaires, attendus", [
    (["2026-06"], ["2026-05", "2026-04"], ["2026-06", "2026-05", "2026-04"]),
    (["2027-01"], ["2026-12"], ["2027-01", "2026-12"]),
])
def test_extension_plusieurs_mois_et_changement_annee(
        campagne, initiaux, supplementaires, attendus):
    dossier, preparer, lancer, _, _ = campagne
    for mois in attendus[:-1]:
        preparer(mois, incomplet=True)
    preparer(supplementaires[-1])
    assert lancer(initiaux) == 0
    manifeste = json.loads((dossier / "v1/manifest.json").read_text())
    assert [s["month"] for s in manifeste["sources"]] == attendus


def test_nouveaux_mois_explicitement_demandes_ne_restaurent_pas_l_ancien_corpus(
        campagne, monkeypatch):
    dossier, preparer, lancer, _, _ = campagne
    preparer("2026-06", incomplet=True)
    preparer("2026-05")
    annotation = builder._cmd_annotate

    def interrompre(args):
        raise RuntimeError("interruption")

    monkeypatch.setattr(builder, "_cmd_annotate", interrompre)
    assert lancer(["2026-06"]) == 1
    monkeypatch.setattr(builder, "_cmd_annotate", annotation)
    preparer("2026-07")
    assert lancer(["2026-07"]) == 0
    manifeste = json.loads((dossier / "v1/manifest.json").read_text())
    assert [s["month"] for s in manifeste["sources"]] == ["2026-07"]
