"""Tests de la selection, de la publication et de l'orchestration du banc.

Tout est synthetique : les positions de test n'ont que les champs lus par la
selection, et l'annotation est remplacee par un espion pour verifier la
repartition entre moteurs et la reprise sur cache.
"""
import collections
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import evaluation.build_position_bench as build_position_bench
from evaluation.build_position_bench import (
    TAILLE_BANC,
    TAILLE_SOUS_BANC,
    AuditEchoue,
    annoter_positions,
    est_productif,
    finaliser,
    lire_jsonl_zst,
    main,
    quotas_sous_banc,
    select_positions,
    select_search_ids,
    write_dataset,
)
from evaluation.position_bench_teacher import AnnotationStore
from evaluation.position_bench_sources import extract_candidates, read_games

SEL_FINAL = "lapzero-position-bench-v1-final"

PETITS_QUOTAS = {
    ("ouverture", "disputee"): 2,
    ("ouverture", "avantage"): 1,
    ("ouverture", "decisive"): 1,
    ("milieu", "disputee"): 4,
    ("milieu", "avantage"): 2,
    ("milieu", "decisive"): 2,
    ("finale", "disputee"): 2,
    ("finale", "avantage"): 1,
    ("finale", "decisive"): 1,
}

CELLULES = [(phase, bucket)
            for phase in ("ouverture", "milieu", "finale")
            for bucket in ("disputee", "avantage", "decisive")]


def _stable(cle: str) -> int:
    return int(hashlib.sha256(cle.encode()).hexdigest(), 16)


def _record(cle, *, fingerprint=None, event=None, blanc=None, noir=None,
            turn=None):
    position_id = hashlib.sha256(f"position-{cle}".encode()).hexdigest()
    indice = int(cle.rsplit("/", 1)[-1]) if "/" in cle else 0
    return {
        "position_id": position_id,
        "phase": cle.split("/")[0] if isinstance(cle, str) else cle[0],
        "wdl_bucket": cle.split("/")[1] if isinstance(cle, str) else cle[1],
        "game_fingerprint": fingerprint or f"jeu-{cle}",
        "event_id": event if event is not None else f"evt-{_stable(cle) % 100003}",
        "white_id": blanc if blanc is not None else f"blanc-{_stable(cle) % 100003}",
        "black_id": noir if noir is not None else f"noir-{_stable('n' + cle) % 100003}",
        "turn": (indice % 2) if turn is None else turn,
    }


def _vivier(candidats_par_case=6, quotas=PETITS_QUOTAS, **extra):
    vivier = []
    for phase, bucket in CELLULES:
        for i in range(candidats_par_case):
            if quotas.get((phase, bucket), 0) == 0:
                continue
            cle = f"{phase}/{bucket}/{i}"
            vivier.append(_record(cle, **extra))
    return vivier


def _selection_sous_banc():
    records = []
    for cle, quota in quotas_sous_banc().items():
        for i in range(quota):
            records.append(_record(f"{cle[0]}/{cle[1]}/{i}"))
    return records


def _annotee_simple(record, score=0.5):
    annotee = dict(record)
    annotee["labels"] = [{"index": 1, "score": score}]
    annotee["s_best"] = score
    return annotee


# ============================================================
#                        SELECTION
# ============================================================

def test_selection_ne_depend_pas_de_l_ordre():
    vivier = _vivier()

    a = select_positions(vivier, PETITS_QUOTAS, salt=SEL_FINAL,
                         event_cap=100, player_cap=20)
    b = select_positions(list(reversed(vivier)), PETITS_QUOTAS,
                         salt=SEL_FINAL, event_cap=100, player_cap=20)

    assert [p["position_id"] for p in a] == [p["position_id"] for p in b]
    assert len({p["game_fingerprint"] for p in a}) == len(a)


def test_selection_respecte_les_neuf_quotas_et_les_couleurs():
    rapport = {}
    selection = select_positions(_vivier(), PETITS_QUOTAS, salt=SEL_FINAL,
                                 event_cap=100, player_cap=20,
                                 rapport=rapport)

    comptes = collections.Counter(
        (p["phase"], p["wdl_bucket"]) for p in selection)
    for cle, quota in PETITS_QUOTAS.items():
        assert comptes[cle] == quota
    assert sum(PETITS_QUOTAS.values()) == 16
    assert rapport["complet"] is True
    assert rapport["couleurs"]["blancs"] == 8
    assert rapport["couleurs"]["noirs"] == 8
    assert len({p["game_fingerprint"] for p in selection}) == len(selection)


def test_selection_production_applique_la_repartition_approuvee():
    attendus = {
        ("ouverture", "disputee"): 1100,
        ("ouverture", "avantage"): 500,
        ("ouverture", "decisive"): 400,
        ("milieu", "disputee"): 3600,
        ("milieu", "avantage"): 1000,
        ("milieu", "decisive"): 1400,
        ("finale", "disputee"): 1250,
        ("finale", "avantage"): 250,
        ("finale", "decisive"): 500,
    }
    rapport = {}

    selection = select_positions(
        _vivier(candidats_par_case=4000), build_position_bench.QUOTAS,
        salt=SEL_FINAL, event_cap=100, player_cap=20, rapport=rapport)

    comptes = collections.Counter(
        (p["phase"], p["wdl_bucket"]) for p in selection)
    assert comptes == attendus
    assert len(selection) == TAILLE_BANC == 10_000
    assert len({p["game_fingerprint"] for p in selection}) == 10_000
    assert rapport["complet"] is True
    assert 4500 <= rapport["couleurs"]["blancs"] <= 5500


def test_selection_refuse_un_evenement_au_dela_du_plafond():
    vivier = []
    for i in range(6):
        vivier.append(_record(f"ouverture/disputee/{i}", event="evt-unique",
                              fingerprint=f"jeu-cible-{i}"))
    for phase, bucket in CELLULES:
        if (phase, bucket) != ("ouverture", "disputee"):
            for i in range(6):
                vivier.append(_record(f"{phase}/{bucket}/{i}"))
    rapport = {}

    select_positions(vivier, PETITS_QUOTAS, salt=SEL_FINAL, event_cap=1,
                     player_cap=100, rapport=rapport)

    assert rapport["manquants"] == {"ouverture/disputee": 1}
    assert rapport["complet"] is False


def test_selection_refuse_un_joueur_au_dela_du_plafond():
    vivier = []
    for i in range(6):
        vivier.append(_record(f"ouverture/disputee/{i}", blanc="unique",
                              fingerprint=f"jeu-cible-{i}"))
    for phase, bucket in CELLULES:
        if (phase, bucket) != ("ouverture", "disputee"):
            for i in range(6):
                vivier.append(_record(f"{phase}/{bucket}/{i}"))
    rapport = {}

    select_positions(vivier, PETITS_QUOTAS, salt=SEL_FINAL, event_cap=100,
                     player_cap=1, rapport=rapport)

    assert rapport["manquants"] == {"ouverture/disputee": 1}
    assert rapport["complet"] is False


def test_selection_ne_depasse_pas_55_pour_cent_d_une_couleur():
    vivier = _vivier(turn=0)
    rapport = {}

    selection = select_positions(vivier, PETITS_QUOTAS, salt=SEL_FINAL,
                                 event_cap=100, player_cap=100,
                                 rapport=rapport)

    assert len(selection) == 8
    assert rapport["complet"] is False
    assert rapport["manquants"]
    assert len(selection) <= 0.55 * sum(PETITS_QUOTAS.values())


def test_selection_vide_le_rapport_sans_lever():
    rapport = {}

    selection = select_positions([], PETITS_QUOTAS, salt=SEL_FINAL,
                                 event_cap=100, player_cap=20,
                                 rapport=rapport)

    assert selection == []
    assert rapport["complet"] is False
    assert rapport["manquants"]


# ============================================================
#                        SOUS-BANC
# ============================================================

def test_les_quotas_du_sous_banc_sont_arrondis_par_plus_grands_restes():
    quotas = quotas_sous_banc()

    assert sum(quotas.values()) == TAILLE_SOUS_BANC
    assert quotas[("ouverture", "disputee")] == 28
    assert quotas[("ouverture", "avantage")] == 13
    assert quotas[("ouverture", "decisive")] == 10
    assert quotas[("milieu", "disputee")] == 92
    assert quotas[("milieu", "avantage")] == 26
    assert quotas[("milieu", "decisive")] == 36
    assert quotas[("finale", "disputee")] == 32
    assert quotas[("finale", "avantage")] == 6
    assert quotas[("finale", "decisive")] == 13


def test_le_sous_banc_a_256_identifiants_et_ses_proportions():
    selection = _selection_sous_banc()

    ids = select_search_ids(selection)

    assert len(ids) == TAILLE_SOUS_BANC
    assert len(set(ids)) == len(ids)
    par_id = {record["position_id"]: record for record in selection}
    comptes = collections.Counter(
        (par_id[i]["phase"], par_id[i]["wdl_bucket"]) for i in ids)
    for cle, quota in quotas_sous_banc().items():
        assert comptes[cle] == quota


def test_le_sous_banc_ne_depend_pas_de_l_ordre():
    selection = _selection_sous_banc()

    reference = select_search_ids(selection)
    melange = select_search_ids(list(reversed(selection)))

    assert melange == reference


# ============================================================
#                     AUDIT ET PUBLICATION
# ============================================================

def test_finaliser_rend_les_identifiants_et_l_audit():
    selection = [_annotee_simple(record)
                 for record in _selection_sous_banc()]

    search_ids, audit = finaliser(selection, selection)

    assert len(search_ids) == TAILLE_SOUS_BANC
    assert audit["passed"] is True
    assert audit["count"] == TAILLE_SOUS_BANC


def test_finaliser_refuse_un_audit_qui_echoue():
    selection = [_annotee_simple(record)
                 for record in _selection_sous_banc()]
    profonde = [_annotee_simple(record)
                for record in _selection_sous_banc()]
    for annotee in profonde[:20]:
        annotee["labels"] = [{"index": 1, "score": 0.0}]
        annotee["s_best"] = 0.0

    with pytest.raises(AuditEchoue):
        finaliser(selection, profonde)


SOURCE = {
    "source_id": "lichess-broadcasts-2026-08",
    "url": "https://database.lichess.org/broadcasts/x-2026-08.pgn.zst",
    "month": "2026-08",
    "archive_sha256": "c" * 64,
    "license": "CC BY-SA 4.0",
    "attribution": "Lichess",
}


def _manifeste(created="2026-09-30T00:00:00+00:00"):
    return {
        "schema_version": 1,
        "dataset_version": "v1",
        "dataset_sha256": "0" * 64,
        "training_forbidden": True,
        "sources": [SOURCE],
        "counts": {"positions": 3, "search": 1},
        "quotas": {"ouverture/disputee": 3},
        "stockfish": {"name": "SF", "binary_sha256": "a" * 64},
        "audit": {"passed": True, "count": 3},
        "search_ids": ["p1"],
        "protocol_defaults": {},
        "builder_revision": "test",
        "created": created,
    }


def test_write_dataset_ecrit_jsonl_manifeste_et_readme(tmp_path):
    records = [_record("finale/decisive/0"), _record("ouverture/disputee/0")]

    chemin = write_dataset(records, _manifeste(), tmp_path / "v1")

    assert chemin == tmp_path / "v1" / "manifest.json"
    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    assert manifeste["dataset_sha256"] == hashlib.sha256(
        (tmp_path / "v1" / "positions.jsonl.zst").read_bytes()).hexdigest()
    relus = lire_jsonl_zst(tmp_path / "v1" / "positions.jsonl.zst")
    assert [r["position_id"] for r in relus] == sorted(
        r["position_id"] for r in records)
    readme = (tmp_path / "v1" / "README.md").read_text(encoding="utf-8")
    assert "training_forbidden" in readme
    assert "CC BY-SA" in readme


def test_write_dataset_refuse_une_version_differente(tmp_path):
    dossier = tmp_path / "v1"
    write_dataset([_record("ouverture/disputee/0")], _manifeste(), dossier)

    with pytest.raises(FileExistsError):
        write_dataset([_record("ouverture/disputee/1")], _manifeste(), dossier)


def test_la_date_de_fabrication_ne_change_pas_le_hash(tmp_path):
    records = [_record("ouverture/disputee/0")]

    write_dataset(records, _manifeste(created="2026-01-01"), tmp_path / "un")
    write_dataset(records, _manifeste(created="2026-12-31"), tmp_path / "deux")

    assert ((tmp_path / "un" / "positions.jsonl.zst").read_bytes()
            == (tmp_path / "deux" / "positions.jsonl.zst").read_bytes())
    manifeste_un = json.loads(
        (tmp_path / "un" / "manifest.json").read_text(encoding="utf-8"))
    manifeste_deux = json.loads(
        (tmp_path / "deux" / "manifest.json").read_text(encoding="utf-8"))
    assert manifeste_un["dataset_sha256"] == manifeste_deux["dataset_sha256"]


def test_est_productif_exige_les_tailles_completes():
    manifeste = _manifeste()

    assert est_productif(manifeste) is False
    manifeste["counts"] = {"positions": TAILLE_BANC,
                           "search": TAILLE_SOUS_BANC}
    assert est_productif(manifeste) is True


# ============================================================
#                      ORCHESTRATION
# ============================================================

def test_annoter_positions_repartit_et_reprend_sur_le_cache(tmp_path,
                                                            monkeypatch):
    appels = []

    def fausse_annotation(moteur, position, nodes):
        appels.append((moteur, position["position_id"], nodes))
        return {"position_id": position["position_id"], "labels": []}

    monkeypatch.setattr(build_position_bench, "annotate_position",
                        fausse_annotation)
    positions = [{"position_id": f"p{i}"} for i in range(5)]
    moteurs = [object(), object()]
    store = AnnotationStore(tmp_path, {"schema": 1})
    sortie = tmp_path / "annotated.jsonl.zst"
    etats = []
    try:
        resultat = annoter_positions(moteurs, positions, 200_000, store,
                                     sortie, progress=etats.append)

        assert [r["position_id"] for r in resultat] == [
            p["position_id"] for p in positions]
        assert len(appels) == 5
        assert {moteur for moteur, _, _ in appels} == set(moteurs)
        # Les etats arrivent dans l'ordre d'achevement, pas de soumission.
        assert sorted(etat["faites"] for etat in etats) == [1, 2, 3, 4, 5]
        assert all(etat["total"] == 5 and etat["reprises"] == 0
                   for etat in etats)

        # Reprise : tout est en cache, aucun appel ni progression.
        appels.clear()
        etats.clear()
        annoter_positions([moteurs[0]], positions, 200_000, store, sortie,
                          progress=etats.append)
        assert appels == []
        assert etats == []
    finally:
        store.close()


def test_annoter_positions_est_independant_du_nombre_de_moteurs(
        tmp_path, monkeypatch):
    def fausse_annotation(moteur, position, nodes):
        return {"position_id": position["position_id"]}

    monkeypatch.setattr(build_position_bench, "annotate_position",
                        fausse_annotation)
    positions = [{"position_id": f"p{i}"} for i in range(6)]
    sorties = []
    for nombre in (1, 4):
        store = AnnotationStore(tmp_path / f"n{nombre}", {"schema": 1})
        sortie = tmp_path / f"n{nombre}.jsonl.zst"
        try:
            resultat = annoter_positions(
                [object() for _ in range(nombre)], positions, 200_000, store,
                sortie)
        finally:
            store.close()
        sorties.append([r["position_id"] for r in resultat])

    attendu = [f"p{i}" for i in range(6)]
    assert sorties[0] == attendu
    assert sorties[1] == attendu


def test_les_moteurs_ne_sont_identifies_qu_une_fois(monkeypatch):
    appels = {"identifications": 0}

    def fausse_identification(moteur, chemin, **options):
        appels["identifications"] += 1
        return {"name": "faux", "networks": []}

    monkeypatch.setattr(build_position_bench, "identifier_stockfish",
                        fausse_identification)
    monkeypatch.setattr(build_position_bench, "configurer_moteur",
                        lambda moteur: None)
    monkeypatch.setattr(
        build_position_bench.chess.engine.SimpleEngine, "popen_uci",
        staticmethod(lambda chemin: object()))
    args = SimpleNamespace(stockfish="x", workers=4, work_dir="travail")

    moteurs, identite = build_position_bench._ouvrir_moteurs(args)

    assert len(moteurs) == 4
    assert appels["identifications"] == 1
    assert identite == {"name": "faux", "networks": []}


def test_main_echoue_proprement_sur_un_dossier_absent(tmp_path, capsys):
    code = main(["finalize", "--work-dir", str(tmp_path / "absent")])

    assert code == 1
    assert "erreur" in capsys.readouterr().err


def test_reprise_extraction_filtre_les_anciens_candidats_terminaux(tmp_path):
    source = dict(SOURCE, url=build_position_bench.URL_BROADCASTS.format(
        mois="2026-08"))
    partie_path = RACINE / "tests/data/position_bench/broadcasts.pgn"
    valide = extract_candidates(read_games(partie_path, source), source)[0]
    repetition = dict(valide, position_id="a" * 64, repetition_count=3)
    cinquante = dict(valide, position_id="b" * 64, halfmove_clock=100)
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "lichess_db_broadcast_2026-08.pgn.zst").write_bytes(b"archive")
    cache = tmp_path / "candidates_2026-08.jsonl.zst"
    build_position_bench.ecrire_jsonl_zst([valide, repetition, cinquante], cache)
    args = SimpleNamespace(work_dir=str(tmp_path), months=["2026-08"],
                           resume=True)

    build_position_bench._cmd_extract(args)

    for chemin in (cache, tmp_path / "candidates.jsonl.zst"):
        assert [c["position_id"] for c in lire_jsonl_zst(chemin)] == [
            valide["position_id"]]


def test_erreur_worker_arrete_les_taches_non_commencees():
    traites = []

    def traitement(moteur, element):
        traites.append(element)
        if element == 0:
            raise RuntimeError("moteur en panne")
        return element

    with pytest.raises(RuntimeError, match="moteur en panne"):
        build_position_bench._en_parallele(
            [object()], list(range(20)), traitement)

    assert traites == [0]


def test_erreur_worker_conserve_le_resultat_du_worker_encore_actif():
    actif = threading.Event()
    enregistrer = []

    def traitement(moteur, element):
        if element == 0:
            assert actif.wait(2)
            raise RuntimeError("moteur en panne")
        if element == 1:
            actif.set()
        return element

    with pytest.raises(RuntimeError, match="moteur en panne"):
        build_position_bench._en_parallele(
            [object(), object()], list(range(20)), traitement,
            a_reception=lambda element, resultat: enregistrer.append(resultat))

    assert 1 in enregistrer
    assert len(enregistrer) < 19


class MoteurReprise:
    def __init__(self):
        self.ferme = False

    def quit(self):
        self.ferme = True


def test_erreur_worker_retente_sur_un_nouveau_moteur():
    initial = MoteurReprise()
    moteurs = [initial]
    ouvertures = []

    def rouvrir():
        moteur = MoteurReprise()
        ouvertures.append(moteur)
        return moteur

    def traitement(moteur, element):
        if moteur is initial:
            raise RuntimeError("annotation invalide")
        return element * 2

    resultats = build_position_bench._en_parallele(
        moteurs, [1, 2], traitement, reouvrir_moteur=rouvrir)

    assert resultats == [2, 4]
    assert initial.ferme
    assert len(ouvertures) == 1
    assert moteurs[0] is ouvertures[0]


def test_deuxieme_erreur_worker_est_fatale():
    initial = MoteurReprise()
    moteurs = [initial]
    ouvertures = []
    traites = []

    def rouvrir():
        moteur = MoteurReprise()
        ouvertures.append(moteur)
        return moteur

    def traitement(moteur, element):
        traites.append(element)
        raise RuntimeError("annotation toujours invalide")

    with pytest.raises(RuntimeError, match="annotation toujours invalide"):
        build_position_bench._en_parallele(
            moteurs, [1, 2, 3], traitement, reouvrir_moteur=rouvrir)

    assert initial.ferme
    assert len(ouvertures) == 1
    assert traites == [1, 1]
