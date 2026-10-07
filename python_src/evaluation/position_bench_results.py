"""Resultats du banc de positions : chargement, sauvegarde et comparaison.

Le dataset publie est relu et controle avant tout chargement de modele : schema
du manifeste, hash du fichier compresse, contrat de chaque position, unicite
des identifiants, couverture des 10 000 positions et des 256 identifiants de
recherche. Un manifeste de fixture, aux comptes reduits, est refuse en mode
production.

Un resultat vit en deux fichiers : un NPZ de tableaux aplatis et un sidecar
JSON qui porte le rapport, les compteurs, les durees et le hash du NPZ. Les
deux se controlent mutuellement ; `np.load` est toujours appele avec
`allow_pickle=False`. L'ecriture passe par des fichiers temporaires et le
sidecar est publie en dernier.

La comparaison est appariee par identifiant de position, avec un bootstrap de
l'ecart de regret de policy ; elle exige le meme dataset et le meme protocole.
Aucune comparaison entre versions differentes n'est inventee.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md (sections 7
et 8) et docs/superpowers/plans/2026-09-30-position-bench.md (tache 7).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from evaluation.build_position_bench import TAILLE_BANC, TAILLE_SOUS_BANC, lire_jsonl_zst
from evaluation.position_bench_metrics import (
    AnnotatedPosition,
    EvalReport,
    Manifest,
    RawResult,
    aggregate,
    score_distribution,
    validate_manifest,
    validate_position,
)

VERSION_METRIQUES = "position-bench-metrics-v1"

CHAMPS_NPZ = (
    "position_ids",
    "legal_offsets",
    "legal_indices",
    "policy_probs",
    "values",
    "policy_regrets",
    "search_ids",
    "search_offsets",
    "search_indices",
    "search_probs",
    "search_regrets",
)


def sha256_fichier(chemin, taille_bloc: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(chemin, "rb") as flux:
        for bloc in iter(lambda: flux.read(taille_bloc), b""):
            digest.update(bloc)
    return digest.hexdigest()


def _ecrire_atomique(chemin: Path, donnees: bytes) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    with open(temporaire, "wb") as flux:
        flux.write(donnees)
    os.replace(temporaire, chemin)


def _ecrire_json_atomique(chemin: Path, charge) -> None:
    _ecrire_atomique(
        chemin,
        json.dumps(charge, sort_keys=True, indent=2,
                   ensure_ascii=False).encode("utf-8"))


# ============================================================
#                      CHARGEMENT DU BANC
# ============================================================

def load_dataset(path, *, production: bool = True):
    """Relit un banc publie et le controle, sans charger de modele.

    Leve avant toute inference des qu'un hash, un compte, un identifiant ou une
    position ne correspond pas : un dataset incoherent ne doit pas produire de
    mesures partielles.
    """
    dossier = Path(path)
    chemin_manifeste = dossier / "manifest.json"
    if not chemin_manifeste.is_file():
        raise FileNotFoundError(f"manifeste absent : {chemin_manifeste}")
    manifeste = cast(
        Manifest, json.loads(chemin_manifeste.read_text(encoding="utf-8")))
    validate_manifest(manifeste)

    chemin_donnees = dossier / "positions.jsonl.zst"
    if not chemin_donnees.is_file():
        raise FileNotFoundError(f"dataset absent : {chemin_donnees}")
    digest = sha256_fichier(chemin_donnees)
    if digest != manifeste["dataset_sha256"]:
        raise ValueError(
            "dataset_sha256 ne correspond pas au fichier ; dataset modifie "
            "ou tronque")

    records = cast(list[AnnotatedPosition], lire_jsonl_zst(chemin_donnees))
    for record in records:
        validate_position(record)

    ids = [str(record["position_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("identifiants de position dupliques")
    if manifeste["counts"]["positions"] != len(records):
        raise ValueError("compte de positions different du manifeste")

    search_ids = [str(i) for i in manifeste["search_ids"]]
    if len(set(search_ids)) != len(search_ids):
        raise ValueError("identifiants de recherche dupliques")
    connus = set(ids)
    if any(i not in connus for i in search_ids):
        raise ValueError("identifiant de recherche absent du dataset")

    if production and (len(records) != TAILLE_BANC
                       or len(search_ids) != TAILLE_SOUS_BANC):
        raise ValueError(
            "dataset non productif refuse : un manifeste de fixture ne peut "
            "pas alimenter une campagne")
    return manifeste, records


# ============================================================
#                   SAUVEGARDE ET RELECTURE
# ============================================================

def save_result(raw: RawResult, report: EvalReport, output_dir,
                checkpoint_stem: str) -> Path:
    """Ecrit le NPZ puis le sidecar ; refuse d'ecraser un resultat different."""
    dossier = Path(output_dir)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_npz = dossier / f"{checkpoint_stem}.npz"
    chemin_sidecar = dossier / f"{checkpoint_stem}.json"

    tableaux: dict[str, np.ndarray] = {
        nom: np.asarray(raw[nom]) for nom in CHAMPS_NPZ}
    tableaux["dataset_sha256"] = np.asarray(report["dataset_sha256"])
    tableaux["model_sha256"] = np.asarray(report["model_sha256"])
    tableaux["protocol_id"] = np.asarray(report["protocol_id"])

    temporaire = chemin_npz.with_name(chemin_npz.name + ".tmp")
    with open(temporaire, "wb") as flux:
        np.savez_compressed(flux, **tableaux)  # pyright: ignore[reportArgumentType]
    digest = sha256_fichier(temporaire)

    if chemin_npz.exists():
        if sha256_fichier(chemin_npz) != digest:
            temporaire.unlink()
            raise FileExistsError(
                f"un resultat different existe deja : {chemin_npz}")
        temporaire.unlink()
    else:
        os.replace(temporaire, chemin_npz)

    sidecar = {
        "checkpoint_stem": checkpoint_stem,
        "npz_sha256": digest,
        "dataset_sha256": report["dataset_sha256"],
        "model_sha256": report["model_sha256"],
        "protocol_id": report["protocol_id"],
        "metrics_version": VERSION_METRIQUES,
        "report": report,
        "raw": {
            "search_counters": raw["search_counters"],
            "timings": raw["timings"],
        },
    }
    _ecrire_json_atomique(chemin_sidecar, sidecar)
    return chemin_npz


def load_result(path):
    """Relit un NPZ et son sidecar, apres controle mutuel de coherence."""
    chemin = Path(path)
    if chemin.suffix == ".json":
        chemin_sidecar = chemin
        chemin_npz = chemin.with_suffix(".npz")
    else:
        chemin_npz = chemin
        chemin_sidecar = chemin.with_suffix(".json")
    for fichier in (chemin_npz, chemin_sidecar):
        if not fichier.is_file():
            raise FileNotFoundError(f"resultat incomplet : {fichier}")

    sidecar = json.loads(chemin_sidecar.read_text(encoding="utf-8"))
    if sha256_fichier(chemin_npz) != sidecar["npz_sha256"]:
        raise ValueError("npz altere depuis la publication")

    with np.load(chemin_npz, allow_pickle=False) as donnees:
        raw = {nom: donnees[nom] for nom in CHAMPS_NPZ}
        for cle in ("dataset_sha256", "model_sha256", "protocol_id"):
            if str(donnees[cle]) != sidecar[cle]:
                raise ValueError(f"coherence npz/sidecar rompue : {cle}")
    raw["search_counters"] = sidecar["raw"]["search_counters"]
    raw["timings"] = sidecar["raw"]["timings"]
    return cast(RawResult, raw), cast(EvalReport, sidecar["report"])


# ============================================================
#                       AGREGATION
# ============================================================

_CHAMPS_FAMILLE = {
    "policy": ("position_ids", "legal_offsets", "legal_indices",
               "policy_probs"),
    "search": ("search_ids", "search_offsets", "search_indices",
               "search_probs"),
}


def construire_lignes(records, raw: RawResult, famille: str) -> list[dict]:
    """Lignes par position, depuis la representation brute sauvegardee.

    `famille` vaut `policy` ou `search` ; chaque ligne porte les metriques
    lues par `position_bench_metrics.aggregate`.
    """
    champs_ids, champs_offsets, _, champs_probas = _CHAMPS_FAMILLE[famille]
    par_id = {str(record["position_id"]): record for record in records}
    offsets = raw[champs_offsets]
    lignes = []
    for rang, position_id in enumerate(raw[champs_ids]):
        record = par_id.get(str(position_id))
        if record is None:
            raise ValueError(f"resultat : position inconnue {position_id}")
        debut, fin = int(offsets[rang]), int(offsets[rang + 1])
        probas = raw[champs_probas][debut:fin]
        metriques = score_distribution(probas, cast(list, record["labels"]))
        turn = int(record["turn"])
        lignes.append({
            "position_id": str(position_id),
            "phase": record["phase"],
            "wdl_bucket": record["wdl_bucket"],
            "player_type": (record["white_type"] if turn == 0
                            else record["black_type"]),
            "policy_expected_regret": float(metriques["expected_regret"]),
            "policy_argmax_regret": float(metriques["argmax_regret"]),
            "near_best_mass": float(metriques["near_best_mass"]),
            "catastrophic_mass": float(metriques["catastrophic_mass"]),
            "value_pred": float(raw["values"][rang]),
            "value_target": float(2.0 * record["s_best"] - 1.0),
            "cp_regret": metriques["cp_regret"],
        })
    return lignes


def agreger_resultat(manifeste, records, raw: RawResult) -> dict:
    """Metriques globales et par tranche, policy et recherche separees."""
    return {
        "policy": aggregate(construire_lignes(records, raw, "policy")),
        "search": aggregate(construire_lignes(records, raw, "search")),
    }


# ============================================================
#                       COMPARAISON
# ============================================================

def paired_bootstrap(delta, *, samples: int = 10000,
                     seed: int = 20260930) -> tuple[float, float]:
    """Intervalle bootstrap apparie a 95 %, en blocs de 128 tirages."""
    valeurs = np.asarray(delta, dtype=np.float64)
    if valeurs.size == 0:
        raise ValueError("bootstrap : ecart vide")
    generateur = np.random.default_rng(seed)
    moyennes = []
    for debut in range(0, samples, 128):
        taille = min(128, samples - debut)
        indices = generateur.integers(0, valeurs.size,
                                      size=(taille, valeurs.size))
        moyennes.append(valeurs[indices].mean(axis=1))
    echantillons = np.concatenate(moyennes)
    bas, haut = np.quantile(echantillons, [0.025, 0.975], method="linear")
    return float(bas), float(haut)


def compare_results(current: RawResult, previous: RawResult,
                    current_meta: Mapping, previous_meta: Mapping) -> dict:
    """Comparaison appariee du regret de policy entre deux passages.

    Les positions sont appariees par identifiant, ce qui rend l'ordre des
    lignes indifferent. Un dataset ou un protocole different refuse la
    comparaison : elle porterait sur autre chose que le modele.
    """
    for cle in ("dataset_sha256", "protocol_id"):
        if current_meta.get(cle) != previous_meta.get(cle):
            raise ValueError(
                f"comparaison impossible : {cle} different entre les deux "
                "resultats")

    ids_courants = [str(i) for i in current["position_ids"]]
    ids_precedents = [str(i) for i in previous["position_ids"]]
    if (len(set(ids_courants)) != len(ids_courants)
            or len(set(ids_precedents)) != len(ids_precedents)):
        raise ValueError("comparaison impossible : identifiants dupliques")
    if set(ids_courants) != set(ids_precedents):
        raise ValueError(
            "comparaison impossible : les deux resultats ne portent pas les "
            "memes positions")

    index = {identifiant: rang
             for rang, identifiant in enumerate(ids_precedents)}
    delta = np.asarray([
        float(current["policy_regrets"][rang])
        - float(previous["policy_regrets"][index[identifiant]])
        for rang, identifiant in enumerate(ids_courants)
    ], dtype=np.float64)

    bas, haut = paired_bootstrap(delta)
    ameliore = int(np.sum(delta < -1e-12))
    degrade = int(np.sum(delta > 1e-12))
    verdict = "ameliore" if haut < 0 else "degrade" if bas > 0 else "inchange"
    return {
        "count": int(delta.size),
        "delta_policy_regret": float(delta.mean()),
        "delta_policy_regret_low95": bas,
        "delta_policy_regret_high95": haut,
        "verdict": verdict,
        "improved": ameliore,
        "unchanged": int(delta.size) - ameliore - degrade,
        "degraded": degrade,
    }


# ============================================================
#                         W&B
# ============================================================

_METRIQUES_GLOBALES = (
    ("policy_expected_regret", "policy_expected_regret"),
    ("policy_argmax_regret", "policy_argmax_regret"),
    ("near_best_mass", "near_best_mass"),
    ("catastrophic_mass", "catastrophic_mass"),
    ("value_mae", "value_mae"),
    ("value_rmse", "value_rmse"),
    ("value_bias", "value_bias"),
    ("value_correlation", "value_correlation"),
    ("policy_argmax_cp_regret_median", "cp_regret_median"),
    ("policy_argmax_cp_regret_p90", "cp_regret_p90"),
    ("policy_within_20cp", "policy_within_20cp"),
    ("policy_within_50cp", "policy_within_50cp"),
    ("policy_within_100cp", "policy_within_100cp"),
    ("policy_cp_coverage", "cp_coverage"),
)


def _metriques_globales(global_, prefixe: str) -> dict:
    resultat = {}
    for cle_sortie, cle_source in _METRIQUES_GLOBALES:
        valeur = global_.get(cle_source)
        if valeur is not None:
            resultat[f"{prefixe}/{cle_sortie}"] = valeur
    return resultat


def _metriques_tranche(valeurs: Mapping, prefixe: str) -> dict:
    resultat = _metriques_globales(valeurs, prefixe)
    resultat[f"{prefixe}/count"] = valeurs["count"]
    return resultat


def wandb_metrics(report: Mapping) -> dict:
    """Metriques W&B d'un rapport ; les diagnostics nuls sont omis.

    Un statut autre que `ok` ne publie que le contexte, le statut, la duree et
    les compteurs de progression : aucun score partiel ne part vers W&B.
    """
    prefixe = "eval/position"
    contexte = {
        f"{prefixe}/status": report["status"],
        f"{prefixe}/duration_s": report["duration_s"],
        f"{prefixe}/completed_positions": report["completed_positions"],
        f"{prefixe}/completed_search_positions":
            report["completed_search_positions"],
        f"{prefixe}/target_s": report.get("target_s"),
        f"{prefixe}/over_target": report.get("over_target"),
        f"{prefixe}/dataset_version": report["dataset_version"],
        f"{prefixe}/protocol_id": report["protocol_id"],
    }
    contexte = {cle: valeur for cle, valeur in contexte.items()
                if valeur is not None}

    if report["status"] != "ok" or not report.get("metrics"):
        return contexte

    metriques = report["metrics"]
    policy = metriques["policy"]
    recherche = metriques["search"]
    cles = _metriques_globales(policy, prefixe)
    cles[f"{prefixe}/search_expected_regret"] = \
        recherche["policy_expected_regret"]
    cles[f"{prefixe}/search_argmax_regret"] = \
        recherche["policy_argmax_regret"]
    for famille in ("phase", "wdl", "player"):
        for tranche, valeurs in policy[famille].items():
            cles.update(_metriques_tranche(
                valeurs, f"{prefixe}/{famille}/{tranche}"))

    comparaison = metriques.get("comparison") or report.get("comparison")
    if comparaison is not None:
        for cle in ("delta_policy_regret", "delta_policy_regret_low95",
                    "delta_policy_regret_high95"):
            if comparaison.get(cle) is not None:
                cles[f"{prefixe}/{cle}"] = comparaison[cle]

    resultat = {}
    # v1 garde les cles simples ; une autre version n'expose que son prefixe.
    if report["dataset_version"] == "v1":
        resultat.update(cles)
    prefixe_versionne = (
        f"{prefixe}/{report['dataset_version']}/{report['protocol_id']}")
    resultat.update({
        cle.replace(prefixe, prefixe_versionne, 1): valeur
        for cle, valeur in cles.items()
    })
    resultat.update(contexte)
    return resultat


def find_previous(output_dir, dataset_sha256: str, protocol_id: str,
                  iteration: int) -> Path | None:
    """Dernier sidecar compatible d'une iteration strictement anterieure."""
    dossier = Path(output_dir)
    if not dossier.is_dir():
        return None
    meilleur = None
    meilleure_iteration = -1
    for chemin in sorted(dossier.glob("*.json")):
        try:
            sidecar = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if sidecar.get("dataset_sha256") != dataset_sha256:
            continue
        if sidecar.get("protocol_id") != protocol_id:
            continue
        iteration_precedente = sidecar.get("report", {}).get("iteration")
        if iteration_precedente is None or iteration_precedente >= iteration:
            continue
        if iteration_precedente > meilleure_iteration:
            meilleur = chemin
            meilleure_iteration = iteration_precedente
    return meilleur
