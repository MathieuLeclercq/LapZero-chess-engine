"""Contrats et metriques pures du banc externe de positions.

Aucun import du moteur, de torch ni de W&B : ce module manipule seulement des
dictionnaires JSON et des tableaux numpy, et se teste donc sans poids ni
session ONNX. Il sert de reference de schema aux autres modules du banc.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md pour les
definitions, et docs/superpowers/plans/2026-09-30-position-bench.md (tache 1).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import NotRequired, TypedDict

import numpy as np

TAILLE_POLICY = 4672
SCHEMA_VERSION = 1

# Tolerances fixees par le plan : regrets et probabilites viennent de calculs
# flottants, les seuils doivent rester inclusifs malgre l'arrondi.
TOLERANCE = 1e-12
TOLERANCE_SOMME = 1e-6

SEUIL_PROXIME = 0.02
SEUIL_CATASTROPHIQUE = 0.20
SEUILS_CP = (20, 50, 100)

PHASES = ("ouverture", "milieu", "finale")
WDL_BUCKETS = ("disputee", "avantage", "decisive")
PLAYER_TYPES = ("human", "bot", "unknown")

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class ContratInvalide(ValueError):
    """Un enregistrement ne respecte pas le contrat JSON du banc."""


# ============================================================
#                     CONTRATS JSON
# ============================================================

class SourceInfo(TypedDict):
    source_id: str
    url: str
    month: str
    archive_sha256: str
    license: str
    attribution: str


class Position(TypedDict):
    position_id: str
    game_id: str
    game_fingerprint: str
    source_id: str
    event_id: str
    white_id: str
    black_id: str
    white_elo: int
    black_elo: int
    white_type: str
    black_type: str
    game_date: str
    start_fen: str
    moves_uci: list[str]
    fen: str
    ply: int
    turn: int
    halfmove_clock: int
    repetition_count: int
    phase: str
    legal_indices: list[int]


class MoveLabel(TypedDict):
    uci: str
    index: int
    wdl: tuple[int, int, int]
    score: float
    cp: int | None
    mate: int | None
    depth: int
    nodes: int


class AnnotatedPosition(Position):
    labels: list[MoveLabel]
    s_best: float
    wdl_bucket: str


class Manifest(TypedDict):
    schema_version: int
    dataset_version: str
    dataset_sha256: str
    training_forbidden: bool
    sources: list[SourceInfo]
    counts: dict
    quotas: dict
    stockfish: dict
    audit: dict
    search_ids: list[str]
    protocol_defaults: dict
    builder_revision: str
    created: NotRequired[str]


class RawResult(TypedDict):
    position_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_indices: np.ndarray
    policy_probs: np.ndarray
    values: np.ndarray
    policy_regrets: np.ndarray
    search_ids: np.ndarray
    search_offsets: np.ndarray
    search_indices: np.ndarray
    search_probs: np.ndarray
    search_regrets: np.ndarray
    search_counters: list[dict]
    timings: dict


class EvalReport(TypedDict):
    status: str
    dataset_version: str
    dataset_sha256: str
    model_sha256: str
    protocol_id: str
    completed_positions: int
    completed_search_positions: int
    duration_s: float
    timings: dict
    metrics: dict
    comparison: dict | None
    result_path: str
    error: str | None
    target_s: NotRequired[float]
    over_target: NotRequired[bool]
    metrics_version: NotRequired[str]
    iteration: NotRequired[int | None]
    global_step: NotRequired[int | None]


@dataclass(frozen=True)
class SearchConfig:
    """Parametres de recherche du sous-banc MCTS, figes pour un protocole."""

    simulations: int = 384
    batch_size: int = 8
    workers: int = 8
    fixed_batch: bool = True
    virtual_loss: int = 2
    fpu: float = 0.30
    collision_attempts: int = 4
    c_puct: float = 1.4
    tt_size: int = 8192
    cache_history_depth: int = 0


@dataclass(frozen=True)
class EvalConfig:
    """Parametres d'un passage d'evaluation recurrent."""

    search: SearchConfig = SearchConfig()
    policy_batch_size: int = 1024
    target_s: float = 300.0
    use_gpu: bool = True
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20260930


# ============================================================
#              VALIDATION DE SCHEMA
# ============================================================

def _champ(record: Mapping[str, object], cle: str, description: str):
    if cle not in record:
        raise ContratInvalide(f"{description} : champ obligatoire manquant {cle!r}")
    return record[cle]


def _chaine(record: Mapping[str, object], cle: str, description: str) -> str:
    valeur = _champ(record, cle, description)
    if not isinstance(valeur, str) or not valeur:
        raise ContratInvalide(
            f"{description} : {cle!r} doit etre une chaine non vide")
    return valeur


def _entier(record: Mapping[str, object], cle: str, description: str, *,
            minimum: int | None = None,
            maximum: int | None = None) -> int:
    valeur = _champ(record, cle, description)
    if isinstance(valeur, bool) or not isinstance(valeur, int):
        raise ContratInvalide(f"{description} : {cle!r} doit etre un entier")
    if minimum is not None and valeur < minimum:
        raise ContratInvalide(
            f"{description} : {cle!r} = {valeur} inferieur a {minimum}")
    if maximum is not None and valeur > maximum:
        raise ContratInvalide(
            f"{description} : {cle!r} = {valeur} superieur a {maximum}")
    return valeur


def _liste(record: Mapping[str, object], cle: str, description: str) -> list:
    valeur = _champ(record, cle, description)
    if not isinstance(valeur, list):
        raise ContratInvalide(f"{description} : {cle!r} doit etre une liste")
    return valeur


def _flottant(record: Mapping[str, object], cle: str, description: str) -> float:
    valeur = _champ(record, cle, description)
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise ContratInvalide(f"{description} : {cle!r} doit etre un nombre")
    resultat = float(valeur)
    if not math.isfinite(resultat):
        raise ContratInvalide(f"{description} : {cle!r} n'est pas fini")
    return resultat


def _trois_wdl(wdl) -> tuple[int, int, int]:
    """Verifie un triplet WDL et le rend type, effectif total strictement positif."""
    if not isinstance(wdl, (tuple, list)) or len(wdl) != 3:
        raise ContratInvalide("WDL : triplet (victoires, nulles, defaites) attendu")
    for valeur in wdl:
        if isinstance(valeur, bool) or not isinstance(valeur, int) or valeur < 0:
            raise ContratInvalide("WDL : chaque effectif doit etre un entier positif")
    if sum(wdl) == 0:
        raise ContratInvalide("WDL : effectif total nul")
    return (int(wdl[0]), int(wdl[1]), int(wdl[2]))


def _valider_wdl(wdl) -> int:
    """Verifie un triplet WDL et rend son total, strictement positif."""
    return sum(_trois_wdl(wdl))


def validate_move_label(label: Mapping[str, object],
                        description: str = "etiquette") -> None:
    """Verifie une etiquette de coup : WDL, score coherent, cp ou mat."""
    if not isinstance(label, Mapping):
        raise ContratInvalide(f"{description} : un objet JSON est attendu")

    wdl = _trois_wdl(_champ(label, "wdl", description))
    score = _flottant(label, "score", description)
    attendu = (wdl[0] + 0.5 * wdl[1]) / sum(wdl)
    if abs(score - attendu) > 1e-9:
        raise ContratInvalide(
            f"{description} : score {score} incoherent avec la WDL {wdl}")

    _chaine(label, "uci", description)
    _entier(label, "index", description, minimum=0, maximum=TAILLE_POLICY - 1)

    cp = _champ(label, "cp", description)
    mate = _champ(label, "mate", description)
    if (cp is None) == (mate is None):
        raise ContratInvalide(
            f"{description} : exactement un de cp et mate doit etre renseigne")
    if cp is not None and (isinstance(cp, bool) or not isinstance(cp, int)):
        raise ContratInvalide(f"{description} : cp doit etre un entier ou null")
    if (mate is not None
            and (isinstance(mate, bool) or not isinstance(mate, int)
                 or mate == 0)):
        raise ContratInvalide(
            f"{description} : mate doit etre un entier non nul ou null")

    _entier(label, "depth", description, minimum=0)
    _entier(label, "nodes", description, minimum=0)


def _valider_annotations(record: Mapping[str, object],
                         legal_indices: list[int], description: str) -> None:
    labels = _liste(record, "labels", description)
    if not labels:
        raise ContratInvalide(f"{description} : aucune etiquette")

    indices = []
    for label in labels:
        validate_move_label(label, f"{description}, etiquette")
        indices.append(label["index"])
    if any(b <= a for a, b in pairwise(indices)):
        raise ContratInvalide(
            f"{description} : etiquettes non triees ou index dupliques")
    if indices != legal_indices:
        manquants = sorted(set(legal_indices) - set(indices))
        en_trop = sorted(set(indices) - set(legal_indices))
        raise ContratInvalide(
            f"{description} : couverture des coups legaux incorrecte "
            f"(manquants {manquants}, superflus {en_trop})")

    s_best = _flottant(record, "s_best", description)
    meilleur = max(label["score"] for label in labels)
    if abs(s_best - meilleur) > 1e-9:
        raise ContratInvalide(
            f"{description} : s_best {s_best} ne correspond pas au meilleur score")

    bucket = _chaine(record, "wdl_bucket", description)
    if bucket not in WDL_BUCKETS:
        raise ContratInvalide(f"{description} : wdl_bucket inconnu {bucket!r}")
    if bucket != wdl_category(s_best):
        raise ContratInvalide(
            f"{description} : wdl_bucket {bucket!r} ne correspond pas a "
            f"s_best {s_best}")


def validate_position(record: Mapping[str, object]) -> None:
    """Verifie une position, annotee ou non, contre le contrat du banc.

    Une position avec plus de 128 coups legaux est valide : la limite
    TT_MAX_MOVES du moteur ne doit jamais devenir un filtre du banc. Quand le
    champ labels est present, chaque coup legal doit avoir exactement une
    etiquette.
    """
    if not isinstance(record, dict):
        raise ContratInvalide("position : un objet JSON est attendu")
    description = f"position {record.get('position_id', '?')!r}"

    for cle in ("position_id", "game_id", "source_id", "event_id",
                "white_id", "black_id", "game_date", "start_fen", "fen"):
        _chaine(record, cle, description)

    fingerprint = _chaine(record, "game_fingerprint", description)
    if not _SHA256.match(fingerprint):
        raise ContratInvalide(
            f"{description} : game_fingerprint doit etre un SHA-256")

    for cle in ("white_elo", "black_elo"):
        _entier(record, cle, description, minimum=0)
    for cle in ("white_type", "black_type"):
        valeur = _chaine(record, cle, description)
        if valeur not in PLAYER_TYPES:
            raise ContratInvalide(
                f"{description} : {cle!r} inconnu {valeur!r}")

    _entier(record, "ply", description, minimum=0)
    _entier(record, "turn", description, minimum=0, maximum=1)
    _entier(record, "halfmove_clock", description, minimum=0)
    _entier(record, "repetition_count", description, minimum=1, maximum=3)

    phase = _chaine(record, "phase", description)
    if phase not in PHASES:
        raise ContratInvalide(f"{description} : phase inconnue {phase!r}")

    moves = _liste(record, "moves_uci", description)
    for coup in moves:
        if not isinstance(coup, str) or not coup:
            raise ContratInvalide(
                f"{description} : un coup UCI vide ou non textuel")

    legal_indices = _liste(record, "legal_indices", description)
    for index in legal_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ContratInvalide(
                f"{description} : un index de coup non entier")
        if not 0 <= index < TAILLE_POLICY:
            raise ContratInvalide(
                f"{description} : index {index} hors de [0, {TAILLE_POLICY})")
    if legal_indices != sorted(legal_indices):
        raise ContratInvalide(
            f"{description} : legal_indices non tries ou dupliques")

    if "labels" in record:
        _valider_annotations(record, legal_indices, description)


def _valider_source(source: Mapping[str, object], description: str) -> None:
    if not isinstance(source, dict):
        raise ContratInvalide(f"{description} : un objet JSON est attendu")
    for cle in ("source_id", "url", "month", "license", "attribution"):
        _chaine(source, cle, description)
    hash_archive = _chaine(source, "archive_sha256", description)
    if not _SHA256.match(hash_archive):
        raise ContratInvalide(
            f"{description} : archive_sha256 doit etre un SHA-256")


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Verifie le manifeste d'une version publiee du banc."""
    if not isinstance(manifest, dict):
        raise ContratInvalide("manifeste : un objet JSON est attendu")

    version = _entier(manifest, "schema_version", "manifeste", minimum=1)
    if version != SCHEMA_VERSION:
        raise ContratInvalide(
            f"manifeste : schema_version {version} inconnu, "
            f"{SCHEMA_VERSION} attendu")

    _chaine(manifest, "dataset_version", "manifeste")
    hash_dataset = _chaine(manifest, "dataset_sha256", "manifeste")
    if not _SHA256.match(hash_dataset):
        raise ContratInvalide("manifeste : dataset_sha256 doit etre un SHA-256")

    if manifest.get("training_forbidden") is not True:
        raise ContratInvalide(
            "manifeste : training_forbidden doit valoir true")

    sources = _liste(manifest, "sources", "manifeste")
    if not sources:
        raise ContratInvalide("manifeste : aucune source")
    for source in sources:
        _valider_source(source, "manifeste, source")

    for cle in ("counts", "quotas", "stockfish", "audit", "protocol_defaults"):
        valeur = _champ(manifest, cle, "manifeste")
        if not isinstance(valeur, dict):
            raise ContratInvalide(f"manifeste : {cle!r} doit etre un objet")

    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ContratInvalide("manifeste : counts doit etre un objet")
    for cle, valeur in counts.items():
        if isinstance(valeur, bool) or not isinstance(valeur, int) or valeur < 0:
            raise ContratInvalide(
                f"manifeste : counts[{cle!r}] doit etre un entier positif")

    search_ids = _liste(manifest, "search_ids", "manifeste")
    if not search_ids:
        raise ContratInvalide("manifeste : aucun identifiant de recherche")
    if any(not isinstance(i, str) or not i for i in search_ids):
        raise ContratInvalide(
            "manifeste : un identifiant de recherche vide ou non textuel")
    if len(set(search_ids)) != len(search_ids):
        raise ContratInvalide("manifeste : identifiants de recherche dupliques")

    _chaine(manifest, "builder_revision", "manifeste")


# ============================================================
#                     METRIQUES PURES
# ============================================================

def wdl_score(wdl: tuple[int, int, int]) -> float:
    """Esperance de resultat S = (V + 0,5 * N) / (V + N + D), dans [0, 1]."""
    total = _valider_wdl(wdl)
    return (wdl[0] + 0.5 * wdl[1]) / total


def wdl_value(wdl: tuple[int, int, int]) -> float:
    """Cible scalaire V = 2 * S - 1 = (V - D) / (V + N + D), dans [-1, 1]."""
    total = _valider_wdl(wdl)
    return (wdl[0] - wdl[2]) / total


def wdl_category(score: float) -> str:
    """Categorie de la spec : disputee, avantage ou decisive.

    Bornes de la spec (revision 2026-09-30) : disputee pour
    0,35 <= S <= 0,65 ; avantage pour 0,15 <= S < 0,35 ou 0,65 < S <= 0,85 ;
    decisive pour S < 0,15 ou S > 0,85.
    """
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ContratInvalide(f"categorie : esperance invalide {score!r}")
    if 0.35 - TOLERANCE <= score <= 0.65 + TOLERANCE:
        return "disputee"
    if (0.15 - TOLERANCE <= score < 0.35) or (0.65 < score <= 0.85 + TOLERANCE):
        return "avantage"
    return "decisive"


def _rang_cp(label: dict) -> tuple[int, float]:
    """Cle de classement d'une etiquette cp/mat, superieure = meilleure.

    Ordre de la spec : mat perdant, puis cp, puis mat gagnant. Les distances
    de mat ne departagent que des mats : le mat gagnant le plus proche est le
    meilleur, le mat perdant le plus lointain aussi.
    """
    mate = label.get("mate")
    if mate is not None:
        if mate > 0:
            return (2, -float(mate))
        return (0, float(mate))
    cp = label.get("cp")
    if cp is None:
        raise ContratInvalide("etiquette : ni cp ni mat")
    return (1, float(cp))


def _valider_etiquettes(labels: list, scores: np.ndarray) -> list[int]:
    indices = []
    for label in labels:
        if not isinstance(label, dict):
            raise ContratInvalide("distribution : une etiquette non objet")
        for cle in ("index", "score", "cp", "mate"):
            if cle not in label:
                raise ContratInvalide(
                    f"distribution : etiquette sans champ {cle!r}")
        if isinstance(label["index"], bool) or not isinstance(label["index"], int):
            raise ContratInvalide("distribution : index non entier")
        indices.append(label["index"])
    if any(b <= a for a, b in pairwise(indices)):
        raise ContratInvalide(
            "distribution : etiquettes non triees ou index dupliques")
    if not np.all(np.isfinite(scores)):
        raise ContratInvalide("distribution : score non fini")
    return indices


def _cp_regret(labels: list, scores: np.ndarray, choisi: int) -> float | None:
    """Regret en centipions du coup choisi, null des qu'un mat est en jeu.

    Le meilleur coup est celui de score maximal, departage par l'ordre
    cp/mat : un argmax de WDL saturee entre plusieurs coups a S = 1 ne doit
    pas designer un coup au hasard. Si le meilleur ou le coup choisi est un
    mat, la position sort du diagnostic cp, qui n'a pas d'echelle finie.
    """
    candidats = [label for label, score in zip(labels, scores)
                 if abs(float(score) - float(scores.max())) <= TOLERANCE]
    if not candidats:
        raise ContratInvalide("distribution : aucun coup de score maximal")
    meilleur = max(candidats, key=_rang_cp)
    choisi_label = labels[choisi]
    if meilleur["mate"] is not None or choisi_label["mate"] is not None:
        return None
    return float(meilleur["cp"]) - float(choisi_label["cp"])


def score_distribution(probabilities, labels: list[dict]) -> dict:
    """Metriques d'une distribution de policy sur des coups annotes.

    La distribution porte sur les seuls coups legaux, dans l'ordre des
    etiquettes (deja triees par index). Elle est validee avant tout calcul :
    une distribution invalide leve, elle n'est jamais renormalisee ni ecretee
    en silence. En cas d'egalite de probabilite, l'argmax retient le plus
    petit index, puisque les etiquettes sont triees.
    """
    if not labels:
        raise ContratInvalide("distribution : aucune etiquette")

    scores = np.asarray([label.get("score") for label in labels],
                        dtype=np.float64)
    _valider_etiquettes(labels, scores)

    proba = np.asarray(probabilities, dtype=np.float64)
    if proba.ndim != 1 or proba.shape[0] != len(labels):
        raise ContratInvalide(
            f"distribution : {proba.shape} probabilites pour {len(labels)} "
            "etiquettes")
    if not np.all(np.isfinite(proba)):
        raise ContratInvalide("distribution : probabilite non finie")
    if np.any(proba < 0.0):
        raise ContratInvalide("distribution : probabilite negative")
    if abs(float(proba.sum()) - 1.0) > TOLERANCE_SOMME:
        raise ContratInvalide(
            f"distribution : somme {float(proba.sum())} au lieu de 1")

    regrets = scores.max() - scores
    choisi = int(np.argmax(proba))
    return {
        "best_score": float(scores.max()),
        "chosen_index": int(labels[choisi]["index"]),
        "expected_regret": float(proba @ regrets),
        "argmax_regret": float(regrets[choisi]),
        "near_best_mass": float(
            proba[regrets <= SEUIL_PROXIME + TOLERANCE].sum()),
        "catastrophic_mass": float(
            proba[regrets >= SEUIL_CATASTROPHIQUE - TOLERANCE].sum()),
        "cp_regret": _cp_regret(labels, scores, choisi),
    }


def value_metrics(predicted, targets) -> dict:
    """MAE, RMSE, biais et correlation de Pearson, en float64.

    La correlation vaut null en dessous de deux valeurs ou quand une des deux
    series est de variance nulle : elle serait alors un artefact.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)
    if pred.ndim != 1 or targ.ndim != 1 or pred.shape != targ.shape:
        raise ContratInvalide(
            f"value : tailles incompatibles {pred.shape} et {targ.shape}")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(targ))):
        raise ContratInvalide("value : valeur non finie")

    if pred.size == 0:
        return {"value_mae": None, "value_rmse": None,
                "value_bias": None, "value_correlation": None}

    erreur = pred - targ
    correlation = None
    if pred.size >= 2:
        ecart_pred = pred - pred.mean()
        ecart_targ = targ - targ.mean()
        denominateur = math.sqrt(
            float(np.sum(ecart_pred ** 2)) * float(np.sum(ecart_targ ** 2)))
        if denominateur > 0.0:
            correlation = float(np.sum(ecart_pred * ecart_targ) / denominateur)

    return {
        "value_mae": float(np.mean(np.abs(erreur))),
        "value_rmse": float(math.sqrt(float(np.mean(erreur ** 2)))),
        "value_bias": float(np.mean(erreur)),
        "value_correlation": correlation,
    }


def _valider_ligne(row: dict) -> None:
    if not isinstance(row, dict):
        raise ContratInvalide("agregation : une ligne non objet")
    phase = row.get("phase")
    if phase not in PHASES:
        raise ContratInvalide(f"agregation : phase inconnue {phase!r}")
    if row.get("wdl_bucket") not in WDL_BUCKETS:
        raise ContratInvalide(
            f"agregation : wdl_bucket inconnu {row.get('wdl_bucket')!r}")
    if row.get("player_type") not in PLAYER_TYPES:
        raise ContratInvalide(
            f"agregation : player_type inconnu {row.get('player_type')!r}")
    for cle in ("policy_expected_regret", "policy_argmax_regret",
                "near_best_mass", "catastrophic_mass",
                "value_pred", "value_target"):
        _flottant(row, cle, "agregation")
    if row.get("cp_regret") is not None:
        _flottant(row, "cp_regret", "agregation")


def _metriques_slice(rows: list[dict]) -> dict:
    """Metriques d'un sous-ensemble de lignes, plus son effectif.

    Les correlations de value sont recalculees sur les valeurs brutes de la
    tranche, jamais moyennees depuis les sous-groupes.
    """
    vide = {
        "count": 0,
        "policy_expected_regret": None,
        "policy_argmax_regret": None,
        "near_best_mass": None,
        "catastrophic_mass": None,
        "value_mae": None,
        "value_rmse": None,
        "value_bias": None,
        "value_correlation": None,
        "cp_regret_median": None,
        "cp_regret_p90": None,
        "policy_within_20cp": 0.0,
        "policy_within_50cp": 0.0,
        "policy_within_100cp": 0.0,
        "cp_coverage": 0.0,
    }
    if not rows:
        return vide

    for row in rows:
        _valider_ligne(row)

    count = len(rows)
    metriques = {
        "count": count,
        "policy_expected_regret": float(
            np.mean([row["policy_expected_regret"] for row in rows])),
        "policy_argmax_regret": float(
            np.mean([row["policy_argmax_regret"] for row in rows])),
        "near_best_mass": float(
            np.mean([row["near_best_mass"] for row in rows])),
        "catastrophic_mass": float(
            np.mean([row["catastrophic_mass"] for row in rows])),
    }
    metriques.update(value_metrics(
        [row["value_pred"] for row in rows],
        [row["value_target"] for row in rows]))

    regrets = [float(row["cp_regret"]) for row in rows
               if row.get("cp_regret") is not None]
    metriques["cp_coverage"] = len(regrets) / count
    if regrets:
        serie = np.asarray(regrets, dtype=np.float64)
        metriques["cp_regret_median"] = float(
            np.quantile(serie, 0.5, method="linear"))
        metriques["cp_regret_p90"] = float(
            np.quantile(serie, 0.9, method="linear"))
    else:
        metriques["cp_regret_median"] = None
        metriques["cp_regret_p90"] = None
    for seuil in SEUILS_CP:
        metriques[f"policy_within_{seuil}cp"] = (
            sum(1 for regret in regrets if regret <= seuil + TOLERANCE) / count)
    return metriques


def aggregate(rows: list[dict]) -> dict:
    """Agrege des lignes par position en global, phase, categorie WDL et type.

    Chaque ligne porte les cles de `_valider_ligne` : phase, wdl_bucket,
    player_type, les regrets et masses de policy, `value_pred`, `value_target`
    et `cp_regret` (null quand la position sort du diagnostic cp). Chaque
    tranche porte son effectif dans `count` ; une tranche vide rend des
    statistiques null et une couverture nulle.
    """
    resultat = dict(_metriques_slice(rows))
    resultat["phase"] = {
        phase: _metriques_slice(
            [row for row in rows if row.get("phase") == phase])
        for phase in PHASES
    }
    resultat["wdl"] = {
        bucket: _metriques_slice(
            [row for row in rows if row.get("wdl_bucket") == bucket])
        for bucket in WDL_BUCKETS
    }
    resultat["player"] = {
        type_joueur: _metriques_slice(
            [row for row in rows if row.get("player_type") == type_joueur])
        for type_joueur in PLAYER_TYPES
    }
    return resultat
