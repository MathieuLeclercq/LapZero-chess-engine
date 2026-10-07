"""Evaluation recurrente du banc de positions : policy, value puis MCTS.

Le premier passage charge le dataset par lots de 1024 : chaque position est
rejouee dans le C++, verifiee (FEN, identite reseau, coups legaux et
etiquettes), puis le lot part dans une session ONNX GPU unique. Les
probabilites sont calculees par un softmax masque aux seuls coups legaux et
jointes aux labels Stockfish pre-calcules.

Le sous-banc MCTS parcourt 256 identifiants dans l'ordre du manifeste, sur un
seul objet MCTS et un seul evaluateur persistants : racine neuve, table de
transposition froide et compteurs remis a zero a chaque position. Aucun appel
n'est decoupe : chaque position vaut exactement le budget demande.

Aucune adaptation au temps ecoule : une cible de duree depassee est publiee
telle quelle, sans reduire les budgets.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md (section 7) et
docs/superpowers/plans/2026-09-30-position-bench.md (tache 6).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
from evaluation.build_position_bench import lire_jsonl_zst
from evaluation.position_bench_metrics import (
    TAILLE_POLICY,
    EvalConfig,
    EvalReport,
    RawResult,
    score_distribution,
)
from evaluation.position_bench_results import (
    VERSION_METRIQUES,
    agreger_resultat,
    compare_results,
    find_previous,
    load_dataset,
    load_result,
    save_result,
    sha256_fichier,
)
from evaluation.position_bench_sources import position_identity, replay_position

TOLERANCE_VALEUR = 1e-6

RACINE = _runtime.REPO_ROOT
BANC_DEFAUT = RACINE / "data" / "position_bench" / "v1"
RESULTATS_DEFAUT = RACINE / "position_bench_results"


def _preparer_plateau(record: Mapping[str, object], exiger_labels: bool = True):
    """Rejoue et verifie une position avant toute inference.

    Le FEN, l'identite reseau, la liste triee des coups legaux et la
    couverture des etiquettes doivent correspondre exactement : une donnee
    incoherente doit faire echouer le passage, pas produire une mesure
    approximative.
    """
    position_id = str(record["position_id"])
    board = replay_position(record)
    if board.to_fen() != record["fen"]:
        raise ValueError(f"position {position_id} : FEN rejouee differente")
    if position_identity(board) != position_id:
        raise ValueError(f"position {position_id} : identite reseau differente")
    indices = sorted(board.get_legal_move_indices())
    if indices != list(cast(list[int], record["legal_indices"])):
        raise ValueError(
            f"position {position_id} : coups legaux differents de "
            "l'enregistrement")
    if exiger_labels:
        labels = cast(list, record["labels"])
        if [label["index"] for label in labels] != indices:
            raise ValueError(
                f"position {position_id} : etiquettes incompletes ou "
                "desordonnees")
    return board, indices


def _valider_sorties(logits, values, lignes: int) -> None:
    if logits.shape != (lignes, TAILLE_POLICY) or values.shape != (lignes,):
        raise RuntimeError(
            f"evaluateur : sorties de forme {logits.shape} et {values.shape} "
            f"pour {lignes} lignes")
    if not (np.all(np.isfinite(logits)) and np.all(np.isfinite(values))):
        raise RuntimeError("evaluateur : sortie non finie")
    if np.any(values < -1.0 - TOLERANCE_VALEUR) \
            or np.any(values > 1.0 + TOLERANCE_VALEUR):
        raise RuntimeError(
            "evaluateur : value hors de [-1, 1] ; sortie refusee, jamais "
            "ecretee")


def _probabilites_masquees(logits_ligne, indices: list[int]) -> np.ndarray:
    """Softmax limite aux coups legaux, en float64 puis stocke en float32."""
    choisis = np.asarray(logits_ligne[indices], dtype=np.float64)
    choisis -= choisis.max()
    exponentielles = np.exp(choisis)
    return (exponentielles / exponentielles.sum()).astype(np.float32)


def _compteurs_dict(compteurs) -> dict:
    return {
        "completed_simulations": int(compteurs.completed_simulations),
        "nn_calls": int(compteurs.nn_calls),
        "nn_batches": int(compteurs.nn_batches),
        "tt_hits": int(compteurs.tt_hits),
        "tt_misses": int(compteurs.tt_misses),
        "leaf_collisions": int(compteurs.leaf_collisions),
    }


def _passe_policy(records, evaluator, config: EvalConfig, clock,
                  exiger_labels: bool) -> dict:
    """Parcours direct par lots : un seul lot de tenseurs vit a la fois."""
    offsets = [0]
    indices_plats: list[int] = []
    probas_plates: list[float] = []
    valeurs: list[float] = []
    regrets: list[float] = []
    temps_rejeu = 0.0
    temps_policy = 0.0

    for debut in range(0, len(records), config.policy_batch_size):
        lot = records[debut:debut + config.policy_batch_size]
        marque = clock()
        plateaux = []
        legaux_lot = []
        for record in lot:
            plateau, indices = _preparer_plateau(record, exiger_labels)
            plateaux.append(np.asarray(plateau.get_alphazero_tensor(),
                                       dtype=np.float32))
            legaux_lot.append(indices)
        temps_rejeu += clock() - marque

        marque = clock()
        logits, values = evaluator.predict_batch(np.stack(plateaux))
        temps_policy += clock() - marque
        _valider_sorties(logits, values, len(lot))

        for record, ligne, indices in zip(lot, logits, legaux_lot,
                                          strict=True):
            probas = _probabilites_masquees(ligne, indices)
            probas_plates.extend(probas)
            indices_plats.extend(indices)
            offsets.append(len(indices_plats))
            if exiger_labels:
                regrets.append(float(score_distribution(
                    probas, cast(list, record["labels"]))["expected_regret"]))
        valeurs.extend(float(valeur) for valeur in values)

    return {
        "offsets": offsets,
        "indices": indices_plats,
        "probas": probas_plates,
        "valeurs": valeurs,
        "regrets": regrets,
        "replay_s": temps_rejeu,
        "policy_s": temps_policy,
    }


def _passe_recherche(records, search_ids, evaluator, config: EvalConfig,
                     mcts_factory, clock, exiger_labels: bool = True) -> dict:
    """Un seul MCTS pour tout le sous-banc, racine et table froides."""
    reglages = config.search
    par_id = {str(record["position_id"]): record for record in records}

    if not search_ids:
        return {
            "offsets": [0], "indices": [], "probas": [], "compteurs": [],
            "regrets": [], "setup_s": 0.0, "replay_s": 0.0, "mcts_s": 0.0,
            "durees": [],
        }

    marque = clock()
    mcts = mcts_factory(evaluator, reglages.tt_size,
                        reglages.cache_history_depth)
    mcts.set_fixed_batch(reglages.fixed_batch)
    mcts.set_tuning(reglages.virtual_loss, reglages.fpu,
                    reglages.collision_attempts)
    setup_s = clock() - marque

    offsets = [0]
    indices_plats: list[int] = []
    probas_plates: list[float] = []
    regrets: list[float] = []
    compteurs_liste: list[dict] = []
    durees: list[float] = []
    temps_rejeu = 0.0
    temps_mcts = 0.0

    for position_id in search_ids:
        record = par_id.get(str(position_id))
        if record is None:
            raise ValueError(f"recherche : position inconnue {position_id}")
        marque = clock()
        plateau = replay_position(record)
        temps_rejeu += clock() - marque

        mcts.clear_evaluation_cache()
        mcts.reset_counters()
        marque = clock()
        pi = mcts.mcts_search(plateau, reglages.simulations,
                              reglages.c_puct, False,
                              reglages.batch_size, reglages.workers)
        duree = clock() - marque
        temps_mcts += duree
        durees.append(duree)

        compteurs = _compteurs_dict(mcts.get_counters())
        if compteurs["completed_simulations"] != reglages.simulations:
            raise ValueError(
                f"recherche {position_id} : "
                f"{compteurs['completed_simulations']} simulations sur "
                f"{reglages.simulations}")
        compteurs_liste.append(compteurs)

        indices = list(cast(list[int], record["legal_indices"]))
        visites = np.asarray([pi[index] for index in indices],
                             dtype=np.float64)
        total = float(visites.sum())
        if total <= 0.0:
            raise ValueError(f"recherche {position_id} : aucune visite")
        normalisees = (visites / total).astype(np.float32)
        indices_plats.extend(indices)
        probas_plates.extend(normalisees)
        offsets.append(len(indices_plats))
        if exiger_labels:
            regrets.append(float(score_distribution(
                normalisees, cast(list, record["labels"]))["expected_regret"]))

    return {
        "offsets": offsets,
        "indices": indices_plats,
        "probas": probas_plates,
        "compteurs": compteurs_liste,
        "regrets": regrets,
        "setup_s": setup_s,
        "replay_s": temps_rejeu,
        "mcts_s": temps_mcts,
        "durees": durees,
    }


def evaluate_positions(records, search_ids, evaluator, config: EvalConfig, *,
                       mcts_factory,
                       clock=time.perf_counter) -> RawResult:
    """Policy/value sur toutes les positions, puis MCTS sur le sous-banc.

    `records` est dans l'ordre du manifeste et porte les annotations ;
    `search_ids` fixe l'ordre des recherches. `mcts_factory(evaluateur,
    tt_size, cache_history_depth)` est injecte pour les tests.
    """
    debut_total = clock()
    position_ids = [str(record["position_id"]) for record in records]
    policy = _passe_policy(records, evaluator, config, clock, True)
    recherche = _passe_recherche(records, search_ids, evaluator, config,
                                 mcts_factory, clock)

    resultat: dict = {
        "position_ids": np.asarray(position_ids, dtype="<U64"),
        "legal_offsets": np.asarray(policy["offsets"], dtype=np.int64),
        "legal_indices": np.asarray(policy["indices"], dtype=np.int32),
        "policy_probs": np.asarray(policy["probas"], dtype=np.float32),
        "values": np.asarray(policy["valeurs"], dtype=np.float32),
        "policy_regrets": np.asarray(policy["regrets"], dtype=np.float32),
        "search_ids": np.asarray(search_ids, dtype="<U64"),
        "search_offsets": np.asarray(recherche["offsets"], dtype=np.int64),
        "search_indices": np.asarray(recherche["indices"], dtype=np.int32),
        "search_probs": np.asarray(recherche["probas"], dtype=np.float32),
        "search_regrets": np.asarray(recherche["regrets"], dtype=np.float32),
        "search_counters": recherche["compteurs"],
        "timings": {
            "replay_s": policy["replay_s"] + recherche["replay_s"],
            "policy_s": policy["policy_s"],
            "mcts_setup_s": recherche["setup_s"],
            "mcts_s": recherche["mcts_s"],
            "total_s": clock() - debut_total,
            "search_position_s": recherche["durees"],
        },
    }
    return cast(RawResult, resultat)


def profile_positions(records, evaluator, config: EvalConfig, *,
                      mcts_factory, search_count: int | None = None,
                      clock=time.perf_counter) -> dict:
    """Mesure le parcours direct et la recherche sur des positions brutes.

    Mode `profile` : aucune annotation requise, aucun score, aucune version du
    banc. Sert a estimer le cout avant la longue annotation reelle.
    """
    debut = clock()
    ids = [str(record["position_id"]) for record in records]
    if search_count is None:
        retenus = ids[:32] if len(ids) > 32 else ids
    else:
        retenus = ids[:search_count]
    policy = _passe_policy(records, evaluator, config, clock, False)
    recherche = _passe_recherche(records, retenus, evaluator, config,
                                 mcts_factory, clock, exiger_labels=False)
    return {
        "mode": "profile",
        "count": len(records),
        "searches": len(retenus),
        "timings": {
            "replay_s": policy["replay_s"] + recherche["replay_s"],
            "policy_s": policy["policy_s"],
            "mcts_setup_s": recherche["setup_s"],
            "mcts_s": recherche["mcts_s"],
            "total_s": clock() - debut,
            "search_position_s": recherche["durees"],
        },
    }


# ============================================================
#                EVALUATION D'UN CHECKPOINT
# ============================================================

@lru_cache(maxsize=1)
def _hash_module() -> str:
    import chess_engine

    return sha256_fichier(Path(str(chess_engine.__file__)))


def protocol_id(config: EvalConfig) -> str:
    """Hash canonique du protocole de mesure, independant du modele."""
    import onnxruntime

    charge = {
        "metrics_version": VERSION_METRIQUES,
        "search": asdict(config.search),
        "policy_batch_size": config.policy_batch_size,
        "target_s": config.target_s,
        "use_gpu": config.use_gpu,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "onnxruntime": onnxruntime.__version__,
        "module_sha256": _hash_module(),
    }
    return hashlib.sha256(json.dumps(
        charge, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluate_checkpoint(onnx_path, bench_path, output_dir, config: EvalConfig,
                        *, iteration=None, global_step=None, previous=None,
                        extra_timings=None) -> EvalReport:
    """Evalue un ONNX sur le banc, sauvegarde le resultat et compare.

    La duree publiee couvre le chargement, l'inference, la recherche,
    l'agregation et l'ecriture locale. Un depassement de la cible est un
    passage valide : rien n'est interrompu ni invalide pour cette raison.
    """
    import chess_engine

    chemin_modele = Path(onnx_path)
    if not chemin_modele.is_file() or chemin_modele.suffix.lower() != ".onnx":
        raise ValueError(
            f"modele ONNX introuvable ou invalide : {chemin_modele}")

    manifeste, records = load_dataset(bench_path)
    protocole = protocol_id(config)
    hash_modele = sha256_fichier(chemin_modele)
    if previous is None and iteration is not None:
        previous = find_previous(output_dir, manifeste["dataset_sha256"],
                                 protocole, iteration)
    evaluateur = chess_engine.ONNXEvaluator(str(chemin_modele), config.use_gpu)
    fabrique = lambda evaluateur, taille, profondeur: chess_engine.MCTS(
        evaluateur, taille, profondeur)

    debut = time.perf_counter()
    raw = evaluate_positions(records, list(manifeste["search_ids"]),
                             evaluateur, config, mcts_factory=fabrique)

    marque = time.perf_counter()
    metriques = agreger_resultat(manifeste, records, raw)
    aggregate_s = time.perf_counter() - marque

    comparison = None
    if previous is not None:
        precedent, rapport_precedent = load_result(previous)
        comparison = compare_results(
            raw, precedent,
            {"dataset_sha256": manifeste["dataset_sha256"],
             "protocol_id": protocole},
            {"dataset_sha256": rapport_precedent["dataset_sha256"],
             "protocol_id": rapport_precedent["protocol_id"]})
    metriques["comparison"] = comparison

    timings = dict(raw["timings"])
    timings["aggregate_s"] = aggregate_s
    if extra_timings:
        timings.update(extra_timings)

    chemin_resultat = Path(output_dir) / f"{chemin_modele.stem}.npz"
    rapport: EvalReport = {
        "status": "ok",
        "dataset_version": manifeste["dataset_version"],
        "dataset_sha256": manifeste["dataset_sha256"],
        "model_sha256": hash_modele,
        "protocol_id": protocole,
        "completed_positions": len(records),
        "completed_search_positions": len(raw["search_ids"]),
        "duration_s": time.perf_counter() - debut,
        "timings": timings,
        "metrics": metriques,
        "comparison": comparison,
        "result_path": str(chemin_resultat),
        "error": None,
        "target_s": config.target_s,
        "over_target": False,
        "metrics_version": VERSION_METRIQUES,
        "iteration": iteration,
        "global_step": global_step,
    }

    marque = time.perf_counter()
    save_result(raw, rapport, output_dir, chemin_modele.stem)
    timings["write_s"] = time.perf_counter() - marque

    rapport["duration_s"] = time.perf_counter() - debut
    rapport["over_target"] = rapport["duration_s"] > config.target_s
    save_result(raw, rapport, output_dir, chemin_modele.stem)
    return rapport


def _resoudre_modele(chemin, dossier_cache):
    """Chemin ONNX et duree d'export ; un .pt passe par le cache par hash."""
    import evaluation.puzzle_bench as puzzle_bench

    source = Path(chemin)
    if not source.is_file():
        raise ValueError(f"modele introuvable : {source}")
    if source.suffix.lower() == ".onnx":
        return source, 0.0
    if source.suffix.lower() != ".pt":
        raise ValueError(
            f"modele non reconnu (attendu .onnx ou .pt) : {source}")
    digest = sha256_fichier(source)
    dossier = Path(dossier_cache) / digest[:16]
    debut = time.perf_counter()
    onnx, _ = puzzle_bench.resoudre_modele(source, dossier)
    return Path(onnx), time.perf_counter() - debut


# ============================================================
#                            CLI
# ============================================================

def _parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="position_bench")
    sous = parser.add_subparsers(dest="commande", required=True)

    valider = sous.add_parser("validate", help="verifie un banc publie")
    valider.add_argument("--bench", default=str(BANC_DEFAUT))

    evaluer = sous.add_parser("evaluate", help="evalue un checkpoint")
    evaluer.add_argument("--model", required=True)
    evaluer.add_argument("--bench", default=str(BANC_DEFAUT))
    evaluer.add_argument("--output-dir", default=str(RESULTATS_DEFAUT))
    evaluer.add_argument("--iteration", type=int)
    evaluer.add_argument("--global-step", type=int)
    evaluer.add_argument("--cpu", action="store_true")

    comparer = sous.add_parser("compare", help="compare deux resultats")
    comparer.add_argument("--current", required=True)
    comparer.add_argument("--previous", required=True)

    reagreger = sous.add_parser("reaggregate",
                                help="recalcule les metriques d'un resultat")
    reagreger.add_argument("--result", required=True)
    reagreger.add_argument("--bench", default=str(BANC_DEFAUT))
    reagreger.add_argument("--output", required=True)

    profil = sous.add_parser("profile", help="chronometre sans score")
    profil.add_argument("--candidates", required=True)
    profil.add_argument("--model", required=True)
    profil.add_argument("--search-count", type=int, default=32)
    profil.add_argument("--output-dir", default=str(RESULTATS_DEFAUT))
    return parser


def main(argv=None) -> int:
    args = _parseur().parse_args(argv)
    try:
        if args.commande == "validate":
            _cmd_validate(args)
        elif args.commande == "evaluate":
            _cmd_evaluate(args)
        elif args.commande == "compare":
            _cmd_compare(args)
        elif args.commande == "reaggregate":
            _cmd_reaggregate(args)
        else:
            _cmd_profile(args)
    except (ValueError, RuntimeError, OSError) as erreur:
        print(f"erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args) -> None:
    manifeste, records = load_dataset(args.bench)
    print(json.dumps({
        "dataset_version": manifeste["dataset_version"],
        "positions": len(records),
        "search": len(manifeste["search_ids"]),
        "dataset_sha256": manifeste["dataset_sha256"],
    }, sort_keys=True))


def _cmd_evaluate(args) -> None:
    onnx, export_s = _resoudre_modele(
        args.model, Path(args.output_dir) / "cache_onnx")
    config = EvalConfig(use_gpu=not args.cpu)
    rapport = evaluate_checkpoint(
        onnx, args.bench, args.output_dir, config,
        iteration=args.iteration, global_step=args.global_step,
        extra_timings={"model_export_s": export_s} if export_s else None)
    print(json.dumps({
        "status": rapport["status"],
        "result_path": rapport["result_path"],
        "duration_s": rapport["duration_s"],
        "over_target": rapport.get("over_target"),
    }, sort_keys=True))


def _cmd_compare(args) -> None:
    courant, rapport_courant = load_result(args.current)
    precedent, rapport_precedent = load_result(args.previous)
    comparaison = compare_results(
        courant, precedent,
        {"dataset_sha256": rapport_courant["dataset_sha256"],
         "protocol_id": rapport_courant["protocol_id"]},
        {"dataset_sha256": rapport_precedent["dataset_sha256"],
         "protocol_id": rapport_precedent["protocol_id"]})
    print(json.dumps(comparaison, sort_keys=True))


def _cmd_reaggregate(args) -> None:
    raw, rapport = load_result(args.result)
    manifeste, records = load_dataset(args.bench)
    if manifeste["dataset_sha256"] != rapport["dataset_sha256"]:
        raise ValueError("le resultat ne porte pas sur ce dataset")
    metriques = agreger_resultat(manifeste, records, raw)
    charge = {"dataset_version": manifeste["dataset_version"],
              "protocol_id": rapport["protocol_id"],
              "metrics": metriques}
    chemin = Path(args.output)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(charge, sort_keys=True, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(chemin)}, sort_keys=True))


def _cmd_profile(args) -> None:
    import chess_engine

    records = lire_jsonl_zst(args.candidates)
    onnx, export_s = _resoudre_modele(
        args.model, Path(args.output_dir) / "cache_onnx")
    evaluateur = chess_engine.ONNXEvaluator(str(onnx), True)
    fabrique = lambda evaluateur, taille, profondeur: chess_engine.MCTS(
        evaluateur, taille, profondeur)
    rapport = profile_positions(
        records, evaluateur, EvalConfig(),
        mcts_factory=fabrique, search_count=args.search_count)
    if export_s:
        rapport["timings"]["model_export_s"] = export_s
    dossier = Path(args.output_dir)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "profile.json"
    chemin.write_text(json.dumps(rapport, sort_keys=True, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(chemin),
                      "searches": rapport["searches"]}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
