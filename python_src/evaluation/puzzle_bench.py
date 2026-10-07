"""Banc de puzzles Lichess : orchestration.

Mesure un modele sur data/puzzles_bench.txt et ecrit un CSV par puzzle plus un
rapport agrege. Le scoring vit dans bench_metrics, testable sans modele.

Voir docs/etudes/evaluation-puzzles/protocole/2026-08-14-puzzle-bench-design.md
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import hashlib
import json
import os
import sys
import time
from pathlib import Path
from core.data_paths import ONNX_DIR

RACINE_PYTHON = _runtime.PYTHON_ROOT
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))
os.add_dll_directory(str(RACINE_PYTHON))

import chess_engine

# Un MCTS neuf est cree a chaque recherche. Au defaut de 2 097 143 entrees a
# 1040 octets, chaque instance reserverait 2,03 Gio, soit 32,5 Gio a 16
# travailleurs pour 31,4 Gio de RAM. Une recherche ne stocke au plus que
# `simulations` positions distinctes, donc 8192 entrees suffisent largement.
TAILLE_TT = 8192

# Granularite de la fenetre de temps : une tranche de 8 simulations, meme
# valeur que le batch vise. Le dernier appel complet depasse la fenetre, c'est
# ce depassement qui est enregistre.
TRANCHE_SIMULATIONS = 8


def exporter_onnx(chemin_pt: Path, sortie: Path) -> dict:
    """Exporte un checkpoint .pt en ONNX a axes dynamiques.

    torch n'est importe qu'ici, donc jamais dans un processus travailleur ou il
    couterait 475 Mio. L'architecture est deduite du checkpoint plutot que
    codee en dur, sans quoi un modele de taille differente casserait en
    silence.
    """
    import torch

    from core.model import ChessNet

    checkpoint = torch.load(chemin_pt, map_location="cpu", weights_only=True)
    etat = checkpoint["model_state_dict"]

    num_filters = etat["conv_input.weight"].shape[0]
    num_res_blocks = 1 + max(
        int(cle.split(".")[1]) for cle in etat if cle.startswith("res_blocks."))

    modele = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    modele.load_state_dict(etat)
    modele.eval()

    sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        modele,
        torch.randn(1, 119, 8, 8),
        str(sortie),
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch_size"},
                      "policy": {0: "batch_size"},
                      "value": {0: "batch_size"}},
        # Sous torch 2.13 l'exporteur par defaut reclame onnxscript, absent.
        dynamo=False,
    )
    return {
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "num_res_blocks": num_res_blocks,
        "num_filters": num_filters,
    }


def resoudre_modele(chemin: Path, dossier_onnx: Path) -> tuple[Path, dict]:
    """Accepte un .onnx tel quel, ou exporte un .pt s'il le faut."""
    chemin = Path(chemin)
    if chemin.suffix == ".onnx":
        return chemin, {"iteration": None, "global_step": None,
                        "num_res_blocks": None, "num_filters": None}

    sortie = Path(dossier_onnx) / f"{chemin.stem}.onnx"
    if not sortie.exists():
        return sortie, exporter_onnx(chemin, sortie)

    # Un export deja present est reutilise, mais l'architecture est relue pour
    # le contexte du rapport.
    import torch

    checkpoint = torch.load(chemin, map_location="cpu", weights_only=True)
    etat = checkpoint["model_state_dict"]
    return sortie, {
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "num_res_blocks": 1 + max(int(c.split(".")[1]) for c in etat
                                  if c.startswith("res_blocks.")),
        "num_filters": etat["conv_input.weight"].shape[0],
    }


def faire_policy_fn(session):
    """Renvoie policy_fn(board) -> (probabilites sur les index legaux, value).

    Softmax masque sur les coups legaux. Le C++ fait un softmax sur les 4672
    sorties (onnx_evaluator.cpp:74-89) puis renormalise sur les coups legaux
    (mcts.cpp:184-198) : renormaliser un softmax global sur un sous-ensemble
    est identique a un softmax sur ce seul sous-ensemble, et un test verrouille
    l'accord.
    """
    import math

    import numpy as np

    def policy_fn(board):
        tenseur = np.asarray(board.get_alphazero_tensor(),
                             dtype=np.float32).reshape(1, 119, 8, 8)
        logits, value = session.run(None, {"input": tenseur})
        logits = logits[0]

        indices = board.get_legal_move_indices()
        maxi = max(float(logits[i]) for i in indices)
        exps = {i: math.exp(float(logits[i]) - maxi) for i in indices}
        somme = sum(exps.values())
        return ({i: e / somme for i, e in exps.items()},
                float(np.asarray(value).reshape(-1)[0]))

    return policy_fn


def faire_search_fn(evaluateur, simulations: int, c_puct: float,
                    batch_size: int = 0,
                    cache_history_depth: int = 0,
                    worker_count: int = 1,
                    search_seconds: float | None = None,
                    horloge=time.perf_counter,
                    virtual_loss: int = 1,
                    fpu_reduction: float = 0.30,
                    collision_attempts: int = 4):
    """Renvoie search_fn(board) -> distribution de visites sur 4672.

    A simulations fixes, un MCTS neuf est cree pour chaque puzzle afin que les
    mesures restent independantes entre les lignes du banc.

    En mode temps (search_seconds), l'objet MCTS est conserve entre les puzzles
    de ce processus : avant chaque fenetre, l'arbre d'analyse et la TT sont
    remis a froid hors chronometrage, mais les threads du pool restent en
    place. La recherche avance par tranches de TRANCHE_SIMULATIONS, puis
    s'arrete apres la tranche qui franchit la fenetre. Le bilan renvoye
    contient les simulations terminees et le depassement du dernier appel
    complet.

    Le lot fixe est toujours actif : c'est le reglage de production, il evite
    la re-planification d'ONNX Runtime a chaque changement de forme.
    """
    def regler(mcts):
        mcts.set_fixed_batch(True)
        if (virtual_loss, fpu_reduction, collision_attempts) != (1, 0.30, 4):
            mcts.set_tuning(virtual_loss, fpu_reduction, collision_attempts)

    if search_seconds is None:
        def search_fn(board):
            mcts = chess_engine.MCTS(
                evaluateur, TAILLE_TT, cache_history_depth)
            regler(mcts)
            return mcts.mcts_search(board, simulations, c_puct, False,
                                    batch_size, worker_count)
        return search_fn

    etat = {"mcts": None}

    def search_fn(board):
        from evaluation.bench_metrics import TAILLE_POLICY

        mcts = etat["mcts"]
        if mcts is None:
            mcts = chess_engine.MCTS(
                evaluateur, TAILLE_TT, cache_history_depth)
            regler(mcts)
            etat["mcts"] = mcts
        mcts.reset_analysis()
        mcts.clear_evaluation_cache()
        mcts.reset_counters()

        debut = horloge()
        while True:
            mcts.step_analysis(board, TRANCHE_SIMULATIONS, c_puct,
                               batch_size, worker_count)
            ecoule = horloge() - debut
            if ecoule >= search_seconds:
                break

        pi = [0.0] * TAILLE_POLICY
        for stats in mcts.get_analysis_results():
            pi[stats.move_idx] = float(stats.visits)
        total = sum(pi)
        if total > 0.0:
            pi = [visite / total for visite in pi]

        compteurs = mcts.get_counters()
        return pi, {
            "simulations": int(compteurs.completed_simulations),
            "depassement_s": max(0.0, ecoule - search_seconds),
        }

    return search_fn


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def chemin_sidecar(chemin_csv: Path) -> Path:
    """Le contexte complet d'un CSV voyage a cote de lui, pas dedans.

    Deux campagnes ne sont comparables que si modele, banc, budget et
    politique de TT coincident : le sidecar porte ces hashes et ces reglages,
    que le multicore_comparison refuse d'interpoler.
    """
    return chemin_csv.with_name(chemin_csv.stem + ".meta.json")


def ecrire_sidecar(contexte: dict, chemin_csv: Path) -> None:
    chemin_sidecar(chemin_csv).write_text(
        json.dumps(contexte, indent=2, sort_keys=True), encoding="utf-8")


CHAMPS_CSV = (
    "ligne", "rating", "themes", "plies_historique", "nb_coups_legaux",
    "coup_reseau", "reussi_reseau", "p_correct_reseau", "rang_correct_reseau",
    "value_reseau", "coup_recherche", "reussi_recherche",
    "part_visites_correct", "duree_s", "simulations_recherche",
    "depassement_s", "erreur",
)

# Etat par processus travailleur : la session onnxruntime et l'evaluateur ne
# sont crees qu'une fois par travailleur, pas a chaque puzzle.
_ETAT: dict = {}


def initialiser_travailleur(onnx: str, simulations, c_puct: float,
                            sans_historique: bool, batch_size: int,
                            cache_history_depth: int,
                            search_workers: int = 1,
                            search_seconds: float | None = None,
                            virtual_loss: int = 1,
                            fpu_reduction: float = 0.30,
                            collision_attempts: int = 4,
                            comparer_historique: bool = False) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx, options,
                                   providers=["CPUExecutionProvider"])

    _ETAT["policy_fn"] = faire_policy_fn(session)
    _ETAT["search_fn"] = faire_search_fn(
        chess_engine.ONNXEvaluator(onnx, False), simulations, c_puct,
        batch_size, cache_history_depth, search_workers, search_seconds,
        virtual_loss=virtual_loss, fpu_reduction=fpu_reduction,
        collision_attempts=collision_attempts)
    _ETAT["sans_historique"] = sans_historique
    _ETAT["comparer_historique"] = comparer_historique


def initialiser_gpu(onnx: str, simulations, c_puct: float,
                    sans_historique: bool, batch_size: int,
                    cache_history_depth: int, search_workers: int,
                    search_seconds: float | None,
                    virtual_loss: int = 1,
                    fpu_reduction: float = 0.30,
                    collision_attempts: int = 4,
                    comparer_historique: bool = False) -> None:
    """Un seul processus GPU : la policy brute garde sa session CPU, dont le
    cout est exclu du temps de recherche."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx, options,
                                   providers=["CPUExecutionProvider"])

    _ETAT["policy_fn"] = faire_policy_fn(session)
    _ETAT["search_fn"] = faire_search_fn(
        chess_engine.ONNXEvaluator(onnx, True), simulations, c_puct,
        batch_size, cache_history_depth, search_workers, search_seconds,
        virtual_loss=virtual_loss, fpu_reduction=fpu_reduction,
        collision_attempts=collision_attempts)
    _ETAT["sans_historique"] = sans_historique
    _ETAT["comparer_historique"] = comparer_historique


def traiter_lot(lot: list) -> list:
    """lot : liste de (index, ligne brute).

    Renvoie des PuzzleMeasure, ou des couples (avec, sans) en mode comparaison.
    """
    from evaluation.bench_metrics import measure_puzzle, parse_bench_line

    resultats = []
    for index, ligne in lot:
        puzzle = parse_bench_line(index, ligne)
        avec = measure_puzzle(
            puzzle, _ETAT["policy_fn"], _ETAT["search_fn"],
            sans_historique=_ETAT["sans_historique"])
        if _ETAT.get("comparer_historique"):
            sans = measure_puzzle(
                puzzle, _ETAT["policy_fn"], _ETAT["search_fn"],
                sans_historique=True)
            resultats.append((avec, sans))
        else:
            resultats.append(avec)
    return resultats


def ecrire_csv(mesures: list, chemin: Path) -> None:
    """Une ligne par puzzle, dans l'ordre du fichier de banc.

    Le pool rend les lots dans l'ordre d'achevement, donc le tri est necessaire
    pour que l'index de ligne reste un identifiant utilisable.
    """
    import csv
    import dataclasses

    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CHAMPS_CSV))
        writer.writeheader()
        for mesure in sorted(mesures, key=lambda m: m.ligne):
            writer.writerow(dataclasses.asdict(mesure))


def sous_echantillon(lignes: list, combien: int) -> list:
    """Retient `combien` lignes a pas regulier sur tout le fichier.

    Prendre les premieres lignes donnerait un echantillon biaise : le pipeline
    ecrit le banc tranche de rating par tranche de rating, donc les 200
    premieres lignes appartiennent toutes a la meme tranche. Le pas regulier
    reste deterministe, ce qui compte puisque le fichier n'a pas de PuzzleId et
    que l'index de ligne fait office d'identifiant.
    """
    if combien <= 0 or combien >= len(lignes):
        return lignes

    pas = len(lignes) / combien
    return [lignes[min(len(lignes) - 1, int(i * pas))] for i in range(combien)]


def _lots(lignes: list, taille: int) -> list:
    return [lignes[i:i + taille] for i in range(0, len(lignes), taille)]


def main() -> int:
    import argparse
    import multiprocessing as mp

    from evaluation.bench_metrics import (aggregate, comparer_historique,
                               format_comparaison, format_report)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="checkpoint .pt ou modele .onnx")
    parser.add_argument("--banc", type=Path,
                        default=Path("../data/puzzles_bench.txt"))
    parser.add_argument("--dossier-onnx", type=Path,
                        default=ONNX_DIR)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-rapport", type=Path, default=None)
    # 700 simulations : le banc sert a deux comparaisons relatives, la
    # recherche contre la policy brute et une iteration contre la precedente.
    # Les deux valent a budget fixe, quel qu'il soit.
    parser.add_argument("--simulations", type=int, default=None,
                        help="budget fixe ; defaut 700 si --search-seconds "
                             "n'est pas fourni")
    parser.add_argument("--search-seconds", type=float, default=None,
                        help="budget de temps par puzzle, exclusif d'un "
                             "--simulations explicite")
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="taille du lot MCTS, 0 conserve la recherche sequentielle")
    parser.add_argument("--cache-history-depth", type=int, default=0,
                        help="politique TT : -1 pour legacy, 0 a 7 pour h0 a h7")
    parser.add_argument("--travailleurs", type=int, default=16,
                        help="processus Python independants ; en mode GPU, "
                             "seul 1 est permis")
    parser.add_argument("--search-workers", type=int, default=1,
                        help="workers CPU internes au C++ par recherche, sans "
                             "creer de processus Python supplementaires")
    parser.add_argument("--virtual-loss", type=int, default=1,
                        help="unites de n_in_flight par descente")
    parser.add_argument("--fpu", type=float, default=0.30,
                        help="coefficient du terme FPU")
    parser.add_argument("--collision-attempts", type=int, default=4,
                        help="facteur du budget de tentatives de collecte")
    parser.add_argument("--gpu", action="store_true",
                        help="recherche sur le GPU, un seul processus")
    # 2500 sur les 5000 du fichier : la demi-largeur de Wilson globale passe
    # de 1,2 a 1,6 point seulement, et le passage tient en deux fois moins de
    # temps. Le pas etant regulier et le fichier ecrit tranche par tranche, on
    # garde 625 puzzles par tranche de rating.
    parser.add_argument("--limite", type=int, default=2500,
                        help="sous-echantillon de N puzzles, a pas regulier sur "
                             "tout le fichier ; 0 pour tout prendre. Les N "
                             "premieres lignes tomberaient toutes dans la meme "
                             "tranche de rating, le banc etant ecrit tranche "
                             "par tranche")
    parser.add_argument("--sans-historique", action="store_true",
                        help="presente les puzzles avec l'historique vide")
    parser.add_argument("--comparer-historique", action="store_true",
                        help="mesure chaque puzzle avec et sans historique, "
                             "puis rapporte l'ecart apparie entre les deux "
                             "passes")
    args = parser.parse_args()

    if args.batch_size < 0:
        parser.error("--batch-size doit etre positif ou nul")
    if args.cache_history_depth < -1 or args.cache_history_depth > 7:
        parser.error("--cache-history-depth doit etre compris entre -1 et 7")
    if args.search_workers < 1:
        parser.error("--search-workers doit etre au moins 1")
    if args.search_seconds is not None and args.search_seconds <= 0:
        parser.error("--search-seconds doit etre positif")
    if args.search_seconds is not None and args.simulations is not None:
        parser.error("--search-seconds est exclusif d'un --simulations "
                     "explicite : choisir un seul budget")
    if args.gpu and args.travailleurs != 1:
        parser.error("le mode GPU exige --travailleurs 1 : les workers de "
                     "recherche sont internes au C++ et ne doivent pas "
                     "multiplier les sessions GPU")
    if args.search_workers > 1 and args.batch_size == 0:
        parser.error("la recherche multicoeur exige une taille de batch "
                     "positive")
    if args.virtual_loss < 1:
        parser.error("--virtual-loss doit etre au moins 1")
    if args.fpu < 0.0:
        parser.error("--fpu doit etre positif ou nul")
    if args.collision_attempts < 1:
        parser.error("--collision-attempts doit etre au moins 1")
    if args.sans_historique and args.comparer_historique:
        parser.error("--sans-historique et --comparer-historique sont "
                     "exclusifs : la comparaison mesure elle-meme les deux "
                     "passes")

    simulations = args.simulations
    if simulations is None and args.search_seconds is None:
        simulations = 700
    budget_label = (f"{simulations} simulations"
                    if args.search_seconds is None
                    else f"{args.search_seconds:g} s par puzzle")

    if not args.banc.exists():
        print(f"fichier de banc introuvable : {args.banc}", file=sys.stderr)
        return 2

    onnx, meta = resoudre_modele(args.model, args.dossier_onnx)
    if not Path(onnx).exists():
        print(f"modele ONNX introuvable : {onnx}", file=sys.stderr)
        return 2

    with open(args.banc, encoding="utf-8") as f:
        lignes = list(enumerate(f))
    total_fichier = len(lignes)
    if args.limite:
        lignes = sous_echantillon(lignes, args.limite)

    # Posee dans le parent pour etre heritee : sous Windows le pool utilise
    # spawn et reimporte le module avant d'executer l'initialiseur, donc la
    # poser dans l'initialiseur serait trop tard.
    os.environ["OMP_NUM_THREADS"] = "1"

    suffixe = (" (comparaison avec / sans historique)"
               if args.comparer_historique else
               (" (sans historique)" if args.sans_historique else ""))
    politique = ("legacy" if args.cache_history_depth == -1 else
                 f"h{args.cache_history_depth}")
    print(f"{len(lignes)} puzzles sur {total_fichier} du fichier, "
          f"{budget_label}, premier coup seul, "
          f"batch {args.batch_size}, TT {politique}, "
          f"{args.travailleurs} travailleurs, "
          f"{args.search_workers} workers de recherche, "
          f"{'GPU' if args.gpu else 'CPU'}{suffixe}")

    debut = time.perf_counter()
    lots = _lots(lignes, 16)
    mesures: list = []
    mesures_sans: list = []

    def collecter(resultat):
        """Range un lot selon le mode : mesures seules, ou couples apparies."""
        if args.comparer_historique:
            for avec, sans in resultat:
                mesures.append(avec)
                mesures_sans.append(sans)
        else:
            mesures.extend(resultat)

    if args.gpu:
        # Les DLL CUDA de torch rendent celles du provider ONNX visibles au
        # processus avant la construction de l'evaluateur C++.
        import torch  # noqa: F401

        initialiser_gpu(str(onnx), simulations, args.c_puct,
                        args.sans_historique, args.batch_size,
                        args.cache_history_depth, args.search_workers,
                        args.search_seconds,
                        virtual_loss=args.virtual_loss,
                        fpu_reduction=args.fpu,
                        collision_attempts=args.collision_attempts,
                        comparer_historique=args.comparer_historique)
        for i, lot in enumerate(lots, 1):
            collecter(traiter_lot(lot))
            if i % 10 == 0 or i == len(lots):
                ecoule = time.perf_counter() - debut
                print(f"  {len(mesures)}/{len(lignes)} puzzles, "
                      f"{ecoule / 60.0:.1f} min, "
                      f"{len(mesures) / max(1e-9, ecoule):.1f} puzzles/s",
                      flush=True)
    else:
        with mp.Pool(args.travailleurs, initializer=initialiser_travailleur,
                     initargs=(str(onnx), simulations, args.c_puct,
                               args.sans_historique, args.batch_size,
                               args.cache_history_depth, args.search_workers,
                               args.search_seconds, args.virtual_loss,
                               args.fpu, args.collision_attempts,
                               args.comparer_historique)) as pool:
            for i, resultat in enumerate(
                    pool.imap_unordered(traiter_lot, lots), 1):
                collecter(resultat)
                if i % 10 == 0 or i == len(lots):
                    ecoule = time.perf_counter() - debut
                    print(f"  {len(mesures)}/{len(lignes)} puzzles, "
                          f"{ecoule / 60.0:.1f} min, "
                          f"{len(mesures) / max(1e-9, ecoule):.1f} puzzles/s",
                          flush=True)
    duree = time.perf_counter() - debut

    stats = aggregate(mesures)
    stats_sans = aggregate(mesures_sans) if args.comparer_historique else None
    contexte = {
        "modele": Path(onnx).name,
        "iteration": meta.get("iteration"),
        "global_step": meta.get("global_step"),
        "simulations": simulations,
        "search_seconds": args.search_seconds,
        "budget_label": budget_label,
        "c_puct": args.c_puct,
        "batch_size": args.batch_size,
        "cache_history_depth": args.cache_history_depth,
        "fichier_banc": str(args.banc),
        "sans_historique": args.sans_historique,
        "comparer_historique": args.comparer_historique,
        "duree_totale_s": duree,
        "travailleurs": args.travailleurs,
        "search_workers": args.search_workers,
        "fixed_batch": True,
        "virtual_loss": args.virtual_loss,
        "fpu_reduction": args.fpu,
        "collision_attempts": args.collision_attempts,
        "accelerateur": "GPU" if args.gpu else "CPU",
        "modele_sha256": sha256_fichier(Path(onnx)),
        "banc_sha256": sha256_fichier(args.banc),
        "critere": "premier_coup_recherche",
    }

    suffixe_workers = f"_w{args.search_workers}" if args.search_workers > 1 else ""
    out_csv = args.out_csv or Path(
        f"../data/bench_results/{Path(onnx).stem}_batch{args.batch_size}"
        f"_{politique}{suffixe_workers}.csv")
    out_rapport = args.out_rapport or Path(
        f"../docs/superpowers/specs/{time.strftime('%Y-%m-%d')}"
        f"-puzzle-bench-batch{args.batch_size}-{politique}"
        f"{suffixe_workers}-resultats.md")

    ecrire_csv(mesures, out_csv)
    ecrire_sidecar(contexte, out_csv)
    out_rapport.parent.mkdir(parents=True, exist_ok=True)

    if args.comparer_historique:
        contexte_sans = dict(contexte)
        contexte_sans["sans_historique"] = True
        comparaison = comparer_historique(mesures, mesures_sans)
        out_csv_sans = out_csv.with_name(
            out_csv.stem + "_sans" + out_csv.suffix)
        ecrire_csv(mesures_sans, out_csv_sans)
        ecrire_sidecar(contexte_sans, out_csv_sans)
        out_rapport.write_text(
            format_report(stats, contexte)
            + format_comparaison(comparaison, contexte)
            + format_report(stats_sans, contexte_sans),
            encoding="utf-8")
        print(f"\nCSV avec : {out_csv}")
        print(f"CSV sans : {out_csv_sans}")
    else:
        out_rapport.write_text(
            format_report(stats, contexte), encoding="utf-8")
        print(f"\nCSV     : {out_csv}")

    print(f"Rapport : {out_rapport}")
    print(f"Duree   : {duree / 60.0:.1f} min")
    erreurs = stats.erreurs
    if stats_sans is not None:
        erreurs = erreurs or stats_sans.erreurs
    if erreurs:
        # Un CSV contenant des lignes en erreur ne doit jamais se faire passer
        # pour une campagne reussie : la comparaison les refusera, mais le
        # statut d'echec doit deja etre visible ici.
        print(f"Erreurs de donnees : {erreurs}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    sys.exit(main())
