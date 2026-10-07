"""Banc de diagnostic du self-play, branche sur le vrai chemin de production.

Le pilote reprend le perimetre de `train_self_play.generate_games` : creation de
l'evaluateur ONNX, appel C++ complet, conversion des exemples, nettoyage. Il
ajoute le rapport de phases du gestionnaire (mode 1) et, en mode 2, le travail
mural par worker de la collecte.

Trois debits distincts sont rapportes : coups nouveaux par seconde, exemples
conserves par seconde et evaluations reseau par seconde. Le dernier est un
compteur de diagnostic, pas un objectif a maximiser isolement.

Outils de diagnostic, hors validation. Usage :

    python python_src/dev_tools/selfplay_diagnostics.py --games 128 \
        --concurrent 128 --passes 1

Les timings bruts vont dans out/selfplay_diagnostics/ (ignore par git).
"""

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
PYTHON_SRC = RACINE / "python_src"
sys.path.insert(0, str(PYTHON_SRC))

from training.selfplay_diag import (
    collecter_configuration,
    formater_bilan,
    formater_diagnostic,
    resumer_passage,
)

MODELE_DEFAUT = str(
    RACINE / "data" / "checkpoints" / "onnx"
    / "2026_04_30_09h53_iter436_unsupervised.onnx")
PUZZLES_DEFAUT = str(RACINE / "training_data" / "puzzles_train.txt")


def mesurer_passage(args, mode, puzzles):
    """Une generation complete, les zones Python mesurees separement.

    Un evaluateur neuf par passage : sa creation fait partie du perimetre de la
    metrique de production et doit rester visible comme zone separee.
    """
    import chess_engine
    import torch
    from core.lib import convert_game_results

    zones = {"creation_evaluateur": 0.0, "appel_cpp": 0.0,
             "conversion": 0.0, "nettoyage": 0.0}
    pic_vram_mio = None

    debut = time.perf_counter()
    evaluateur = chess_engine.ONNXEvaluator(args.model, True)
    zones["creation_evaluateur"] = time.perf_counter() - debut

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    debut = time.perf_counter()
    parties, stats, timing = (
        chess_engine.generate_self_play_games_with_diagnostics(
            evaluateur, args.concurrent, args.slow_sims, args.fast_sims,
            args.games, args.ratio, args.tt, puzzles, mode,
            args.start_positions))
    zones["appel_cpp"] = time.perf_counter() - debut

    debut = time.perf_counter()
    data, _ = convert_game_results(parties)
    zones["conversion"] = time.perf_counter() - debut
    exemples = len(data)
    parties_jouees = len(parties)
    coups_nouveaux = int(stats.new_plies)

    debut = time.perf_counter()
    del data, parties, evaluateur
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pic_vram_mio = torch.cuda.max_memory_allocated() / (1 << 20)
    zones["nettoyage"] = time.perf_counter() - debut

    total_python = sum(zones.values())
    lignes_reseau = int(timing.batch_rows) + int(timing.unit_network_rows)
    debit = {
        "coups_nouveaux_par_s": coups_nouveaux / zones["appel_cpp"]
        if zones["appel_cpp"] else 0.0,
        "exemples_par_s": exemples / total_python if total_python else 0.0,
        "parties_par_s": parties_jouees / total_python
        if total_python else 0.0,
        "evaluations_reseau_par_s": lignes_reseau / zones["appel_cpp"]
        if zones["appel_cpp"] else 0.0,
    }
    resume = resumer_passage(timing, zones, debit)
    resume["fin"]["parties_terminees"] = int(stats.games_completed)
    resume["fin"]["coups_nouveaux"] = coups_nouveaux
    resume["fin"]["historique_rejoue"] = int(stats.replayed_plies)
    resume["fin"]["exemples_conserves"] = exemples
    resume["memoire"] = {"vram_pic_mio": pic_vram_mio}
    return resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELE_DEFAUT)
    parser.add_argument("--games", type=int, default=128,
                        help="parties demandees par passage")
    parser.add_argument("--concurrent", type=int, default=128,
                        help="places simultanees")
    parser.add_argument("--slow-sims", type=int, default=700)
    parser.add_argument("--fast-sims", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--tt", type=int, default=4_000_000)
    parser.add_argument("--puzzles", default=PUZZLES_DEFAUT)
    parser.add_argument("--start-positions", default="",
                        help="fichier de FEN, une par ligne : chaque partie part "
                             "d'une position tiree au hasard, sans puzzle")
    parser.add_argument("--mode", type=int, default=1, choices=(0, 1, 2),
                        help="0 instrumentation eteinte (cout du profilage), "
                             "1 phases, 2 phases et detail par worker")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--output", default=str(RACINE / "out" /
                                                "selfplay_diagnostics"))
    args = parser.parse_args()

    import chess_engine
    import torch  # noqa: F401  (charge les DLL CUDA de torch avant ONNX)

    if not os.path.isfile(args.model):
        raise SystemExit(f"modele introuvable : {args.model}")
    puzzles = args.puzzles if os.path.isfile(args.puzzles) else ""
    if not puzzles:
        print("[diag] puzzles introuvables, injection desactivee")

    racine = RACINE
    configuration = collecter_configuration(
        modele=args.model,
        binaire=chess_engine.__file__,
        puzzles=puzzles,
        concurrent=args.concurrent,
        games=args.games,
        slow_sims=args.slow_sims,
        fast_sims=args.fast_sims,
        slow_ratio=args.ratio,
        tt_size=args.tt,
        mode=args.mode,
        passes=args.passes,
        racine=racine)
    configuration["date"] = dt.datetime.now().isoformat(timespec="seconds")
    configuration["positions_depart"] = args.start_positions

    os.makedirs(args.output, exist_ok=True)
    horodatage = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_json = os.path.join(
        args.output, f"diag_{horodatage}_mode{args.mode}.json")

    resumes = []
    for passage in range(args.passes):
        print(f"[diag] passage {passage + 1}/{args.passes} "
              f"({args.concurrent} places, {args.games} parties, "
              f"{args.slow_sims}/{args.fast_sims} simulations, mode "
              f"{args.mode})", flush=True)
        resume = mesurer_passage(args, args.mode, puzzles)
        resume["config"] = configuration
        resumes.append(resume)
        if args.mode == 0:
            # Rapport de phases vide par construction : le mode 0 sert a
            # mesurer le cout du profilage, seul le debit Python parle.
            print(f"[diag] instrumentation eteinte : appel C++ "
                  f"{resume['temps']['zones_python_s']['appel_cpp']:.1f} s, "
                  f"coups nouveaux/s "
                  f"{resume['debit']['coups_nouveaux_par_s']:.2f}", flush=True)
        else:
            print(formater_diagnostic(resume), flush=True)

    print(formater_bilan(resumes), flush=True)
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump({"configuration": configuration, "passages": resumes},
                  fichier, indent=2, sort_keys=True)
    print(f"[diag] rapport ecrit : {chemin_json}")


if __name__ == "__main__":
    main()
