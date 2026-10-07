"""Banc du cas chaud de l'arbre UCI.

Joue une position avec un MCTS persistant, en tranches de simulations, et
applique update_root apres chaque coup comme le fait le bot. Rapport par coup :
debit, appels reseau, hits de table et fraction de simulations qui evitent le
reseau.

Le resultat principal du 2026-09-19 : un hit de table n'evite pas l'inference,
car il materialise les enfants et la descente continue vers une feuille reseau.
Voir docs/etudes/2026-09-19-cout-calcul-cpu-gpu/protocole/2026-09-19-cout-calcul-cpu-gpu.md, section 8.4.
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import json
import os
import statistics
import sys
import time
from pathlib import Path

RACINE_PYTHON = _runtime.PYTHON_ROOT
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))
os.add_dll_directory(str(RACINE_PYTHON))

import chess_engine

POSITIONS = (
    ("ouverture",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("finale", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)


def agreger_coups(lignes: list) -> dict:
    """Resume par position : medianes de debit, d'appels reseau et de fraction
    de simulations sans inference."""
    groupes: dict = {}
    for ligne in lignes:
        groupes.setdefault(ligne["position"], []).append(ligne)

    resumes = {}
    for position, lot in groupes.items():
        resumes[position] = {
            "coups": len(lot),
            "nps_median": statistics.median(
                ligne["sims_par_seconde"] for ligne in lot),
            "nn_calls_median": statistics.median(
                ligne["nn_calls"] for ligne in lot),
            "sans_reseau_median": statistics.median(
                ligne["sans_reseau"] for ligne in lot),
        }
    return resumes


def jouer_position(mcts, nom: str, fen: str, coups: int, simulations: int,
                   tranche: int) -> list:
    from evaluation.bench_metrics import index_to_uci

    board = chess_engine.Chessboard()
    board.load_fen(fen)
    mcts.reset_analysis()

    lignes = []
    for coup in range(coups):
        mcts.reset_counters()
        debut = time.perf_counter()
        fait = 0
        while fait < simulations:
            bloc = min(tranche, simulations - fait)
            mcts.step_analysis(board, bloc, 1.4, 8, 8)
            fait += bloc
        duree = time.perf_counter() - debut

        resultats = mcts.get_analysis_results()
        if not resultats:
            break
        compteurs = mcts.get_counters()
        lignes.append({
            "position": nom,
            "coup": coup,
            "sims": simulations,
            "sims_par_seconde": simulations / duree,
            "nn_calls": compteurs.nn_calls,
            "tt_hits": compteurs.tt_hits,
            "terminal_hits": compteurs.terminal_hits,
            "sans_reseau": (simulations - compteurs.nn_calls) / simulations,
        })

        meilleur = resultats[0].move_idx
        if not board.move_piece_uci(index_to_uci(board, meilleur)):
            break
        mcts.update_root(meilleur)
    return lignes


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--moves", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=700)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"modele introuvable : {args.model}", file=sys.stderr)
        return 2

    if args.gpu:
        import torch  # noqa: F401  (DLL CUDA)

    evaluateur = chess_engine.ONNXEvaluator(str(args.model), args.gpu)
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)
    mcts.set_fixed_batch(True)

    lignes = []
    for nom, fen in POSITIONS:
        lignes.extend(jouer_position(
            mcts, nom, fen, args.moves, args.simulations, args.slices))

    for nom, resume in agreger_coups(lignes).items():
        print(f"{nom:10} : {resume['nps_median']:7.1f} sims/s median, "
              f"{resume['nn_calls_median']:.0f} appels reseau median, "
              f"sans reseau {100 * resume['sans_reseau_median']:.1f} %")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(lignes, indent=2), encoding="utf-8")
        print(f"Mesures : {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
