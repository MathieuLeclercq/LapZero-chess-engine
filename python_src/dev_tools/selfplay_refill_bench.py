"""Banc de debit self-play : nombre de places contre horizon de generation.

Le SelfPlayManager joue `total` parties avec `concurrent` places. Depuis le
correctif R9, le nombre de departs est exactement `total` : les places sont
relancees tant qu'il reste des departs a effectuer, puis les parties engagees
finissent et sont toutes collectees. Il n'y a plus de partie abandonnee.

Ce banc compare des couples (concurrent, total) en mesurant le debit reel :
temps mural, parties par seconde, coups joues par seconde. Un total egal au
nombre de places ne vide plus le pool au fil des fins, il reduit simplement le
renouvellement a zero.

Attention : les couples ou total est inferieur au nombre de places n'utilisent
qu'une partie des places (min(concurrent, total) parties demarrent).

Outil de diagnostic, hors validation.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402,F401  (charge les DLL CUDA de torch : cublasLt)
import chess_engine  # noqa: E402

from core.data_paths import ONNX_DIR
MODELE = str(ONNX_DIR / "2026_04_30_09h53_iter436_unsupervised.onnx")


def mesurer(evaluateur, concurrent, total, slow, fast, ratio, tt_size):
    """Une configuration (places, horizon), avec les compteurs du scheduler.

    Le debit de jeu nouveau se mesure sur les plies reellement joues, jamais sur
    total_real_moves, qui inclut l'historique rejoue d'un puzzle et gonflerait
    un compteur presente comme du jeu produit.
    """
    debut = time.perf_counter()
    parties, stats = chess_engine.generate_self_play_games_with_stats(
        evaluateur, concurrent, slow, fast, total, ratio, tt_size)
    duree = time.perf_counter() - debut

    positions = sum(p.state_tensors.shape[0] for p in parties)
    return {
        "concurrent": concurrent,
        "total": total,
        "duree": duree,
        "parties": len(parties),
        "debuts": int(stats.games_started),
        "fins": int(stats.games_completed),
        "actives": int(stats.active_slots),
        "nouveaux_plies": int(stats.new_plies),
        "historique_rejoue": int(stats.replayed_plies),
        "positions": positions,
        "parties_par_s": len(parties) / duree,
        "nouveaux_plies_par_s": stats.new_plies / duree,
        "ms_par_ply": (1000.0 * duree / stats.new_plies
                       if stats.new_plies else 0.0),
        "positions_par_s": positions / duree,
    }


def formater_bilan(resultats):
    """Tableau honnete : debuts, fins, plies nouveaux et historique rejoue sont
    des colonnes distinctes, et le debit n'utilise que les plies nouveaux."""
    entete = ("  {:>11} {:>7} {:>6} {:>8} {:>11} {:>12} {:>9} {:>10} "
              "{:>14} {:>8}")
    lignes = ["[bench]" + entete.format(
        "places/horizon", "debuts", "fins", "actives", "nouv. plies",
        "hist. rejoue", "exemples", "duree (s)", "nouv. plies/s", "ms/ply")]
    for r in resultats:
        lignes.append("[bench]" + entete.format(
            f"({r['concurrent']},{r['total']})", str(r["debuts"]),
            str(r["fins"]), str(r["actives"]), str(r["nouveaux_plies"]),
            str(r["historique_rejoue"]), str(r["positions"]),
            f"{r['duree']:.0f}", f"{r['nouveaux_plies_par_s']:.1f}",
            f"{r['ms_par_ply']:.2f}"))
    return lignes


def afficher(resultats):
    for ligne in formater_bilan(resultats):
        print(ligne)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELE)
    parser.add_argument("--sims", type=int, default=100,
                        help="simulations par coup lent (defaut 100)")
    parser.add_argument("--fast", type=int, default=20,
                        help="simulations par coup rapide (defaut 20)")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--tt", type=int, default=4_000_000)
    parser.add_argument("--paires", default="32:1,32:4,64:1,64:4",
                        help="liste concurrent:total, total en multiples de concurrent")
    args = parser.parse_args()

    print(f"[bench] modele {os.path.basename(args.model)}")
    print(f"[bench] simulations {args.sims} lentes / {args.fast} rapides, "
          f"ratio {args.ratio}")

    evaluateur = chess_engine.ONNXEvaluator(args.model, True)
    resultats = []
    for paire in args.paires.split(","):
        concurrent, multiple = (int(x) for x in paire.split(":"))
        total = concurrent * multiple
        print(f"[bench] lancement pool ({concurrent},{total})", flush=True)
        resultats.append(mesurer(evaluateur, concurrent, total, args.sims,
                                 args.fast, args.ratio, args.tt))

    afficher(resultats)

    for i in range(0, len(resultats) - 1, 2):
        a, b = resultats[i], resultats[i + 1]
        if a["concurrent"] == b["concurrent"]:
            print(f"[bench] places {a['concurrent']} : ({a['concurrent']},"
                  f"{a['total']}) contre ({b['concurrent']},{b['total']}) = "
                  f"x{b['nouveaux_plies_par_s'] / a['nouveaux_plies_par_s']:.2f} "
                  f"sur les plies nouveaux, "
                  f"x{b['parties_par_s'] / a['parties_par_s']:.2f} sur les parties, "
                  f"{a['ms_par_ply']:.2f} -> {b['ms_par_ply']:.2f} ms/ply")


if __name__ == "__main__":
    main()
