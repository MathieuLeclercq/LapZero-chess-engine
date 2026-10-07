"""Lance la boucle self-play + entrainement sur une machine distante.

Ne touche pas a la configuration de train_self_play.py : toutes les valeurs
qui changent d'une campagne a l'autre sont des arguments. Les defauts
reprennent la production locale (512 parties, 256 places, 700/100 simulations).

A lancer depuis le dossier python_src pour conserver le replay_buffer existant.
Les modeles sont lus et ecrits dans data/checkpoints, quel que soit le terminal.

L'evaluation periodique est le banc externe de positions, toutes les 4
iterations par defaut (`--eval-every 0` le desactive). L'ancrage Stockfish
n'est plus lance par la boucle ; `stockfish_player.py` reste l'outil manuel.

Exemple :

    python run_selftrain.py --iterations 4
    python run_selftrain.py --iterations 4 --wandb-mode disabled
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import argparse
import glob
import os
from core.data_paths import DEFAULT_CHECKPOINT


def etat_du_buffer(dossier):
    """Compte les shards et les positions deja presentes dans le buffer."""
    shards = sorted(glob.glob(os.path.join(dossier, "shard_*.npz")))
    positions = 0
    for shard in shards:
        try:
            positions += int(os.path.splitext(shard)[0].split("_")[-1])
        except (ValueError, IndexError):
            continue
    return shards, positions


def date_du_shard(chemin):
    morceaux = os.path.basename(chemin).split("_")
    if len(morceaux) > 2 and len(morceaux[1]) == 8:
        return f"{morceaux[1][:4]}-{morceaux[1][4:6]}-{morceaux[1][6:8]}"
    return "?"


def analyser_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--games", type=int, default=512,
                        help="parties par iteration")
    parser.add_argument("--concurrent", type=int, default=256,
                        help="places du pool self-play")
    parser.add_argument("--slow-sims", type=int, default=700)
    parser.add_argument("--fast-sims", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--buffer", type=int, default=750_000,
                        help="plafond du replay buffer sur disque")
    parser.add_argument("--sampling-ratio", type=float, default=14.0)
    parser.add_argument("--dataloader-workers", type=int, default=0,
                        help="processus de chargement des donnees ; sur Windows "
                             "chaque worker relance Python, donc 0 par defaut")
    parser.add_argument("--eval-every", type=int, default=4,
                        help="banc de positions tous les N tours ; 0 desactive")
    parser.add_argument("--position-bench", default=None,
                        help="dossier du banc ; par defaut data/position_bench/v1")
    parser.add_argument("--eval-search-workers", type=int, default=8)
    parser.add_argument("--eval-target-seconds", type=float, default=300.0)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt de reprise ; par defaut iter436")
    parser.add_argument("--wandb-mode", choices=["offline", "disabled", "online"],
                        default="offline",
                        help="offline ecrit les metriques en local sans reseau, "
                             "disabled ne logue rien, online envoie directement")
    parser.add_argument("--selfplay-diagnostics", type=int, default=0,
                        choices=(0, 1, 2),
                        help="0 desactive, 1 phases, compteurs de recherche et "
                             "memoire, 2 ajoute le detail par worker")
    parser.add_argument("--fast-dirichlet-noise",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="bruit de Dirichlet sur les coups rapides ; "
                             "--no-fast-dirichlet-noise le reserve aux coups "
                             "lents")
    return parser.parse_args()


def main():
    args = analyser_arguments()

    if args.wandb_mode != "online":
        os.environ["WANDB_MODE"] = args.wandb_mode

    shards, positions = etat_du_buffer("replay_buffer")
    if shards:
        print(f"[Buffer] {len(shards)} shards, {positions} positions, "
              f"du {date_du_shard(shards[0])} au {date_du_shard(shards[-1])} "
              f"(plafond {args.buffer})")
    else:
        print(f"[Buffer] vide (plafond {args.buffer})")

    from training.train_self_play import pipeline

    pipeline(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        concurrent_games=args.concurrent,
        slow_sims=args.slow_sims,
        fast_sims=args.fast_sims,
        slow_ratio=0.25,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_buffer_size=args.buffer,
        target_sampling_ratio=args.sampling_ratio,
        eval_every=args.eval_every,
        position_bench_path=args.position_bench,
        eval_search_workers=args.eval_search_workers,
        eval_target_s=args.eval_target_seconds,
        eval_output_dir=args.eval_output_dir,
        checkpoint_path=(args.checkpoint
                         or str(DEFAULT_CHECKPOINT)),
        data_workers=args.dataloader_workers,
        selfplay_diagnostics=args.selfplay_diagnostics,
        fast_dirichlet_noise=args.fast_dirichlet_noise,
    )


if __name__ == "__main__":
    main()
