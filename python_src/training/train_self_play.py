
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import argparse
import functools
import gc
import logging
import math
import os
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from core.data_paths import CHECKPOINTS_DIR, ONNX_DIR, DEFAULT_CHECKPOINT

import numpy as np

warnings.filterwarnings("ignore", module="requests")

import chess_engine
import torch
import torch.nn.functional as f
from core.lib import (
    append_to_disk_buffer,
    convert_game_results,
    export_model_to_onnx,
    run_with_interrupt,
)
from core.model import ChessNet
from monitoring.moniteur_memoire import mesurer_memoire, metriques_memoire
from evaluation.position_bench import evaluate_checkpoint, find_previous, protocol_id
from evaluation.position_bench_metrics import EvalConfig, SearchConfig
from evaluation.position_bench_results import load_dataset, wandb_metrics
from training.replay_loader import iter_replay_batches, lister_shards
from torch.amp import GradScaler

import wandb

# Chemin absolu : l'injection de puzzles est chargee cote C++ et un chemin
# relatif dependrait du repertoire de lancement.
PUZZLES_PATH = str(_runtime.REPO_ROOT
                   / "training_data" / "puzzles_train.txt")

RACINE = _runtime.REPO_ROOT
BANC_DEFAUT = RACINE / "data" / "position_bench" / "v1"
RESULTATS_EVAL_DEFAUT = RACINE / "position_bench_results"


# ============================================================
#                     SELF-PLAY (GPU Batched)
# ============================================================
def generate_games(
        onnx_path,
        games_per_iter,
        concurrent_games,
        slow_sims,
        fast_sims,
        slow_ratio,
        tt_size=2097143,
        diagnostics=0,
        fast_dirichlet_noise=True):
    """
    Génère des parties via le SelfPlayManager C++ avec inférence batchée sur GPU.

    Quand `diagnostics` vaut 1 ou 2, le rapport de phases C++ est imprimé et
    renvoyé ; le déroulement de la génération reste identique. Le chronométrage
    Python sépare la création de l'évaluateur, l'appel C++, la conversion et le
    nettoyage, pour que chaque poste ait son propre dénominateur.

    `fast_dirichlet_noise` est transmis par nom au binding : à faux, seuls les
    coups lents recoivent le bruit de Dirichlet.
    """
    zones = {"creation_evaluateur": 0.0, "appel_cpp": 0.0,
             "conversion": 0.0, "nettoyage": 0.0}
    memoire = {}

    debut = time.perf_counter()
    evaluator = chess_engine.ONNXEvaluator(onnx_path, True)
    zones["creation_evaluateur"] = time.perf_counter() - debut
    if diagnostics > 0:
        memoire["apres_creation_evaluateur"] = mesurer_memoire()

    debut = time.perf_counter()
    if diagnostics > 0:
        game_results, stats_selfplay, timing = run_with_interrupt(
            functools.partial(
                chess_engine.generate_self_play_games_with_diagnostics,
                fast_dirichlet_noise=fast_dirichlet_noise),
            evaluator,
            concurrent_games,
            slow_sims,
            fast_sims,
            games_per_iter,
            slow_ratio,
            tt_size,
            PUZZLES_PATH,
            diagnostics
        )
    else:
        game_results = run_with_interrupt(
            functools.partial(
                chess_engine.generate_self_play_games,
                fast_dirichlet_noise=fast_dirichlet_noise),
            evaluator,
            concurrent_games,
            slow_sims,
            fast_sims,
            games_per_iter,
            slow_ratio,
            tt_size,
            PUZZLES_PATH
        )
        stats_selfplay = None
        timing = None
    zones["appel_cpp"] = time.perf_counter() - debut
    if diagnostics > 0:
        memoire["apres_generation_cpp"] = mesurer_memoire()

    debut = time.perf_counter()
    data, stats = convert_game_results(game_results)
    zones["conversion"] = time.perf_counter() - debut
    if diagnostics > 0:
        memoire["apres_conversion"] = mesurer_memoire()

    num_games = len(game_results)
    total_ply_played = sum(res.total_real_moves for res in game_results)
    avg_length = total_ply_played / num_games

    print(f"\n{'=' * 30}")
    print("      BILAN DE L'ITERATION")
    print(f"{'=' * 30}")
    print(f"  Parties jouées                     : {num_games}")
    print(f"  Positions générées (slow moves)    : {len(data)}")
    print(f"  Positions par Partie               : {avg_length:.1f}")
    print(f"{'-' * 30}")
    print(f"  Victoires (mat)                    : {stats['checkmates']}")
    print(f"  Pat (Stalemate)                    : {stats['stalemates']}")
    print(f"  Répétition                         : {stats['repetition']}")
    print(f"  Règle des 50 coups                 : {stats['50_moves']}")
    print(f"  Matériel insuffisant               : {stats['insuff_mat']}")
    print(f"  Non terminées (Max)                : {stats['max_moves']}")
    print(f"{'=' * 30}\n")

    # Libération explicite de l'évaluateur ONNX GPU
    debut = time.perf_counter()
    del game_results
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    zones["nettoyage"] = time.perf_counter() - debut
    if diagnostics > 0:
        memoire["apres_nettoyage_selfplay"] = mesurer_memoire()

    diagnostic = None
    if diagnostics > 0:
        from training.selfplay_diag import formater_diagnostic, resumer_passage

        assert stats_selfplay is not None and timing is not None
        total_python = sum(zones.values())
        debit = {
            "coups_nouveaux_par_s": (
                int(stats_selfplay.new_plies) / zones["appel_cpp"]
                if zones["appel_cpp"] else 0.0),
            "exemples_par_s": (len(data) / total_python
                               if total_python else 0.0),
            "parties_par_s": (num_games / total_python
                              if total_python else 0.0),
            "evaluations_reseau_par_s": (
                (int(timing.batch_rows) + int(timing.unit_network_rows))
                / zones["appel_cpp"] if zones["appel_cpp"] else 0.0),
        }
        diagnostic = resumer_passage(timing, zones, debit)
        diagnostic["memoire"] = memoire
        diagnostic["fin"]["parties_terminees"] = int(
            stats_selfplay.games_completed)
        diagnostic["fin"]["coups_nouveaux"] = int(stats_selfplay.new_plies)
        diagnostic["fin"]["historique_rejoue"] = int(
            stats_selfplay.replayed_plies)
        diagnostic["fin"]["exemples_conserves"] = len(data)
        print(formater_diagnostic(diagnostic))

    return data, avg_length, stats, diagnostic


# ============================================================
#                     TRAINING
# ============================================================
def train_on_shards(model, optimizer, scaler, device, buffer_folder, learning_rate,
                    batch_size=256, global_step=0, samples_per_epoch=15000,
                    data_workers=0, training_stats=None):
    """Entraîne sur `samples_per_epoch` tirages équilibrés du replay buffer.

    Le chargeur (`replay_loader`) tire exactement S exemples avec remise, les
    regroupe par fichier puis émet `ceil(S/B)` lots de tailles proches depuis
    un réservoir borné : aucune mise à jour à une frontière de fichier, aucun
    petit reliquat et un mélange borné. Les losses journalisées sont pondérées
    par l'effectif des lots ; `global_step` compte chaque tentative, comme le
    scaler AMP.
    """
    if data_workers != 0:
        raise ValueError(
            "data_workers doit valoir 0 : le chargeur replay est synchrone")
    if samples_per_epoch < 0:
        raise ValueError("samples_per_epoch doit être positif ou nul")
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif")
    if training_stats is not None:
        training_stats["samples_seen"] = 0

    shards = lister_shards(buffer_folder)
    if samples_per_epoch == 0 or not shards:
        return global_step

    model.train()
    # Générateur NumPy initialisé depuis le RNG CPU de PyTorch : torch.manual_seed
    # contrôle ainsi aussi les tirages du chargeur.
    graine = int(torch.randint(0, 2**63 - 1, (1,), device="cpu").item())
    rng = np.random.default_rng(graine)

    epoch_loss = 0.0
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    samples_seen = 0
    num_batches = 0
    batch_size_min = None
    batch_size_max = None

    for x, target_pi, y_value in iter_replay_batches(
            shards, samples_per_epoch, batch_size, rng=rng):
        x = x.to(device)
        target_pi = target_pi.to(device)
        y_value = y_value.to(device)
        effectif = int(x.size(0))

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            p_logits, v_pred = model(x)
            log_probs = f.log_softmax(p_logits, dim=1)
            policy_loss = -torch.sum(target_pi * log_probs, dim=1).mean()
            value_loss = f.mse_loss(v_pred.view(-1), y_value)
            loss = policy_loss + value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Moyennes pondérées par l'effectif : la moyenne de lots de tailles
        # différentes n'est pas la moyenne des exemples. La tentative compte
        # même si le scaler AMP saute la mise à jour, comme global_step ; ce
        # saut n'est pas observable via l'API publique du scaler.
        epoch_loss += loss.item() * effectif
        epoch_policy_loss += policy_loss.item() * effectif
        epoch_value_loss += value_loss.item() * effectif
        samples_seen += effectif
        num_batches += 1
        global_step += 1
        if batch_size_min is None:
            batch_size_min = effectif
            batch_size_max = effectif
        else:
            batch_size_min = min(batch_size_min, effectif)
            batch_size_max = max(batch_size_max, effectif)

    if num_batches > 0:
        diviseur = max(1, samples_seen)
        wandb.log({
            "train/epoch_loss": epoch_loss / diviseur,
            "train/epoch_policy_loss": epoch_policy_loss / diviseur,
            "train/epoch_value_loss": epoch_value_loss / diviseur,
            "train/global_step": global_step,
            "train/learning_rate": learning_rate,
            "train/samples_seen": samples_seen,
            "train/num_batches": num_batches,
            "train/batch_size_min": batch_size_min,
            "train/batch_size_max": batch_size_max,
        }, step=global_step)
        print(f"    Training — loss: {epoch_loss / diviseur:.4f} "
              f"({num_batches} batches)")

    # Compteur reel consomme par la passe, meme si le scaler AMP a saute un pas.
    if training_stats is not None:
        training_stats["samples_seen"] = samples_seen
    torch.cuda.empty_cache()
    return global_step


def _sauver_checkpoint(chemin, charge):
    torch.save(charge, chemin)


# ============================================================
#                     PIPELINE
# ============================================================
def pipeline(
        num_iterations=2,
        games_per_iter=128,
        concurrent_games=128,
        slow_sims=700,
        fast_sims=100,
        slow_ratio=0.25,
        batch_size=1024,
        learning_rate=1e-4,
        num_res_blocks=10,
        num_filters=128,
        max_buffer_size=100_000,
        target_sampling_ratio=14.0,
        eval_every=4,
        position_bench_path=None,
        eval_search_workers=8,
        eval_target_s=300.0,
        eval_output_dir=None,
        checkpoint_path=None,
        data_workers=0,
        selfplay_diagnostics=0,
        wandb_mode=None,
        dependances=None,
        eval_stockfish_every=None,
        stockfish_path=None,
        stockfish_elo=None,
        stockfish_nodes=None,
        num_sim_eval_sf=None,
        fast_dirichlet_noise=True,
):
    """Boucle self-play + entrainement, avec banc de positions integre.

    La cadence du banc vaut `eval_every` iterations, 4 par defaut ; 0 la
    desactive. `position_bench_path=None` resout le banc dans le depot, jamais
    depuis le repertoire courant. Les anciens parametres Stockfish sont refuses
    avec un message explicite : l'ancrage se lance manuellement, avec
    stockfish_player.py.
    """
    logging.getLogger("torch").setLevel(logging.ERROR)

    if stockfish_path is not None or stockfish_elo is not None \
            or stockfish_nodes is not None or num_sim_eval_sf is not None:
        raise ValueError(
            "l'ancrage Stockfish n'est plus lance par la boucle de self-play ; "
            "utiliser stockfish_player.py manuellement")
    if eval_stockfish_every is not None:
        print("[avertissement] eval_stockfish_every est devenu eval_every")
        eval_every = eval_stockfish_every
    if eval_every < 0:
        raise ValueError("eval_every doit etre positif ou nul")
    target_sampling_ratio = float(target_sampling_ratio)
    if not math.isfinite(target_sampling_ratio) or target_sampling_ratio <= 0:
        raise ValueError(
            "target_sampling_ratio doit etre fini et strictement positif")

    def composant(cle, defaut):
        if dependances and cle in dependances:
            return dependances[cle]
        return defaut

    cuda_disponible = composant("cuda_disponible", torch.cuda.is_available)
    if not cuda_disponible():
        raise RuntimeError("un GPU CUDA est requis pour la boucle self-play")
    gpu_device = torch.device("cuda")

    # Prevalidation du banc avant la premiere generation : sans dataset, on
    # s'arrete ici avec un message, jamais par une substitution Stockfish.
    if eval_every > 0:
        if position_bench_path is None:
            position_bench_path = BANC_DEFAUT
        if eval_output_dir is None:
            eval_output_dir = RESULTATS_EVAL_DEFAUT
        charger_banc = composant("charger_banc", load_dataset)
        manifeste, _ = charger_banc(position_bench_path)
    else:
        manifeste = None

    hyperparams = {
        "num_iterations": num_iterations,
        "games_per_iter": games_per_iter,
        "concurrent_games": concurrent_games,
        "slow_sims": slow_sims,
        "fast_sims": fast_sims,
        "slow_ratio": slow_ratio,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_res_blocks": num_res_blocks,
        "num_filters": num_filters,
        "max_buffer_size": max_buffer_size,
        "target_sampling_ratio": target_sampling_ratio,
        "eval_every": eval_every,
        "position_bench_path": (str(position_bench_path)
                                if position_bench_path else None),
        "eval_search_workers": eval_search_workers,
        "eval_target_s": eval_target_s,
        "eval_output_dir": (str(eval_output_dir)
                            if eval_output_dir else None),
        "data_workers": data_workers,
        "selfplay_diagnostics": selfplay_diagnostics,
        "fast_dirichlet_noise": fast_dirichlet_noise,
    }

    timestamp = datetime.now(UTC).strftime("%Y_%m_%d_%Hh%M")
    creer_modele = composant(
        "creer_modele",
        lambda: ChessNet(num_res_blocks=num_res_blocks,
                         num_filters=num_filters).to(gpu_device))
    model = creer_modele()
    creer_optimiseur = composant(
        "creer_optimiseur",
        lambda parametres: torch.optim.AdamW(parametres, lr=learning_rate,
                                             weight_decay=1e-4))
    optimizer = creer_optimiseur(model.parameters())
    creer_scaler = composant("creer_scaler",
                             lambda: GradScaler("cuda", enabled=True))
    scaler = creer_scaler()
    charger_checkpoint = composant(
        "charger_checkpoint",
        lambda chemin: torch.load(chemin, map_location=gpu_device,
                                  weights_only=True))

    global_step = 0
    start_iteration = 0

    if checkpoint_path:
        checkpoint = charger_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_iteration = checkpoint.get("iteration", 0)
        global_step = checkpoint.get("global_step", 0)
        print(f"Checkpoint chargé : {checkpoint_path} (Reprise à l'itération {start_iteration})")

    module_wandb = composant("wandb", wandb)
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode
    module_wandb.init(project="alphazero-chess",
                      name=f"{timestamp}_self_play", config=hyperparams)

    generer = composant("generer_parties", generate_games)
    entrainer = composant("entrainer", train_on_shards)
    exporter = composant("exporter", export_model_to_onnx)
    sauver = composant("sauver_checkpoint", _sauver_checkpoint)
    ecrire_buffer = composant("ecrire_buffer", append_to_disk_buffer)
    evaluer = composant("evaluer", evaluate_checkpoint)
    chercher_precedent = composant("chercher_precedent", find_previous)
    calculer_protocole = composant("protocol_id", protocol_id)
    horloge = composant("horloge", time.time)
    mesurer = composant("mesurer_memoire", mesurer_memoire)

    buffer_folder = "replay_buffer"

    # ── INITIALISATION AVANT LA BOUCLE (Génération de l'iter 0 ou reprise) ──
    current_onnx_name = f"{timestamp}_iter{start_iteration}_unsupervised"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    current_onnx_path = str(ONNX_DIR / f"{current_onnx_name}.onnx")
    exporter(model, current_onnx_path, gpu_device)
    print(f"Modèle ONNX initial prêt pour le self-play : {current_onnx_path}")

    # Frequence effective et charge du CPU pendant chaque generation : sans
    # elles, un bridage du CPU ne se distingue pas d'un autre ralentissement.
    moniteur_cpu = None
    if selfplay_diagnostics > 0:
        from monitoring.moniteur_cpu import creer_echantillonneur
        moniteur_cpu = creer_echantillonneur()

    for iteration in range(start_iteration, start_iteration + num_iterations):
        print(f"\n{'=' * 50}")
        print(f"  ITERATION {iteration + 1}/{start_iteration + num_iterations}")
        print(f"{'=' * 50}")

        memoire = {}
        if selfplay_diagnostics > 0:
            memoire["avant_selfplay"] = mesurer()
        # ── 1. Self-Play (C++ / GPU batched) ──
        if moniteur_cpu is not None:
            moniteur_cpu.reinitialiser()
        start_time = horloge()
        new_data, avg_length, stats, diagnostic = generer(
            current_onnx_path, games_per_iter,
            concurrent_games, slow_sims, fast_sims, slow_ratio,
            tt_size=4_000_000,
            diagnostics=selfplay_diagnostics,
            fast_dirichlet_noise=fast_dirichlet_noise
        )
        generation_time = horloge() - start_time
        if selfplay_diagnostics > 0 and diagnostic is not None:
            memoire.update(diagnostic.get("memoire", {}))
        metriques_cpu = (moniteur_cpu.metriques("selfplay/cpu")
                         if moniteur_cpu is not None else {})
        games_per_sec = games_per_iter / generation_time
        saved_pos_per_sec = len(new_data) / generation_time

        print(
            f"  Vitesse : {games_per_sec:.2f} parties/s | "
            f"{saved_pos_per_sec:.0f} saved positions/s (Total: {generation_time:.1f}s)")

        # ── 2. Sauvegarde des nouvelles positions sur disque ──
        buffer_size = ecrire_buffer(new_data, buffer_folder, max_buffer_size)
        num_new_positions = len(new_data)
        del new_data
        if selfplay_diagnostics > 0:
            memoire["apres_liberation_donnees"] = mesurer()

        # ── 3. Training (GPU / PyTorch) ──
        samples_per_epoch = round(target_sampling_ratio * num_new_positions)
        print(f"  Entraînement sur {samples_per_epoch} samples...")

        # Compteur reel de la passe pour le ratio effectif : jamais le budget
        # demande, qui peut differer des tirages consommes.
        training_stats = {}
        global_step = entrainer(
            model, optimizer, scaler, gpu_device, buffer_folder, learning_rate,
            batch_size=batch_size, global_step=global_step,
            samples_per_epoch=samples_per_epoch, data_workers=data_workers,
            training_stats=training_stats
        )

        if selfplay_diagnostics > 0:
            memoire["apres_entrainement"] = mesurer()

        num_games = max(1, games_per_iter)
        draw_rate = 1 - (stats["checkmates"] / num_games)
        journal = {
            "selfplay/buffer_size": buffer_size,
            "selfplay/new_positions": num_new_positions,
            "selfplay/avg_game_length": avg_length,
            "selfplay/draw_rate": draw_rate,
            "selfplay/draws_repetition": stats["repetition"] / num_games,
            "selfplay/draws_50_moves": stats["50_moves"] / num_games,
            "selfplay/draws_stalemate": stats["stalemates"] / num_games,
            "selfplay/draws_insuff_mat": stats["insuff_mat"] / num_games,
            "selfplay/draws_max_moves": stats["max_moves"] / num_games,
            "selfplay/games_per_sec": games_per_sec,
            "selfplay/saved_positions_per_sec": saved_pos_per_sec,
            "selfplay/iteration": iteration + 1,
            "train/replay_ratio_target": target_sampling_ratio,
        }
        if num_new_positions > 0:
            journal["train/replay_ratio_effective"] = (
                training_stats.get("samples_seen", 0) / num_new_positions)
        if diagnostic is not None:
            from training.selfplay_diag import metriques_wandb
            journal.update(metriques_wandb(diagnostic))
        journal.update(metriques_cpu)

        # ── 4. Sauvegarde checkpoint .pt ET .onnx ──
        ckpt_filename = f"{timestamp}_iter{iteration + 1}_unsupervised"
        save_pt_path = str(CHECKPOINTS_DIR / f"{ckpt_filename}.pt")

        sauver(save_pt_path, {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "iteration": iteration + 1,
            "global_step": global_step,
        })

        if selfplay_diagnostics > 0:
            memoire["apres_sauvegarde_pt"] = mesurer()

        # Nouvel export ONNX qui servira pour l'itération suivante et l'évaluation
        current_onnx_path = str(ONNX_DIR / f"{ckpt_filename}.onnx")
        exporter(model, current_onnx_path, gpu_device)
        if selfplay_diagnostics > 0:
            memoire["apres_export_onnx"] = mesurer()

        print(f"  Checkpoints sauvegardés : {ckpt_filename} (.pt et .onnx)")

        # ── 5. Évaluation du banc externe de positions ──
        if eval_every > 0 and (iteration + 1) % eval_every == 0:
            assert manifeste is not None
            config = EvalConfig(
                search=SearchConfig(workers=eval_search_workers),
                target_s=eval_target_s)
            precedent = chercher_precedent(
                eval_output_dir, manifeste["dataset_sha256"],
                calculer_protocole(config), iteration + 1)
            if precedent is None:
                print("  Aucun resultat precedent compatible : comparaison nulle.")
            rapport = evaluer(
                Path(current_onnx_path), position_bench_path,
                eval_output_dir, config,
                iteration=iteration + 1, global_step=global_step,
                previous=precedent)
            journal.update(wandb_metrics(rapport))
            if selfplay_diagnostics > 0:
                memoire["apres_evaluation"] = mesurer()
        else:
            print("  Évaluation du banc ignorée.")

        if selfplay_diagnostics > 0:
            # Les memes mesures restent dans W&B ; la console ne garde que le
            # rapport de phases et de compteurs, pas les dix lignes memoire.
            journal.update(metriques_memoire(memoire))
        module_wandb.log(journal, step=global_step)
        module_wandb.log({}, commit=True)

    module_wandb.finish()


# ============================================================
#                         CLI
# ============================================================

def analyser_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Boucle self-play LapZero")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--games", type=int, default=512,
                        help="parties par iteration")
    parser.add_argument("--concurrent", type=int, default=256,
                        help="places du pool self-play")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--buffer", type=int, default=750_000)
    parser.add_argument("--sampling-ratio", type=float, default=14.0)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--selfplay-diagnostics", type=int, default=0,
                        choices=(0, 1, 2),
                        help="1 phases, compteurs de recherche et memoire ; "
                             "2 ajoute le detail par worker")
    parser.add_argument("--fast-dirichlet-noise",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="bruit de Dirichlet sur les coups rapides ; "
                             "--no-fast-dirichlet-noise le reserve aux coups "
                             "lents")
    parser.add_argument("--eval-every", type=int, default=4,
                        help="banc de positions tous les N tours ; 0 desactive")
    parser.add_argument("--position-bench", default=None,
                        help="dossier du banc ; par defaut data/position_bench/v1")
    parser.add_argument("--eval-search-workers", type=int, default=8)
    parser.add_argument("--eval-target-seconds", type=float, default=300.0)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt de reprise")
    parser.add_argument("--wandb-mode",
                        choices=["offline", "disabled", "online"],
                        default=None)
    # Migration : ces drapeaux n'ont plus d'effet dans la boucle.
    parser.add_argument("--stockfish", default=None,
                        help="OBSOLETE : l'ancrage se lance manuellement")
    parser.add_argument("--stockfish-elo", type=int, default=None,
                        help="OBSOLETE : l'ancrage se lance manuellement")
    return parser.parse_args(argv)


def principal(argv=None):
    args = analyser_arguments(argv)
    if args.stockfish is not None or args.stockfish_elo is not None:
        raise SystemExit(
            "l'ancrage Stockfish n'est plus lance par la boucle de self-play ; "
            "utiliser stockfish_player.py manuellement, ou retirer --stockfish "
            "et --stockfish-elo")
    try:
        pipeline(
            num_iterations=args.iterations,
            games_per_iter=args.games,
            concurrent_games=args.concurrent,
            slow_sims=700,
            fast_sims=100,
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
            wandb_mode=args.wandb_mode,
        )
    except KeyboardInterrupt:
        print("\n[Interruption] Entraînement stoppé manuellement.")
        print("Synchronisation des dernières métriques avec WandB en cours...")
        wandb.finish()
        print("Arrêt propre terminé.")


if __name__ == "__main__":
    principal()
