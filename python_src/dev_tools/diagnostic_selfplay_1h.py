"""Compare iter436 au dernier ONNX, avec une limite de 58 minutes.

Depuis python_src : python dev_tools/diagnostic_selfplay_1h.py
Les mesures et le rapport vont dans out/diagnostic_selfplay_1h/<date>/.
Le superviseur reste independant de torch et du moteur C++.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ABBA = ("old", "new", "new", "old")
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def resolve_models(repo, old_model, new_model):
    candidates = []
    for folder in (repo / "data" / "checkpoints" / "onnx",):
        if folder.is_dir():
            for path in folder.glob("*.onnx"):
                match = re.search(r"iter(\d+)(?:_|\.)", path.name)
                if match:
                    candidates.append((int(match[1]), path.stat().st_mtime_ns, path))
    selected = []
    for explicit, label, iteration in ((old_model, "old-model", 436),
                                        (new_model, "new-model", None)):
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() != ".onnx":
                raise ValueError(f"--{label} doit designer un fichier .onnx existant : {path}")
        else:
            matches = [item for item in candidates if iteration is None or item[0] == iteration]
            if not matches:
                raise ValueError(f"Checkpoint introuvable ; preciser --{label} chemin.onnx")
            path = max(matches)[2].resolve()
        selected.append(path)
    if selected[0] == selected[1]:
        raise ValueError("Les deux checkpoints doivent etre differents")
    return tuple(selected)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_positions(puzzles):
    positions = [
        {"name": "depart", "fen": START_FEN, "moves": []},
        {"name": "italienne", "fen": START_FEN,
         "moves": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6".split()},
        {"name": "gambit_dame", "fen": START_FEN,
         "moves": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8".split()},
        {"name": "finale_tours_fen", "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "moves": []},
        {"name": "finale_pions_fen", "fen": "8/5pk1/6p1/3pP3/3P1P2/6K1/8/8 w - - 0 40", "moves": []},
    ]
    with open(puzzles, encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0] and fields[1]:
                positions.append({"name": f"puzzle_ligne_{index + 1}",
                                  "fen": fields[0], "moves": fields[1].split()})
            if len(positions) == 8:
                break
    return positions


def memory_snapshot(process=None):
    import psutil
    process = process or psutil.Process()
    info = process.memory_info()
    return {"rss_bytes": info.rss, "vms_bytes": info.vms,
            "private_bytes": getattr(info, "private", None),
            "threads": process.num_threads(),
            "handles": process.num_handles() if hasattr(process, "num_handles") else None}


def emit(path, kind, **values):
    row = {"kind": kind, "timestamp": time.time(), "pid": os.getpid(), **values}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_events(path):
    if not Path(path).is_file():
        return []
    events = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def monitor_process(pid, prefix, stop, interval):
    import psutil
    columns = ["timestamp", "elapsed_s", "pid", "launcher_pid", "rss_bytes", "private_bytes", "vms_bytes",
               "threads", "handles", "process_cpu_percent", "system_cpu_percent",
               "cpu_per_core_percent", "cpu_freq_reported_mhz", "available_ram_bytes"]
    try:
        launcher = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    sampled = {}
    psutil.cpu_percent(percpu=True)
    started = time.monotonic()
    nvidia = shutil.which("nvidia-smi")
    next_gpu = started
    with open(str(prefix) + ".telemetry.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        stream.flush()
        while not stop.wait(interval):
            try:
                candidates = [launcher] + launcher.children(recursive=True)
                live = []
                for candidate in candidates:
                    try:
                        live.append((candidate.memory_info().rss, candidate.pid))
                    except psutil.NoSuchProcess:
                        continue
                if not live:
                    break
                selected_pid = max(live)[1]
                first_sample = selected_pid not in sampled
                process = sampled.setdefault(selected_pid, psutil.Process(selected_pid))
                row = memory_snapshot(process)
                per_core = psutil.cpu_percent(percpu=True)
                try:
                    frequency = psutil.cpu_freq()
                except (OSError, NotImplementedError):
                    frequency = None
                cpu = process.cpu_percent()
                row.update(timestamp=time.time(), elapsed_s=time.monotonic() - started,
                           pid=selected_pid, launcher_pid=pid,
                           process_cpu_percent=None if first_sample else cpu,
                           system_cpu_percent=statistics.mean(per_core),
                           cpu_per_core_percent=json.dumps(per_core),
                           cpu_freq_reported_mhz=frequency.current if frequency else None,
                           available_ram_bytes=psutil.virtual_memory().available)
                writer.writerow(row)
                stream.flush()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if nvidia and time.monotonic() >= next_gpu:
                next_gpu = time.monotonic() + 10
                try:
                    result = subprocess.run(
                        [nvidia, "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        timeout=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    emit(str(prefix) + ".gpu.jsonl", "gpu_device", returncode=result.returncode,
                         columns="index,utilization_percent,memory_MiB,temp_C,power_W,sm_MHz",
                         rows=result.stdout.strip().splitlines())
                except (OSError, subprocess.TimeoutExpired):
                    pass


def stop_child(process):
    import psutil
    if process.poll() is None:
        try:
            root = psutil.Process(process.pid)
            root.suspend()
            descendants = root.children(recursive=True)
            for child in descendants:
                try:
                    child.suspend()
                except psutil.NoSuchProcess:
                    pass
            owned = list(reversed(descendants)) + [root]
            for child in owned:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(owned, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=1)
        except psutil.NoSuchProcess:
            pass
        process.wait(timeout=2)


def run_bounded(command, prefix, deadline, sample_seconds=2):
    started = time.monotonic()
    if started >= deadline:
        return {"status": "skipped", "returncode": None, "elapsed_s": 0}
    stop = threading.Event()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled")
    with open(str(prefix) + ".log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        monitor = threading.Thread(target=monitor_process,
                                   args=(process.pid, prefix, stop, sample_seconds), daemon=True)
        monitor.start()
        status = "complete"
        try:
            next_notice = started + 60
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    stop_child(process)
                    break
                if time.monotonic() >= next_notice:
                    print(f"[{prefix.name}] en cours, budget restant {remaining / 60:.1f} min", flush=True)
                    next_notice = time.monotonic() + 60
                time.sleep(min(0.1, remaining))
            if status == "complete" and process.returncode != 0:
                status = "error"
        finally:
            stop_child(process)
            stop.set()
            monitor.join(timeout=1.5)
    return {"status": status, "returncode": process.returncode,
            "elapsed_s": time.monotonic() - started, "pid": process.pid}


def fixed_comparisons(events):
    groups = {}
    for row in events:
        if row.get("kind") == "fixed_result":
            groups.setdefault(row["position"], {})[row["pass_index"]] = row
    comparisons = []
    for name, passes in groups.items():
        if set(passes) != {0, 1, 2, 3} or any(passes[i]["model"] != ABBA[i] for i in range(4)):
            continue
        row = {"position": name}
        for metric in ("elapsed_s", "nodes", "max_depth", "nn_calls", "tt_hits",
                       "visited_root_moves", "selection_ms", "evaluator_ms"):
            for model, indices in (("old", (0, 3)), ("new", (1, 2))):
                values = [passes[i].get(metric) for i in indices]
                row[f"{metric}_{model}"] = statistics.mean(values) if all(v is not None for v in values) else None
        row["time_ratio_new_old"] = row["elapsed_s_new"] / row["elapsed_s_old"]
        comparisons.append(row)
    return comparisons


def write_report(output, config, events, stages):
    events = sorted(events, key=lambda row: row.get("timestamp", 0))
    comparisons = fixed_comparisons(events)
    games = [row for row in events if row.get("kind") == "selfplay_result"]
    cleanups = [row for row in events if row.get("kind") == "memory" and row.get("point") == "after_cleanup"]
    summary = {"stages": stages, "fixed_comparisons": comparisons,
               "selfplay_results": games, "memory_after_cleanup": cleanups}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Diagnostic self-play", "", f"Ancien : `{config['models']['old']}`",
             f"Recent : `{config['models']['new']}`", "",
             f"Petites generations : {config['games']} parties / {config['concurrent']} places.", "",
             "Une hausse de memoire ne prouve pas une fuite. Un plateau peut venir des allocateurs.",
             "Un resultat negatif en une heure n'exclut pas une derive lente.", "",
             "## Execution", "", "| Phase | Etat | Duree (s) |", "|---|---|---:|"]
    for stage in stages:
        lines.append(f"| {stage['phase']} | {stage['status']} | {stage['elapsed_s']:.1f} |")
    errors = [row for row in events if row.get("kind") == "error"]
    for error in errors:
        lines += ["", f"Erreur : {error['message']}"]
    lines += ["", "Un timeout conserve les mesures precedentes ; le passage interrompu est incomplet.",
              "", "## Positions identiques", "",
              "Moyennes des deux passages par modele, uniquement pour les positions avec ABBA complet.",
              "TT et arbre neufs par recherche ; preparation, inspection et destruction hors chrono.",
              "Recherche mono-worker, sans bruit de Dirichlet. Elle ne mesure pas le debit du pool self-play.", "",
              "| Position | Recent/ancien temps | Noeuds ancien/recent | Profondeur ancien/recent |", "|---|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['position']} | {row['time_ratio_new_old']:.3f} | "
                     f"{row['nodes_old']:.0f} / {row['nodes_new']:.0f} | "
                     f"{row['max_depth_old']:.1f} / {row['max_depth_new']:.1f} |")
    expected = len(config.get("positions", []))
    if not comparisons or len(comparisons) < expected:
        lines += ["", "Comparaison de positions incomplete : aucune conclusion sur les positions manquantes."]
    lines += ["", "Details des appels reseau, hits TT et phases de recherche dans summary.json.",
              "", "## Generations reelles", "",
              "Ordre ABBA dans un meme processus, puis repetitions recentes si le budget le permet.",
              "Le controle fresh utilise un processus neuf. Les parties sont aleatoires, non appariees.", "",
              "| Phase / passage | Modele | Total (s) | Simulations/s C++ | Exemples/s total | ONNX (s) | Softmax (s) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in games:
        lines.append(f"| {row['phase']} / {row['pass_index'] + 1} | {row['model']} | "
                     f"{row['total_s']:.1f} | {row['sims_per_s']:.1f} | {row['saved_per_s']:.2f} | "
                     f"{row['onnx_s']:.1f} | {row['softmax_s']:.1f} |")
    abba = {row["pass_index"]: row for row in games if row["phase"] == "selfplay" and row["pass_index"] < 4}
    if set(abba) == {0, 1, 2, 3}:
        old = statistics.mean(abba[i]["sims_per_s"] for i in (0, 3))
        new = statistics.mean(abba[i]["sims_per_s"] for i in (1, 2))
        if old:
            lines += ["", f"Ratio recent/ancien du debit moyen : {new / old:.3f} (2 passages chacun, exploratoire)."]
    else:
        lines += ["", "Comparaison self-play ABBA incomplete : pas de ratio global ancien/recent."]
    lines += ["", "## Memoire apres nettoyage", "",
              "MB decimaux. Les comparaisons de derive se font au sein d'un meme PID.", "",
              "| Phase / passage | Modele | PID | Privee (MB) | Residente (MB) | Threads |", "|---|---|---:|---:|---:|---:|"]
    for row in cleanups:
        private = f"{row['private_bytes'] / 1e6:.1f}" if row.get("private_bytes") is not None else "n/d"
        lines.append(f"| {row.get('phase', '?')} / {row['pass_index'] + 1} | {row['model']} | "
                     f"{row.get('pid', '?')} | {private} | {row['rss_bytes'] / 1e6:.1f} | {row['threads']} |")
    lines += ["", "## Lecture des mesures", "",
              "- Recent plus lent sur les memes positions : regarder profondeur, noeuds, selection et appels reseau.",
              "- Memoire privee ou threads en hausse apres nettoyage : retention a localiser, pas preuve de fuite.",
              "- Amelioration dans fresh : compatible avec un effet du processus, a confirmer sur davantage de passages.",
              "- Temps ONNX inclut transferts, synchronisation et runtime ; ce n'est pas du calcul GPU pur.",
              "- CPU processus dans les CSV : pourcentage par intervalle, pouvant depasser 100 % sur plusieurs coeurs.",
              "- PID dans les CSV : processus du banc le plus resident, pour suivre le moteur derriere un lanceur Python.",
              "- Frequence CPU rapportee par le systeme : indicative, pas une mesure fiable de frequence effective.",
              "- Temperature CPU indisponible ici ; utiliser un journal de capteurs externe si necessaire.",
              "- GPU : mesures du peripherique entier via nvidia-smi, pas allocations du seul processus.",
              "- Le pool reduit, l'absence d'entrainement et l'absence d'optimiseur PyTorch resident limitent l'extrapolation.",
              "- Les positions FEN sans coups n'ont pas d'historique ; les puzzles conservent les coups rejoues.", ""]
    (output / "rapport.md").write_text("\n".join(lines), encoding="utf-8")


def load_runtime(repo, use_gpu):
    src = repo / "python_src"
    sys.path.insert(0, str(src))
    dll_handles = []
    if os.name == "nt":
        dll_handles.append(os.add_dll_directory(str(src)))
    import torch
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible dans torch ; activer l'environnement selfplay du fixe")
    import chess_engine
    if not hasattr(chess_engine, "generate_self_play_games_with_diagnostics"):
        raise RuntimeError("Ce binaire chess_engine ne contient pas le diagnostic self-play")
    return chess_engine, torch, dll_handles


def board_for(engine, position):
    board = engine.Chessboard()
    board.load_fen(position["fen"])
    for move in position["moves"]:
        if not board.move_piece_uci(move):
            raise ValueError(f"Historique invalide dans {position['name']} : {move}")
    if not board.has_any_legal_move():
        raise ValueError(f"Position terminale dans {position['name']}")
    return board


def warm_evaluator(engine, evaluator, batch):
    board = engine.Chessboard()
    board.set_startup_pieces()
    mcts = engine.MCTS(evaluator, 1024)
    mcts.step_analysis(board, max(8, batch), 1.4, batch, 1)


def fixed_worker(config, engine, torch, events, deadline):
    from core.lib import run_with_interrupt
    for position in config["positions"]:
        for pass_index, model in enumerate(ABBA):
            if time.monotonic() >= deadline:
                return
            emit(events, "fixed_start", position=position["name"], pass_index=pass_index, model=model)
            evaluator = engine.ONNXEvaluator(config["models"][model], config["device"] == "cuda")
            warm_evaluator(engine, evaluator, config["fixed_batch"])
            board = board_for(engine, position)
            mcts = engine.MCTS(evaluator, config["tt"])
            mcts.set_timing_enabled(True)
            started = time.perf_counter()
            run_with_interrupt(mcts.step_analysis, board, config["fixed_sims"], 1.4,
                               config["fixed_batch"], 1)
            elapsed = time.perf_counter() - started
            timing = mcts.get_last_timing()
            counters = mcts.get_counters()
            tree = mcts.inspect_tree()
            if tree.violations or tree.pending or tree.en_vol:
                raise RuntimeError(f"Invariants MCTS invalides pour {position['name']}")
            moves = mcts.get_analysis_results()
            prior = [float(move.prior) for move in moves if hasattr(move, "prior")]
            emit(events, "fixed_result", position=position["name"], pass_index=pass_index,
                 model=model, elapsed_s=elapsed, requested_sims=config["fixed_sims"],
                 nodes=int(tree.nodes), max_depth=int(tree.max_depth), root_visits=int(tree.root_visits),
                 nn_calls=int(counters.nn_calls), nn_batches=int(counters.nn_batches),
                 tt_hits=int(counters.tt_hits), tt_misses=int(counters.tt_misses),
                 visited_root_moves=sum(move.visits > 0 for move in moves),
                 prior_entropy=-sum(p * math.log(p) for p in prior if p > 0) if prior else None,
                 selection_ms=timing.selection_ns / 1e6, evaluator_ms=timing.evaluator_ns / 1e6,
                 expansion_ms=timing.expansion_ns / 1e6, actual_fen=board.to_fen())
            del mcts, board, evaluator, moves
            gc.collect()
            if config["device"] == "cuda":
                torch.cuda.empty_cache()


def one_generation(config, engine, torch, events, phase, pass_index, model):
    from core.lib import convert_game_results, run_with_interrupt
    from training.selfplay_diag import resumer_passage

    def snapshot(point):
        emit(events, "memory", phase=phase, pass_index=pass_index, model=model,
             point=point, **memory_snapshot())

    snapshot("before_evaluator")
    started = time.perf_counter()
    evaluator = engine.ONNXEvaluator(config["models"][model], config["device"] == "cuda")
    zones = {"creation_evaluateur": time.perf_counter() - started}
    snapshot("after_evaluator")
    emit(events, "selfplay_start", phase=phase, pass_index=pass_index, model=model)
    started = time.perf_counter()
    results, stats, timing = run_with_interrupt(
        engine.generate_self_play_games_with_diagnostics, evaluator, config["concurrent"],
        config["slow_sims"], config["fast_sims"], config["games"], 0.25,
        config["tt"], config["puzzles"], 1)
    zones["appel_cpp"] = time.perf_counter() - started
    snapshot("after_cpp")
    started = time.perf_counter()
    data, conclusions = convert_game_results(results)
    zones["conversion"] = time.perf_counter() - started
    count = len(data)
    if len(results) != config["games"] or stats.games_completed != config["games"]:
        raise RuntimeError("Quota de parties termine incoherent")
    snapshot("after_conversion")
    started = time.perf_counter()
    del data, results, evaluator
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    zones["nettoyage"] = time.perf_counter() - started
    snapshot("after_cleanup")
    total = sum(zones.values())
    generation_s = timing.generation_wall_ns / 1e9
    network_rows = timing.batch_rows + timing.unit_network_rows
    debit = {"coups_nouveaux_par_s": stats.new_plies / zones["appel_cpp"],
             "exemples_par_s": count / total, "parties_par_s": config["games"] / total,
             "evaluations_reseau_par_s": network_rows / zones["appel_cpp"]}
    report = resumer_passage(timing, zones, debit)
    emit(events, "selfplay_result", phase=phase, pass_index=pass_index, model=model,
         total_s=total, sims_per_s=timing.completed_sims / generation_s if generation_s else 0,
         saved_per_s=count / total, new_plies=int(stats.new_plies), replayed_plies=int(stats.replayed_plies),
         saved_positions=count, conclusions=conclusions,
         onnx_s=(timing.onnx_run_ns + timing.root_expansion_onnx_run_ns) / 1e9,
         softmax_s=(timing.onnx_softmax_ns + timing.root_expansion_onnx_softmax_ns) / 1e9,
         cpp_outside_generation_s=zones["appel_cpp"] - generation_s, report=report)
    return total


def worker(config_path, phase, deadline):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    events = Path(config["output"]) / f"{phase}.events.jsonl"
    try:
        engine, torch, dll_handles = load_runtime(Path(config["repo"]), config["device"] == "cuda")
        emit(events, "runtime", phase=phase, python=sys.version, torch=torch.__version__,
             torch_cuda=torch.version.cuda, binary=engine.__file__, binary_sha256=sha256(engine.__file__),
             gpu=torch.cuda.get_device_name(0) if config["device"] == "cuda" else None)
        if phase == "fixed":
            fixed_worker(config, engine, torch, events, deadline)
        else:
            evaluator = engine.ONNXEvaluator(config["models"]["new"], config["device"] == "cuda")
            warm_evaluator(engine, evaluator, config["fixed_batch"])
            del evaluator
            gc.collect()
            if config["device"] == "cuda":
                torch.cuda.empty_cache()
            emit(events, "memory", phase=phase, point="after_warmup", **memory_snapshot())
            order = ("new",) if phase == "fresh" else ABBA + ("new", "new")
            durations = []
            for pass_index, model in enumerate(order):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (durations and remaining < 1.2 * max(durations) + 2):
                    emit(events, "not_started", phase=phase, pass_index=pass_index,
                         model=model, reason="budget restant insuffisant")
                    break
                durations.append(one_generation(config, engine, torch, events, phase, pass_index, model))
        emit(events, "worker_done", phase=phase)
        return 0
    except Exception as error:
        emit(events, "error", phase=phase, message=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1


def training_processes():
    import psutil
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            arguments = process.info["cmdline"] or []
            if any(Path(token).name.lower() in ("run_selftrain.py", "train_self_play.py")
                   or token in ("run_selftrain", "train_self_play") for token in arguments):
                found.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--old-model", help="ONNX ancien ; defaut : iter436")
    parser.add_argument("--new-model", help="ONNX recent ; defaut : plus grande iteration trouvee")
    parser.add_argument("--puzzles", type=Path)
    parser.add_argument("--output", type=Path, help="Nouveau dossier de sortie (doit ne pas exister)")
    parser.add_argument("--minutes", type=float, default=58, help="Budget total, maximum 58 minutes")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--concurrent", type=int, default=32)
    parser.add_argument("--tt", type=int, default=4_000_000)
    parser.add_argument("--slow-sims", type=int, default=700)
    parser.add_argument("--fast-sims", type=int, default=100)
    parser.add_argument("--fixed-sims", type=int, default=700)
    parser.add_argument("--fixed-batch", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Afficher les chemins et parametres sans calcul")
    parser.add_argument("--worker", choices=("fixed", "selfplay", "fresh"), help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.config, args.worker, args.deadline)
    started = time.monotonic()
    if not math.isfinite(args.minutes) or not 0 < args.minutes <= 58:
        parser.error("--minutes doit etre compris entre 0 (exclu) et 58")
    if any(getattr(args, name) <= 0 for name in ("games", "concurrent", "tt", "slow_sims", "fast_sims", "fixed_sims", "fixed_batch")):
        parser.error("Les tailles et budgets de simulations doivent etre strictement positifs")
    try:
        import psutil
        repo = args.repo.resolve()
        old, new = resolve_models(repo, args.old_model, args.new_model)
        puzzles = (args.puzzles or repo / "training_data" / "puzzles_train.txt").resolve()
        if not puzzles.is_file():
            raise ValueError(f"Puzzles introuvables : {puzzles} ; preciser --puzzles")
        active = training_processes()
        if active:
            raise ValueError(f"Entrainement encore present (PID {active}). L'arreter proprement avant le banc.")
        output = (args.output or repo / "out" / "diagnostic_selfplay_1h" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
        config = {key: getattr(args, key) for key in ("minutes", "games", "concurrent", "tt", "slow_sims", "fast_sims", "fixed_sims", "fixed_batch", "device")}
        config.update(repo=str(repo), output=str(output), models={"old": str(old), "new": str(new)},
                      puzzles=str(puzzles), positions=fixed_positions(puzzles),
                      host=platform.node(), platform=platform.platform(), python=sys.version,
                      logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                      ram_bytes=psutil.virtual_memory().total,
                      randomness="random_device C++ non controlable ; self-play non apparie")
        print(f"Ancien : {old}\nRecent : {new}\n"
              f"Self-play : {args.games} parties / {args.concurrent} places, "
              f"{args.slow_sims}/{args.fast_sims} simulations, TT {args.tt:,}\n"
              f"Budget : {args.minutes:g} minutes. Sortie : {output}", flush=True)
        if args.dry_run:
            print("Verification des chemins terminee. Aucun calcul lance.")
            return 0
        config["hashes"] = {"old": sha256(old), "new": sha256(new), "puzzles": sha256(puzzles)}
        if config["hashes"]["old"] == config["hashes"]["new"]:
            raise ValueError("Les deux fichiers ONNX ont le meme contenu")
        output.mkdir(parents=True, exist_ok=False)
        config_path = output / "config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        stages = []
        deadline = started + args.minutes * 60
        schedule = [("fixed", started + args.minutes * 60 * 0.14),
                    ("selfplay", started + args.minutes * 60 * 0.83), ("fresh", deadline)]
        try:
            for phase, phase_deadline in schedule:
                print(f"[{phase}] debut. Mesures : {output}", flush=True)
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", phase,
                           "--config", str(config_path), "--deadline", str(phase_deadline)]
                result = run_bounded(command, output / phase, phase_deadline)
                stages.append({"phase": phase, **result})
                print(f"[{phase}] {result['status']} ({result['elapsed_s']:.1f} s)", flush=True)
                events = [row for path in sorted(output.glob("*.events.jsonl")) for row in read_events(path)]
                write_report(output, config, events, stages)
                if result["status"] == "error":
                    break
        except KeyboardInterrupt:
            stages.append({"phase": "interruption_utilisateur", "status": "interrupted", "elapsed_s": time.monotonic() - started})
        finally:
            events = [row for path in sorted(output.glob("*.events.jsonl")) for row in read_events(path)]
            write_report(output, config, events, stages)
        print(f"Rapport : {output / 'rapport.md'}", flush=True)
        return 1 if any(row["status"] in ("error", "interrupted") for row in stages) else 0
    except (ValueError, OSError, ImportError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
