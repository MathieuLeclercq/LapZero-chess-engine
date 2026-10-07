"""Harnais de mesure et de verification de la recherche MCTS.

Reference reproductible du debit, et filet contre la corruption silencieuse
d'arbre, avant le batching avec virtual loss.

Voir docs/superpowers/specs/2026-09-11-search-bench-design.md
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import os
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from core.data_paths import ONNX_DIR

RACINE_PYTHON = _runtime.PYTHON_ROOT
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))
os.add_dll_directory(str(RACINE_PYTHON))

import chess_engine

# Jamais le defaut : une TTEntry pese 1040 octets et le defaut de MCTS est
# 2 097 143 entrees, soit 2,03 Gio par instance.
TAILLE_TT = 8192

# Trois positions de reference. Un debit mesure sur une seule position ne
# represente rien : sur une position de mat, 381 simulations sur 400 s'arretent
# sur un noeud terminal sans appeler le reseau, ce qui affiche 5624 sims/s.
# Les trois positions ci-dessous ont ete verifiees exemptes de ce biais, zero
# noeud terminal et 401 inferences pour 400 simulations. Les deux dernieres
# viennent des positions de reference du perft, donc deja verifiees legales.
POSITIONS = (
    ("ouverture",
     "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("milieu",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("finale",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)


@dataclass(frozen=True)
class Mesure:
    position: str
    chemin: str
    simulations: int
    batch_size: int
    duree_s: float
    nn_calls: int
    nn_batches: int
    tt_hits: int
    tt_misses: int
    terminal_hits: int
    cache_history_depth: int
    tt_position_matches: int
    tt_rule50_rejects: int
    tt_context_rejects: int
    tt_history_rejects: int
    leaf_collisions: int = 0
    repetition: int = 0
    worker_count: int = 1
    pool_etat: str = "froid"
    timing: dict[str, int] | None = None


TIMING_FIELDS = (
    "wall_ns", "selection_ns", "tensor_key_ns", "tt_probe_store_ns",
    "tt_wait_ns", "board_copy_ns", "batch_assembly_ns", "evaluator_ns",
    "expansion_ns", "backup_ns", "worker_wait_ns",
)

TIMING_LABELS = (
    ("wall_ns", "total"),
    ("selection_ns", "selection"),
    ("tensor_key_ns", "tenseur et cle"),
    ("tt_probe_store_ns", "lecture/ecriture TT"),
    ("tt_wait_ns", "attente TT"),
    ("board_copy_ns", "copie plateau"),
    ("batch_assembly_ns", "assemblage batch"),
    ("evaluator_ns", "evaluateur"),
    ("expansion_ns", "expansion"),
    ("backup_ns", "backup"),
    ("worker_wait_ns", "attente workers"),
)


def extraire_timing(mcts, enabled: bool) -> dict[str, int] | None:
    if not enabled:
        return None
    timing = mcts.get_last_timing()
    return {name: int(getattr(timing, name)) for name in TIMING_FIELDS}


def sims_par_seconde(m: Mesure) -> float:
    return m.simulations / m.duree_s if m.duree_s > 0 else 0.0


def inferences_par_seconde(m: Mesure) -> float:
    """Positions passees au reseau par seconde.

    Une simulation qui s'arrete sur un noeud terminal ou sur un succes de table
    ne coute aucune evaluation. Ce n'est pas le nombre d'appels batch, expose
    separement par appels_batch_par_seconde.
    """
    return m.nn_calls / m.duree_s if m.duree_s > 0 else 0.0


def appels_batch_par_seconde(m: Mesure) -> float:
    return m.nn_batches / m.duree_s if m.duree_s > 0 else 0.0


def remplissage_moyen(m: Mesure) -> float:
    return m.nn_calls / m.nn_batches if m.nn_batches else 0.0


def taux_table(m: Mesure) -> float:
    total = m.tt_hits + m.tt_misses
    return m.tt_hits / total if total else 0.0


def taux_compteur_tt(m: Mesure, valeur: int) -> float:
    consultations = m.tt_hits + m.tt_misses
    return valeur / consultations if consultations else 0.0


def etiquette_profondeur(depth: int) -> str:
    return "legacy" if depth == -1 else f"h{depth}"


def charger_position(fen: str):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


@dataclass(frozen=True)
class ReglagesMCTS:
    """Reglages de recherche appliques a un MCTS, defauts compris."""
    fixed_batch: bool = False
    virtual_loss: int = 1
    fpu_reduction: float = 0.30
    collision_attempts: int = 4
    timings: bool = False


def configurer_mcts(mcts, reglages: ReglagesMCTS) -> None:
    """Applique explicitement TOUS les reglages, y compris les defauts.

    Un MCTS reutilise garde sinon le lot fixe, le tuning ou le chronometrage
    d'une configuration precedente, et un balayage ne mesurerait plus ce que son
    rapport annonce. Les invariants et les mesures passent par ce meme point.
    """
    mcts.set_fixed_batch(reglages.fixed_batch)
    mcts.set_tuning(reglages.virtual_loss, reglages.fpu_reduction,
                    reglages.collision_attempts)
    mcts.set_timing_enabled(reglages.timings)


def mesurer_mcts_search(evaluateur, fen: str, nom: str,
                        simulations: int, c_puct: float = 1.4,
                        batch_size: int = 0,
                        cache_history_depth: int = 0,
                        tt_size: int = TAILLE_TT,
                        timings: bool = False,
                        fixed_batch: bool = False,
                        virtual_loss: int = 1,
                        fpu_reduction: float = 0.30,
                        collision_attempts: int = 4,
                        slices: int = 0,
                        repetition: int = 0,
                        worker_count: int = 1,
                        pool_etat: str = "froid",
                        mcts=None) -> Mesure:
    """Arbre neuf a chaque appel.

    Un MCTS fourni est reutilise tel quel pour mesurer un pool chaud : le
    coordinateur a deja remis l'arbre et la TT a froid hors chronometrage.
    """
    if mcts is None:
        mcts = chess_engine.MCTS(evaluateur, tt_size, cache_history_depth)
    configurer_mcts(mcts, ReglagesMCTS(
        fixed_batch=fixed_batch, virtual_loss=virtual_loss,
        fpu_reduction=fpu_reduction, collision_attempts=collision_attempts,
        timings=timings))
    board = charger_position(fen)
    mcts.reset_counters()

    debut = time.perf_counter()
    if slices:
        fait = 0
        while fait < simulations:
            bloc = min(slices, simulations - fait)
            mcts.mcts_search(board, bloc, c_puct, False, batch_size,
                             worker_count)
            fait += bloc
    else:
        mcts.mcts_search(board, simulations, c_puct, False, batch_size,
                         worker_count)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(
        position=nom, chemin="mcts_search", simulations=simulations,
        batch_size=batch_size, duree_s=duree, nn_calls=c.nn_calls,
        nn_batches=c.nn_batches, tt_hits=c.tt_hits, tt_misses=c.tt_misses,
        terminal_hits=c.terminal_hits,
        cache_history_depth=cache_history_depth,
        tt_position_matches=c.tt_position_matches,
        tt_rule50_rejects=c.tt_rule50_rejects,
        tt_context_rejects=c.tt_context_rejects,
        tt_history_rejects=c.tt_history_rejects,
        leaf_collisions=getattr(c, "leaf_collisions", 0),
        repetition=repetition, worker_count=worker_count,
        pool_etat=pool_etat,
        timing=extraire_timing(mcts, timings))


def mesurer_step_analysis(evaluateur, fen: str, nom: str,
                          simulations: int, c_puct: float = 1.4,
                          batch_size: int = 0,
                          cache_history_depth: int = 0,
                          tt_size: int = TAILLE_TT,
                          timings: bool = False,
                          fixed_batch: bool = False,
                          virtual_loss: int = 1,
                          fpu_reduction: float = 0.30,
                          collision_attempts: int = 4,
                          slices: int = 0,
                          repetition: int = 0,
                          worker_count: int = 1,
                          pool_etat: str = "froid",
                          mcts=None) -> Mesure:
    """Le chemin reel du bot : arbre d'analyse reutilise entre les coups."""
    if mcts is None:
        mcts = chess_engine.MCTS(evaluateur, tt_size, cache_history_depth)
    configurer_mcts(mcts, ReglagesMCTS(
        fixed_batch=fixed_batch, virtual_loss=virtual_loss,
        fpu_reduction=fpu_reduction, collision_attempts=collision_attempts,
        timings=timings))
    board = charger_position(fen)
    mcts.reset_analysis()
    mcts.reset_counters()

    debut = time.perf_counter()
    if slices:
        fait = 0
        while fait < simulations:
            bloc = min(slices, simulations - fait)
            mcts.step_analysis(board, bloc, c_puct, batch_size, worker_count)
            fait += bloc
    else:
        mcts.step_analysis(board, simulations, c_puct, batch_size,
                           worker_count)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(
        position=nom, chemin="step_analysis", simulations=simulations,
        batch_size=batch_size, duree_s=duree, nn_calls=c.nn_calls,
        nn_batches=c.nn_batches, tt_hits=c.tt_hits, tt_misses=c.tt_misses,
        terminal_hits=c.terminal_hits,
        cache_history_depth=cache_history_depth,
        tt_position_matches=c.tt_position_matches,
        tt_rule50_rejects=c.tt_rule50_rejects,
        tt_context_rejects=c.tt_context_rejects,
        tt_history_rejects=c.tt_history_rejects,
        leaf_collisions=getattr(c, "leaf_collisions", 0),
        repetition=repetition, worker_count=worker_count,
        pool_etat=pool_etat,
        timing=extraire_timing(mcts, timings))


def agreger(mesures: list) -> dict:
    """Groupe par position, chemin, batch, politique de cache, workers et etat
    du pool.

    L'etendue est indispensable : sans elle, un gain de 5 pour cent serait
    indistinguable du bruit de mesure. Separer froid et chaud isole le cout de
    creation des threads du regime etabli.
    """
    groupes: dict = {}
    for m in mesures:
        groupes.setdefault((m.position, m.chemin, m.batch_size,
                            m.cache_history_depth, m.worker_count,
                            m.pool_etat), []).append(m)

    resultat = {}
    for cle, lot in groupes.items():
        debits = [sims_par_seconde(m) for m in lot]
        inferences = [inferences_par_seconde(m) for m in lot]
        appels_batch = [appels_batch_par_seconde(m) for m in lot]
        resultat[cle] = {
            "passages": len(lot),
            "simulations": lot[0].simulations,
            "sims_par_seconde_median": statistics.median(debits),
            "sims_par_seconde_min": min(debits),
            "sims_par_seconde_max": max(debits),
            "inferences_par_seconde_median": statistics.median(inferences),
            "appels_batch_par_seconde_median": statistics.median(appels_batch),
            "remplissage_moyen_median": statistics.median(
                remplissage_moyen(m) for m in lot),
            "taux_table_median": statistics.median(taux_table(m) for m in lot),
            "tt_misses_median": statistics.median(m.tt_misses for m in lot),
            "terminal_hits_median": statistics.median(
                m.terminal_hits for m in lot),
            "tt_position_matches_median": statistics.median(
                m.tt_position_matches for m in lot),
            "tt_rule50_rejects_median": statistics.median(
                m.tt_rule50_rejects for m in lot),
            "tt_context_rejects_median": statistics.median(
                m.tt_context_rejects for m in lot),
            "tt_history_rejects_median": statistics.median(
                m.tt_history_rejects for m in lot),
            "tt_position_match_rate_median": statistics.median(
                taux_compteur_tt(m, m.tt_position_matches) for m in lot),
            "tt_rule50_reject_rate_median": statistics.median(
                taux_compteur_tt(m, m.tt_rule50_rejects) for m in lot),
            "tt_context_reject_rate_median": statistics.median(
                taux_compteur_tt(m, m.tt_context_rejects) for m in lot),
            "tt_history_reject_rate_median": statistics.median(
                taux_compteur_tt(m, m.tt_history_rejects) for m in lot),
        }
        timings = [m.timing for m in lot if m.timing is not None]
        if timings:
            for name in TIMING_FIELDS:
                resultat[cle][f"{name}_median_ms"] = statistics.median(
                    timing[name] for timing in timings) / 1_000_000.0
    return resultat


_EN_TETE = (
    "| Position | Chemin | TT | batch | workers | pool | passages | sims/s (med) "
    "| sims/s (min a max) | positions reseau/s | appels batch/s "
    "| remplissage | taux table | match position | 50 coups | contexte "
    "| historique |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def format_report(agr: dict, contexte: dict, invariants: list) -> str:
    """Rapport markdown.

    Le mot « noeuds par seconde » est proscrit : le perft mesure la generation
    de coups a plus de 1,6 million de noeuds par seconde, la recherche tourne a
    environ 295 simulations par seconde.
    """
    lignes = [
        "# Banc de recherche : resultats",
        "",
        f"Modele : `{contexte['modele']}`, iteration {contexte['iteration']}, "
        f"global_step {contexte['global_step']}",
        f"Protocole : {contexte['passages']} passages, "
        f"{contexte['simulations']} simulations, c_puct {contexte['c_puct']}, "
        f"{contexte.get('accelerateur', 'CPU')}, un seul processus, "
        f"batches demandes {contexte.get('batch_sizes', [0])}, "
        f"workers {contexte.get('worker_counts', [1])}, "
        + ("pool reutilise entre mesures"
           if contexte.get("warmup_pool") else "MCTS neuf par mesure")
        + ", "
        f"politiques TT {contexte.get('cache_history_depths', [0])}",
        "",
        "Trois grandeurs distinctes. Les **simulations par seconde** mesurent le",
        "debit de la recherche. Les **positions reseau par seconde** comptent les",
        "positions effectivement evaluees, les **appels batch par seconde** les",
        "lancements physiques et leur ratio est le **remplissage**. Le **taux de",
        "table** est la part des consultations reussies.",
        "",
        "## Debit",
        "",
        _EN_TETE,
    ]

    for cle in sorted(agr):
        position, chemin, batch_size, cache_history_depth, worker_count, \
            pool_etat = cle
        a = agr[cle]
        def compteur_et_taux(compteur: str, taux: str) -> str:
            return f"{a[compteur]:g} ({100 * a[taux]:.1f} %)"
        lignes.append(
            f"| {position} | {chemin} | {etiquette_profondeur(cache_history_depth)} "
            f"| {batch_size} | {worker_count} | {pool_etat} | {a['passages']} "
            f"| {a['sims_par_seconde_median']:.1f} "
            f"| {a['sims_par_seconde_min']:.1f} a {a['sims_par_seconde_max']:.1f} "
            f"| {a['inferences_par_seconde_median']:.1f} "
            f"| {a['appels_batch_par_seconde_median']:.1f} "
            f"| {a['remplissage_moyen_median']:.1f} "
            f"| {100 * a['taux_table_median']:.1f} % "
            f"| {compteur_et_taux('tt_position_matches_median', 'tt_position_match_rate_median')} "
            f"| {compteur_et_taux('tt_rule50_rejects_median', 'tt_rule50_reject_rate_median')} "
            f"| {compteur_et_taux('tt_context_rejects_median', 'tt_context_reject_rate_median')} "
            f"| {compteur_et_taux('tt_history_rejects_median', 'tt_history_reject_rate_median')} |"
        )

    if any("wall_ns_median_ms" in a for a in agr.values()):
        lignes += [
            "",
            "## Chronometrages",
            "",
            "Durees medianes en millisecondes. Le temps total est le temps mur "
            "de l'appel : les phases sont cumulees par thread, peuvent se "
            "chevaucher et ne s'additionnent jamais en temps mur.",
            "",
            "| Position | Chemin | TT | batch | workers | pool | "
            + " | ".join(label for _name, label in TIMING_LABELS) + " |",
            "|---|---|---|---|---|---|" + "---|" * len(TIMING_LABELS),
        ]
        for cle in sorted(agr):
            position, chemin, batch_size, cache_history_depth, worker_count, \
                pool_etat = cle
            a = agr[cle]
            if "wall_ns_median_ms" not in a:
                continue
            valeurs = " | ".join(
                f"{a[f'{name}_median_ms']:.3f}"
                for name, _label in TIMING_LABELS)
            lignes.append(
                f"| {position} | {chemin} "
                f"| {etiquette_profondeur(cache_history_depth)} "
                f"| {batch_size} | {worker_count} | {pool_etat} "
                f"| {valeurs} |")

    lignes += ["", "## Invariants d'arbre", ""]
    if not invariants:
        lignes.append("Non verifies lors de ce passage.")
    else:
        lignes += [
            "| Position | TT | batch | workers | noeuds | profondeur max "
            "| violations | en vol | Pending |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        total = 0
        for row in invariants:
            position, depth, batch_size, worker_count = row[:4]
            nodes, max_depth, violations = row[4:7]
            en_vol = row[8] if len(row) > 8 else 0
            pending = row[9] if len(row) > 9 else 0
            total += violations
            lignes.append(
                f"| {position} | {etiquette_profondeur(depth)} "
                f"| {batch_size} | {worker_count} "
                f"| {nodes} | {max_depth} | {violations} "
                f"| {en_vol} | {pending} |")
        lignes.append("")
        if total == 0:
            lignes.append("Aucune violation.")
        else:
            lignes.append(f"**{total} violations.** Premiers messages :")
            lignes.append("")
            for row in invariants:
                position, depth, batch_size, worker_count = row[:4]
                for message in row[7]:
                    lignes.append(
                        f"- `{position}`, {etiquette_profondeur(depth)}, "
                        f"batch {batch_size}, workers {worker_count} : {message}")

    return "\n".join(lignes) + "\n"


def main() -> int:
    import argparse

    import evaluation.puzzle_bench as puzzle_bench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="checkpoint .pt ou modele .onnx")
    parser.add_argument("--dossier-onnx", type=Path,
                        default=ONNX_DIR)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--passages", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=1,
                        help="mesures successives par passage et configuration")
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[0],
                        help="tailles de batch a balayer, 0 designe la boucle "
                             "sequentielle conservee")
    parser.add_argument("--cache-history-depths", type=int, nargs="+",
                        default=[0],
                        help="politiques TT a balayer : -1 pour legacy, "
                             "0 a 7 pour h0 a h7")
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1],
                        help="workers CPU de chaque recherche, 1 conserve le "
                             "noyau mono-worker")
    parser.add_argument("--warmup-pool", action="store_true",
                        help="reutilise le MCTS entre les repetitions pour "
                             "isoler le premier appel (creation des threads) des "
                             "mesures de regime etabli")
    parser.add_argument("--gpu", action="store_true",
                        help="utilise le provider CUDA du moteur C++")
    parser.add_argument("--invariants", action="store_true",
                        help="verifie les invariants d'arbre, sans mesurer le debit")
    parser.add_argument("--timings", action="store_true",
                        help="active les chronometrages internes par phase")
    parser.add_argument("--fixed-batch", action="store_true",
                        help="padde les lots d evaluation a une forme fixe")
    parser.add_argument("--virtual-loss", type=int, default=1,
                        help="unites de n_in_flight par descente")
    parser.add_argument("--fpu", type=float, default=0.30,
                        help="coefficient du terme FPU")
    parser.add_argument("--collision-attempts", type=int, default=4,
                        help="facteur du budget de tentatives de collecte")
    parser.add_argument("--slices", type=int, default=0,
                        help="decoupe chaque recherche en tranches de N "
                             "simulations, 0 pour un appel unique")
    parser.add_argument("--tt-size", type=int, default=TAILLE_TT)
    parser.add_argument("--out-json", type=Path, default=None,
                        help="mesures individuelles et contexte en JSON")
    parser.add_argument("--out-rapport", type=Path, default=None)
    args = parser.parse_args()

    if any(taille < 0 for taille in args.batch_sizes):
        parser.error("les tailles de batch doivent etre positives ou nulles")
    if any(depth < -1 or depth > 7 for depth in args.cache_history_depths):
        parser.error("les profondeurs de cache doivent etre comprises entre -1 et 7")
    if args.passages <= 0 or args.repetitions <= 0:
        parser.error("passages et repetitions doivent etre positifs")
    if args.tt_size <= 0:
        parser.error("tt-size doit etre positif")
    if args.virtual_loss < 1:
        parser.error("--virtual-loss doit etre au moins 1")
    if args.fpu < 0.0:
        parser.error("--fpu doit etre positif ou nul")
    if args.collision_attempts < 1:
        parser.error("--collision-attempts doit etre au moins 1")
    if args.slices < 0:
        parser.error("--slices doit etre positif ou nul")
    if any(count < 1 for count in args.worker_counts):
        parser.error("les workers de recherche doivent etre au moins 1")
    if (any(count > 1 for count in args.worker_counts)
            and any(taille == 0 for taille in args.batch_sizes)):
        parser.error("un worker_count superieur a 1 exige une taille de "
                     "batch positive")

    onnx, meta = puzzle_bench.resoudre_modele(args.model, args.dossier_onnx)
    if not Path(onnx).exists():
        print(f"modele ONNX introuvable : {onnx}", file=sys.stderr)
        return 2

    if args.gpu:
        # Les DLL CUDA de torch rendent celles du provider ONNX visibles au
        # processus avant la construction de l'evaluateur C++.
        import torch  # noqa: F401
    evaluateur = chess_engine.ONNXEvaluator(str(onnx), args.gpu)

    # Rodage : la premiere inference initialise la session.
    chauffe = chess_engine.MCTS(
        evaluateur, args.tt_size, args.cache_history_depths[0])
    chauffe.mcts_search(charger_position(POSITIONS[0][1]), 8, args.c_puct,
                        False, args.batch_sizes[0])

    invariants = []
    mesures = []
    pools: dict = {}

    def mcts_du_pool(chemin, profondeur, batch_size, worker_count):
        """Objet MCTS reutilise et son etat de pool, froid ou chaud.

        Un pool chaud est efface hors chronometrage : l'arbre d'analyse et la
        TT repartent a zero, mais les threads du SearchExecutor restent en
        place. C'est ce qui separe le cout de creation du regime etabli.
        """
        cle = (chemin, profondeur, batch_size, worker_count)
        if cle in pools:
            pools[cle].clear_evaluation_cache()
            return pools[cle], "chaud"
        mcts = chess_engine.MCTS(evaluateur, args.tt_size, profondeur)
        pools[cle] = mcts
        return mcts, "froid"

    if args.invariants:
        # inspect_tree n'est jamais appele dans la boucle de mesure de debit :
        # son cout croit avec la taille de l'arbre et fausserait la mesure
        # qu'il protege. Les deux jambes sont donc exclusives.
        reglages = ReglagesMCTS(
            fixed_batch=args.fixed_batch, virtual_loss=args.virtual_loss,
            fpu_reduction=args.fpu,
            collision_attempts=args.collision_attempts,
            timings=args.timings)
        taille_tranche = (args.slices if args.slices > 0
                          else max(args.simulations, 1))
        for depth in args.cache_history_depths:
            for batch_size in args.batch_sizes:
                for worker_count in args.worker_counts:
                    for nom, fen in POSITIONS:
                        mcts = chess_engine.MCTS(
                            evaluateur, args.tt_size, depth)
                        configurer_mcts(mcts, reglages)
                        board = charger_position(fen)
                        mcts.reset_analysis()
                        # Une tranche peut reveler un etat transitoire que la
                        # suivante masquerait : on inspecte au repos apres
                        # chacune, et la violation maximale est retenue.
                        rapports = []
                        fait = 0
                        while True:
                            bloc = min(taille_tranche,
                                       args.simulations - fait)
                            mcts.step_analysis(board, bloc, args.c_puct,
                                               batch_size, worker_count)
                            fait += bloc
                            rapports.append(mcts.inspect_tree())
                            if fait >= args.simulations:
                                break
                        dernier = rapports[-1]
                        violations = max(r.violations for r in rapports)
                        en_vol = max(r.en_vol for r in rapports)
                        pending = max(r.pending for r in rapports)
                        messages = next(
                            (list(r.messages) for r in rapports
                             if r.violations == violations and violations > 0),
                            [])
                        invariants.append((
                            nom, depth, batch_size, worker_count,
                            dernier.nodes, dernier.max_depth, violations,
                            messages, en_vol, pending, len(rapports)))
                        etat = ("OK" if violations == 0 else
                                f"{violations} VIOLATIONS")
                        print(f"  {nom:10} {etiquette_profondeur(depth):6} "
                              f"batch {batch_size:2} workers {worker_count} : "
                              f"{dernier.nodes} noeuds, "
                              f"profondeur {dernier.max_depth}, {etat}, "
                              f"en vol {en_vol}, "
                              f"Pending {pending}, "
                              f"{len(rapports)} controle(s)")
    else:
        for passage in range(args.passages):
            # L'ordre des configurations tourne d'un passage a l'autre : derive
            # thermique et horloge GPU ne doivent pas se confondre avec l'effet
            # du nombre de workers.
            decalage = passage % len(args.worker_counts)
            ordre_workers = (args.worker_counts[decalage:]
                             + args.worker_counts[:decalage])
            for repetition_locale in range(args.repetitions):
                repetition = passage * args.repetitions + repetition_locale
                for depth in args.cache_history_depths:
                    for batch_size in args.batch_sizes:
                        for worker_count in ordre_workers:
                            for nom, fen in POSITIONS:
                                if args.warmup_pool:
                                    pool, pool_etat = mcts_du_pool(
                                        "mcts_search", depth, batch_size,
                                        worker_count)
                                else:
                                    pool, pool_etat = None, "froid"
                                mesures.append(mesurer_mcts_search(
                                    evaluateur, fen, nom, args.simulations,
                                    args.c_puct, batch_size, depth,
                                    tt_size=args.tt_size,
                                    timings=args.timings,
                                    fixed_batch=args.fixed_batch,
                                    virtual_loss=args.virtual_loss,
                                    fpu_reduction=args.fpu,
                                    collision_attempts=args.collision_attempts,
                                    slices=args.slices,
                                    repetition=repetition,
                                    worker_count=worker_count,
                                    pool_etat=pool_etat, mcts=pool))
                                if args.warmup_pool:
                                    pool, pool_etat = mcts_du_pool(
                                        "step_analysis", depth, batch_size,
                                        worker_count)
                                else:
                                    pool, pool_etat = None, "froid"
                                mesures.append(mesurer_step_analysis(
                                    evaluateur, fen, nom, args.simulations,
                                    args.c_puct, batch_size, depth,
                                    tt_size=args.tt_size,
                                    timings=args.timings,
                                    fixed_batch=args.fixed_batch,
                                    virtual_loss=args.virtual_loss,
                                    fpu_reduction=args.fpu,
                                    collision_attempts=args.collision_attempts,
                                    slices=args.slices,
                                    repetition=repetition,
                                    worker_count=worker_count,
                                    pool_etat=pool_etat, mcts=pool))
            print(f"  passage {passage + 1}/{args.passages}", flush=True)

    agr = agreger(mesures)
    for cle in sorted(agr):
        a = agr[cle]
        print(f"  {cle[0]:10} {cle[1]:14} "
              f"{etiquette_profondeur(cle[3]):6} batch {cle[2]:2} "
              f"workers {cle[4]} {cle[5]:5} : "
              f"{a['sims_par_seconde_median']:7.1f} sims/s, "
              f"{a['inferences_par_seconde_median']:7.1f} positions/s, "
              f"remplissage {a['remplissage_moyen_median']:.1f}, "
              f"table {100 * a['taux_table_median']:.1f} %")

    contexte = {
        "modele": Path(onnx).name,
        "iteration": meta.get("iteration"),
        "global_step": meta.get("global_step"),
        "passages": (args.passages * args.repetitions
                     if not args.invariants else 1),
        "repetitions": args.repetitions,
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "accelerateur": "GPU" if args.gpu else "CPU",
        "batch_sizes": args.batch_sizes,
        "cache_history_depths": args.cache_history_depths,
        "worker_counts": args.worker_counts,
        "warmup_pool": args.warmup_pool,
        "tt_size": args.tt_size,
        "timings": args.timings,
        "fixed_batch": args.fixed_batch,
        "virtual_loss": args.virtual_loss,
        "fpu_reduction": args.fpu,
        "collision_attempts": args.collision_attempts,
        "slices": args.slices,
    }

    sortie = args.out_rapport or Path(
        f"../docs/superpowers/specs/{time.strftime('%Y-%m-%d')}"
        "-search-bench-resultats.md")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(format_report(agr, contexte, invariants), encoding="utf-8")
    print(f"\nRapport : {sortie}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        donnees = {
            "contexte": contexte,
            "mesures": [asdict(m) for m in mesures],
        }
        if args.invariants:
            # Le mode invariants ne produit aucune mesure de debit : le JSON
            # doit porter les resultats d'invariants et les reglages effectifs,
            # pas une liste de mesures vide.
            donnees["invariants"] = [
                {
                    "position": row[0],
                    "cache_history_depth": row[1],
                    "batch_size": row[2],
                    "worker_count": row[3],
                    "nodes": row[4],
                    "max_depth": row[5],
                    "violations": row[6],
                    "messages": list(row[7]),
                    "en_vol": row[8],
                    "pending": row[9],
                    "tranches": row[10],
                }
                for row in invariants]
        args.out_json.write_text(
            json.dumps(donnees, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Mesures : {args.out_json}")

    total_violations = sum(row[6] for row in invariants)
    if total_violations:
        print(f"{total_violations} violations d'invariants", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
