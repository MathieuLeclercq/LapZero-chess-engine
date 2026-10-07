"""Rapport de diagnostic du self-play (chantier A0 de l'audit de debit).

Ce module ne connait ni torch ni chess_engine : il met en forme le rapport
d'un pourcentage, d'un debit et d'un passage. Le pilote qui lance la
generation vit dans dev_tools/selfplay_diagnostics.py ; la production
l'utilise seulement quand le diagnostic est active.

Trois debits distincts sont conserves, comme le demande l'audit :
- coups nouveaux par seconde, sans le rejeu des historiques de puzzles ;
- exemples conserves par seconde, au meme perimetre que la metrique de
  production ;
- evaluations reseau par seconde, compteur de diagnostic a ne pas maximiser
  isolement.

Les phases de premier niveau sont disjointes : leur somme et le residu
reconstituent le temps de generation. Les sous-durees ONNX ne doivent pas
etre ajoutees aux phases qui les contiennent.
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import hashlib
import os
import platform
import subprocess
import sys

PHASES = (
    "initial_slots",
    "move_management",
    "collection",
    "assembly_dispatch",
    "batch_evaluator",
    "batch_validation",
    "batch_consume",
    "batch_finalize",
)
PHASES_LISIBLES = (
    "departs initiaux",
    "gestion des coups",
    "collecte (descentes + tenseurs)",
    "assemblage et decision d'envoi",
    "appel a l'evaluateur",
    "validation des sorties",
    "consommation des feuilles",
    "finalisation du lot",
)
BATCH_HISTOGRAMME_BORNES = (
    "1", "2", "4", "8", "16", "32", "64", "128", "256", "512 et +")

ZONES_PYTHON = ("creation_evaluateur", "appel_cpp", "conversion", "nettoyage")
ZONES_PYTHON_LISIBLES = {
    "creation_evaluateur": "creation de l'evaluateur ONNX",
    "appel_cpp": "appel C++ complet (generation)",
    "conversion": "conversion des exemples",
    "nettoyage": "nettoyage et vidage du cache GPU",
}


def ms(duree_ns: float) -> float:
    return duree_ns / 1e6


def part(numerateur_ns: float, total_ns: float) -> float:
    if total_ns <= 0:
        return 0.0
    return 100.0 * numerateur_ns / total_ns


def resumer_passage(timing, zones_python_s: dict, debit: dict,
                    config: dict | None = None) -> dict:
    """Rassemble un passage en une structure unique, sans arrondi trompeur."""
    wall = int(timing.generation_wall_ns)
    phases = []
    for index, nom in enumerate(PHASES):
        duree = int(timing.phase_wall_ns[index])
        phases.append({
            "nom": nom,
            "lisibles": PHASES_LISIBLES[index],
            "duree_ns": duree,
            "part_generation_pct": part(duree, wall),
        })
    sous_onnx = {
        "run_ns": int(timing.onnx_run_ns),
        "softmax_ns": int(timing.onnx_softmax_ns),
        "racines_run_ns": int(timing.root_expansion_onnx_run_ns),
        "racines_softmax_ns": int(
            timing.root_expansion_onnx_softmax_ns),
    }
    sous_onnx["total_ns"] = (
        sous_onnx["run_ns"] + sous_onnx["softmax_ns"]
        + sous_onnx["racines_run_ns"] + sous_onnx["racines_softmax_ns"])

    appel_evaluateur_ns = phases[4]["duree_ns"]
    histogramme = [
        {"borne": BATCH_HISTOGRAMME_BORNES[index], "appels": int(compte)}
        for index, compte in enumerate(timing.batch_histogram)
    ]

    descentes = int(timing.search_descents)
    profondeur = int(timing.search_depth_sum)
    enfants = int(timing.search_children_scored)
    return {
        "recherche": {
            "descentes": descentes,
            "profondeur_cumulee": profondeur,
            "profondeur_moyenne": profondeur / descentes if descentes else 0.0,
            "profondeur_max": int(timing.search_depth_max),
            "enfants_examines": enfants,
            "enfants_par_descente": enfants / descentes if descentes else 0.0,
        },
        "config": config or {},
        "debit": dict(debit),
        "temps": {
            "generation_wall_ns": wall,
            "phases": phases,
            "residu_ns": int(timing.generation_other_wall_ns),
            "sous_onnx": sous_onnx,
            "part_onnx_de_l_appel_pct": part(sous_onnx["total_ns"],
                                            appel_evaluateur_ns),
            "zones_python_s": dict(zones_python_s),
            "total_python_s": sum(zones_python_s.values()),
            "part_appel_cpp_du_total_python_pct": part(
                1e9 * zones_python_s.get("appel_cpp", 0.0),
                1e9 * sum(zones_python_s.values())),
        },
        "reseau": {
            "appels_batch": int(timing.batch_calls),
            "lignes_batch": int(timing.batch_rows),
            "lignes_moyennes": (
                timing.batch_rows / timing.batch_calls
                if timing.batch_calls else 0.0),
            "lignes_max": int(timing.max_batch_rows),
            "changements_taille": int(timing.batch_size_changes),
            "appels_unitaires": int(timing.unit_network_calls),
            "lignes_unitaires": int(timing.unit_network_rows),
            "requetes_feuilles": int(timing.leaf_requests),
            "histogramme": histogramme,
        },
        "vidange": {
            "duree_ns": int(timing.drain_wall_ns),
            "part_generation_pct": part(timing.drain_wall_ns, wall),
            "appels": int(timing.drain_batch_calls),
            "lignes": int(timing.drain_batch_rows),
            "lignes_moyennes": (
                timing.drain_batch_rows / timing.drain_batch_calls
                if timing.drain_batch_calls else 0.0),
        },
        "attente": {
            "tours": int(timing.loop_turns),
            "tours_lot_pret_non_envoye": int(timing.deferred_turns),
            "duree_lot_pret_non_envoye_ns": int(
                timing.deferred_wall_ns),
            "age_max_lot_en_attente_ns": int(
                timing.max_pending_age_ns),
        },
        "fin": {
            "simulations_terminees": int(timing.completed_sims),
            "sans_reseau": int(timing.no_network_sims),
            "terminales": int(timing.terminal_sims),
            "hits_table": int(timing.tt_hits),
            "misses_table": int(timing.tt_misses),
            "expansions_racine": int(timing.root_expansions),
            "exemples_lents_sauves": int(
                timing.slow_examples_saved),
        },
        "workers": {
            "mode": int(timing.mode),
            "nombre": int(timing.worker_count),
            "somme_ns": int(timing.worker_busy_sum_ns),
            "maximum_ns": int(timing.worker_busy_max_ns),
        },
    }


def formater_diagnostic(resume: dict) -> str:
    """Rapport lisible d'un passage. Les pourcentages citent leur denominateur."""
    debit = resume["debit"]
    temps = resume["temps"]
    reseau = resume["reseau"]
    attente = resume["attente"]
    recherche = resume["recherche"]
    fin = resume["fin"]
    workers = resume["workers"]
    wall = temps["generation_wall_ns"]
    lignes = ["[diag] debit (generation C++ et total Python)"]
    lignes.append(
        f"[diag]   coups nouveaux/s   : {debit.get('coups_nouveaux_par_s', 0):.2f}"
        f"   exemples/s : {debit.get('exemples_par_s', 0):.2f}"
        f"   parties/s : {debit.get('parties_par_s', 0):.2f}")
    lignes.append(
        f"[diag]   evaluations reseau/s : "
        f"{debit.get('evaluations_reseau_par_s', 0):.2f} (diagnostic, pas un "
        f"objectif)")
    lignes.append(
        f"[diag] temps : generation C++ {ms(wall):.1f} ms, "
        f"total Python {temps['total_python_s']:.2f} s")
    for phase in temps["phases"]:
        if phase["duree_ns"] == 0:
            continue
        lignes.append(
            f"[diag]   {phase['lisibles']:<32} {ms(phase['duree_ns']):>9.1f} ms "
            f"{phase['part_generation_pct']:>5.1f} % de la generation")
    lignes.append(
        f"[diag]   residu non classe{'':<16} {ms(temps['residu_ns']):>9.1f} ms "
        f"{part(temps['residu_ns'], wall):>5.1f} % de la generation")
    sous = temps["sous_onnx"]
    lignes.append(
        f"[diag]   dont ONNX Run {ms(sous['run_ns']):.1f} ms, softmax "
        f"{ms(sous['softmax_ns']):.1f} ms, racines {ms(sous['racines_run_ns']):.1f}"
        f" + {ms(sous['racines_softmax_ns']):.1f} ms, soit "
        f"{temps['part_onnx_de_l_appel_pct']:.1f} % de l'appel evaluateur")
    lignes.append(
        f"[diag] reseau : {reseau['appels_batch']} appels, "
        f"{reseau['lignes_batch']} lignes, moyenne "
        f"{reseau['lignes_moyennes']:.1f}, max {reseau['lignes_max']}, "
        f"{reseau['changements_taille']} changements de taille")
    lignes.append(
        f"[diag]   requetes feuilles {reseau['requetes_feuilles']}, "
        f"appels unitaires {reseau['appels_unitaires']}")
    lignes.append(
        "[diag]   histogramme : " + ", ".join(
            f"{element['borne']}:{element['appels']}"
            for element in reseau["histogramme"] if element["appels"]))
    vidange = resume["vidange"]
    lignes.append(
        f"[diag] vidange finale : {ms(vidange['duree_ns']):.1f} ms, "
        f"{vidange['part_generation_pct']:.1f} % de la generation, "
        f"{vidange['appels']} appels, moyenne {vidange['lignes_moyennes']:.1f} lignes")
    lignes.append(
        f"[diag] recherche : {recherche['descentes']} descentes, "
        f"profondeur moyenne {recherche['profondeur_moyenne']:.3f}, "
        f"max {recherche['profondeur_max']}, "
        f"{recherche['enfants_examines']} enfants examines "
        f"({recherche['enfants_par_descente']:.2f}/descente)")
    lignes.append(
        f"[diag] attente : {attente['tours']} tours, "
        f"{attente['tours_lot_pret_non_envoye']} avec lot pret non envoye "
        f"({ms(attente['duree_lot_pret_non_envoye_ns']):.1f} ms, age max "
        f"{ms(attente['age_max_lot_en_attente_ns']):.1f} ms)")
    lignes.append(
        f"[diag] fin : {fin['simulations_terminees']} simulations terminees "
        f"dont {fin['sans_reseau']} sans reseau et {fin['terminales']} "
        f"terminales, table {fin['hits_table']} hits / {fin['misses_table']} "
        f"misses, {fin['expansions_racine']} expansions de racine, "
        f"{fin['exemples_lents_sauves']} exemples sauves")
    if workers["mode"] >= 2:
        lignes.append(
            f"[diag] workers ({workers['nombre']}) : travail cumule "
            f"{ms(workers['somme_ns']):.1f} ms, maximum d'une iteration "
            f"{ms(workers['maximum_ns']):.1f} ms (cumul, pas une duree murale)")
    zones = temps["zones_python_s"]
    if zones:
        for zone in ZONES_PYTHON:
            if zone in zones:
                lendemain = ZONES_PYTHON_LISIBLES.get(zone, zone)
                lignes.append(
                    f"[diag] python {lendemain:<32} {zones[zone]:>8.2f} s")
    return "\n".join(lignes)


def metriques_wandb(resume: dict) -> dict:
    """Metriques d'une generation pour W&B, prefixees selfplay/.

    Le cout d'un appel au reseau et le remplissage des lots suffisent a
    separer les deux causes possibles d'un ralentissement : un appel plus cher
    (machine, runtime) ou des lots moins remplis (ordonnancement, vidange).
    """
    temps, reseau, vidange, fin = (resume["temps"], resume["reseau"],
                                   resume["vidange"], resume["fin"])
    wall = temps["generation_wall_ns"]
    appels = reseau["appels_batch"]
    simulations = fin["simulations_terminees"]
    # Seaux 1, 2, 4 et 8 de l'histogramme : les appels d'au plus 8 lignes.
    petits = sum(element["appels"] for element in reseau["histogramme"][:4])
    metriques = {
        f"selfplay/phases/{phase['nom']}_pct": phase["part_generation_pct"]
        for phase in temps["phases"]
    }
    metriques.update({
        "selfplay/phases/residu_pct": part(temps["residu_ns"], wall),
        "selfplay/reseau/lignes_par_appel": reseau["lignes_moyennes"],
        "selfplay/reseau/ms_par_appel": (
            temps["sous_onnx"]["run_ns"] / 1e6 / appels if appels else 0.0),
        "selfplay/reseau/appels": appels,
        "selfplay/reseau/changements_taille": reseau["changements_taille"],
        "selfplay/reseau/petits_lots_pct": 100.0 * petits / appels if appels else 0.0,
        "selfplay/moteur/us_par_simulation": (
            wall / 1e3 / simulations if simulations else 0.0),
        "selfplay/moteur/sans_reseau_pct": (
            100.0 * fin["sans_reseau"] / simulations if simulations else 0.0),
        "selfplay/vidange/duree_s": vidange["duree_ns"] / 1e9,
        "selfplay/vidange/part_pct": vidange["part_generation_pct"],
        "selfplay/vidange/lignes_par_appel": vidange["lignes_moyennes"],
    })
    recherche = resume["recherche"]
    metriques.update({
        f"selfplay/recherche/{nom}": valeur
        for nom, valeur in recherche.items()
    })
    collecte_ns = temps["phases"][2]["duree_ns"]
    for nom, denominateur in (
            ("descente", recherche["descentes"]),
            ("niveau", recherche["profondeur_cumulee"]),
            ("tour", resume["attente"]["tours"])):
        metriques[f"selfplay/collecte/us_par_{nom}"] = (
            collecte_ns / 1e3 / denominateur if denominateur else 0.0)
    return metriques


def formater_bilan(resumes: list[dict]) -> str:
    """Bilan multi-passages : trois debits et leur dispersion, pas un seul point."""
    if not resumes:
        return "[diag] aucun passage"
    lignes = ["[diag] bilan par passage"]
    for index, resume in enumerate(resumes):
        debit = resume["debit"]
        lignes.append(
            f"[diag]   passage {index + 1} : "
            f"coups nouveaux/s {debit.get('coups_nouveaux_par_s', 0):.2f}, "
            f"exemples/s {debit.get('exemples_par_s', 0):.2f}, "
            f"reseau/s {debit.get('evaluations_reseau_par_s', 0):.2f}")
    return "\n".join(lignes)


def vers_dictionnaire(resume: dict) -> dict:
    """Le resume est deja serialisable ; la fonction documente le contrat."""
    return resume


def empreinte_fichier(chemin, longueur: int = 16) -> str:
    """Empreinte courte d'un artefact, pour comparer deux passages."""
    digest = hashlib.sha256()
    with open(chemin, "rb") as fichier:
        for bloc in iter(lambda: fichier.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()[:longueur]


def revision_git(racine) -> str:
    try:
        resultat = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(racine), capture_output=True,
            text=True, timeout=10, check=False)
        if resultat.returncode == 0:
            return resultat.stdout.strip()
    except OSError:
        pass
    return "inconnue"


def collecter_configuration(*, modele, binaire, puzzles, concurrent, games,
                            slow_sims, fast_sims, slow_ratio, tt_size, mode,
                            passes, racine) -> dict:
    """Identite du passage : code, binaire, modele, machine, budgets."""
    configuration = {
        "revision_git": revision_git(racine),
        "modele": str(modele),
        "empreinte_modele": empreinte_fichier(modele),
        "binaire": str(binaire),
        "empreinte_binaire": empreinte_fichier(binaire),
        "puzzles": str(puzzles) if puzzles else None,
        "concurrent_games": concurrent,
        "games": games,
        "slow_sims": slow_sims,
        "fast_sims": fast_sims,
        "slow_ratio": slow_ratio,
        "tt_size": tt_size,
        "diagnostics_mode": mode,
        "passages": passes,
        "python": sys.version.split()[0],
        "plateforme": platform.platform(),
        "processeur": platform.processor(),
        "coeurs_logiques": os.cpu_count(),
    }
    try:
        import torch
        configuration["torch"] = torch.__version__
        if torch.cuda.is_available():
            configuration["gpu"] = torch.cuda.get_device_name(0)
            configuration["cuda"] = torch.version.cuda
    except ImportError:
        configuration["torch"] = None
    return configuration
