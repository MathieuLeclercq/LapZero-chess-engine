"""Memoire du processus aux frontieres de la boucle, en Mio."""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import logging

_MIO = 1024**2


def mesurer_memoire(processus=None):
    """Lit RSS et memoire privee si le systeme expose ce champ."""
    try:
        import psutil
    except ImportError:
        return {}
    try:
        processus = processus if processus is not None else psutil.Process()
        info = processus.memory_info()
    except (psutil.Error, OSError) as erreur:
        logging.getLogger(__name__).warning(
            "Mesure memoire indisponible : %s", erreur)
        return {}
    mesure = {"rss_mio": info.rss / _MIO}
    privee = getattr(info, "private", None)
    if privee is not None:
        mesure["privee_mio"] = privee / _MIO
    return mesure


def metriques_memoire(mesures):
    """Aplatit les mesures par etape pour W&B."""
    return {
        f"memoire/{etape}/{nom}": valeur
        for etape, mesure in mesures.items()
        for nom, valeur in mesure.items()
    }


def formater_memoire(mesures):
    """Affiche les releves du processus en Mio."""
    lignes = []
    for etape, mesure in mesures.items():
        if not mesure:
            continue
        privee = (f"{mesure['privee_mio']:.1f}"
                  if "privee_mio" in mesure else "n/d")
        lignes.append(
            f"[diag] memoire {etape} : privee {privee} Mio, "
            f"RSS {mesure['rss_mio']:.1f} Mio"
            if "rss_mio" in mesure else
            f"[diag] memoire {etape} : privee {privee} Mio")
    return "\n".join(lignes)
