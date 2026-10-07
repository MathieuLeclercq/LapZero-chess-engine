"""Compteurs de travail et memoire aux frontieres de la boucle."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.selfplay_diag import formater_diagnostic, metriques_wandb, resumer_passage
from test_selfplay_diag import timing_factice, ZONES


def test_recherche_mesure_le_travail_et_son_cout():
    resume = resumer_passage(timing_factice(), ZONES, {})
    assert "recherche" in resume
    assert resume["recherche"] == {
        "descentes": 9000, "profondeur_cumulee": 27000,
        "profondeur_moyenne": 3.0, "profondeur_max": 19,
        "enfants_examines": 540000, "enfants_par_descente": 60.0,
    }
    metrics = metriques_wandb(resume)
    assert metrics["selfplay/recherche/profondeur_moyenne"] == 3.0
    assert metrics["selfplay/recherche/enfants_examines"] == 540000
    assert metrics["selfplay/collecte/us_par_descente"] == pytest.approx(500000/9000)
    assert "profondeur" in formater_diagnostic(resume)


def test_recherche_vide_ne_divise_pas_par_zero():
    resume = resumer_passage(timing_factice(
        search_descents=0, search_depth_sum=0, search_depth_max=0,
        search_children_scored=0), {}, {})
    assert "recherche" in resume
    assert resume["recherche"]["profondeur_moyenne"] == 0.0
    assert metriques_wandb(resume)["selfplay/collecte/us_par_descente"] == 0.0


def test_memoire_utilise_des_mio_et_n_invente_pas_la_memoire_privee():
    assert importlib.util.find_spec("monitoring.moniteur_memoire") is not None
    from monitoring.moniteur_memoire import mesurer_memoire, metriques_memoire
    class Processus:
        def memory_info(self):
            return SimpleNamespace(rss=3*1024**2, private=2*1024**2)
    mesure = mesurer_memoire(Processus())
    assert mesure == {"rss_mio": 3.0, "privee_mio": 2.0}
    assert metriques_memoire({"avant_selfplay": mesure}) == {
        "memoire/avant_selfplay/rss_mio": 3.0,
        "memoire/avant_selfplay/privee_mio": 2.0,
    }
    class SansPrivee:
        def memory_info(self):
            return SimpleNamespace(rss=1024**2)
    assert mesurer_memoire(SansPrivee()) == {"rss_mio": 1.0}


def test_memoire_indisponible_ne_bloque_pas_la_run():
    assert importlib.util.find_spec("monitoring.moniteur_memoire") is not None
    import psutil
    from monitoring.moniteur_memoire import mesurer_memoire
    class Processus:
        def memory_info(self):
            raise psutil.AccessDenied(42)
    assert mesurer_memoire(Processus()) == {}
