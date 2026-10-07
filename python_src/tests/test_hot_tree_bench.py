"""Logique pure du banc du cas chaud. Aucun moteur, aucun modele."""
import os
import sys

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from dev_tools.hot_tree_bench import agreger_coups


def _ligne(position, nps, nn_calls, sans_reseau):
    return {
        "position": position,
        "coup": 0,
        "sims": 700,
        "sims_par_seconde": nps,
        "nn_calls": nn_calls,
        "tt_hits": 0,
        "terminal_hits": 0,
        "sans_reseau": sans_reseau,
    }


def test_agreger_coups_resume_par_position():
    lignes = [
        _ligne("ouverture", 1000.0, 700, 0.0),
        _ligne("ouverture", 2000.0, 700, 0.0),
        _ligne("finale", 1500.0, 690, 0.014),
    ]

    resumes = agreger_coups(lignes)

    assert set(resumes) == {"ouverture", "finale"}
    assert resumes["ouverture"]["coups"] == 2
    assert resumes["ouverture"]["nps_median"] == 1500.0
    assert resumes["finale"]["nn_calls_median"] == 690
    assert resumes["finale"]["sans_reseau_median"] == 0.014


def test_agreger_coups_supporte_une_liste_vide():
    assert agreger_coups([]) == {}
