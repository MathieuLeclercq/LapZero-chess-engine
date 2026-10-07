"""Comparaison appariee : bootstrap, refus des campagnes non comparables et
verdicts. Aucun modele, aucun processus."""
import json
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

from dev_tools.multicore_comparison import (
    MARGE_NON_INFERIORITE,
    comparer,
    format_rapport,
    paired_interval,
    verdict_qualite,
)


def test_paired_interval_deux_gains_et_une_perte():
    """Le signe et l'ordre de grandeur du delta sont verrouilles, et le meme
    seed doit donner exactement le meme intervalle."""
    baseline = [False, False, True]
    candidate = [True, True, True]

    delta, bas, haut = paired_interval(baseline, candidate, seed=42)

    assert delta == pytest.approx(2.0 / 3.0)
    assert bas >= 0.0
    assert haut <= 1.0
    assert paired_interval(baseline, candidate, seed=42) == (delta, bas, haut)


def test_paired_interval_tout_identique_est_nul():
    baseline = [True, False, True, False]

    delta, bas, haut = paired_interval(baseline, baseline)

    assert (delta, bas, haut) == (0.0, 0.0, 0.0)


def test_paired_interval_refuse_des_tailles_differentes():
    with pytest.raises(ValueError):
        paired_interval([True], [True, False])
    with pytest.raises(ValueError):
        paired_interval([], [])


def test_verdicts_selon_les_bornes():
    assert verdict_qualite(-0.005, 0.02) == "non-inferiorite"
    assert verdict_qualite(-0.05, -0.02) == "regression"
    assert verdict_qualite(-0.05, 0.005) == "indetermine"
    assert verdict_qualite(MARGE_NON_INFERIORITE, 0.01) == "non-inferiorite"


def test_verdicts_aux_bornes_exactes():
    """Le plan dit high < -0.01 pour une regression et low >= -0.01 pour la
    non-inferiorite : une borne exactement a la marge n'est jamais un echec."""
    assert verdict_qualite(MARGE_NON_INFERIORITE, 0.05) == "non-inferiorite"
    assert verdict_qualite(-0.05, MARGE_NON_INFERIORITE) == "indetermine"


def _ecrire_campagne(tmp_path, nom, reussites, meta_extra=None,
                     erreurs=None, index_force=None):
    import csv as csv_mod

    from evaluation.puzzle_bench import CHAMPS_CSV, chemin_sidecar

    chemin = tmp_path / f"{nom}.csv"
    erreurs = erreurs or {}
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(CHAMPS_CSV))
        writer.writeheader()
        for position, reussi in enumerate(reussites):
            index = position if index_force is None else index_force[position]
            writer.writerow({
                "ligne": index, "rating": 1500, "themes": "fork",
                "plies_historique": 0, "nb_coups_legaux": 30,
                "coup_reseau": "e2e4", "reussi_reseau": False,
                "p_correct_reseau": 0.1, "rang_correct_reseau": 3,
                "value_reseau": 0.0, "coup_recherche": "e2e4",
                "reussi_recherche": reussi, "part_visites_correct": 0.5,
                "duree_s": 0.1, "simulations_recherche": 32,
                "depassement_s": 0.0, "erreur": erreurs.get(position, ""),
            })

    meta = {
        "modele": "m.onnx", "iteration": 316, "global_step": 1,
        "simulations": 32, "search_seconds": None,
        "budget_label": "32 simulations",
        "c_puct": 1.4, "batch_size": 8, "cache_history_depth": 0,
        "fichier_banc": "data/puzzles_bench.txt", "sans_historique": False,
        "duree_totale_s": 1.0, "travailleurs": 1, "search_workers": 1,
        "accelerateur": "CPU", "modele_sha256": "a" * 64,
        "banc_sha256": "b" * 64, "critere": "premier_coup_recherche",
    }
    meta.update(meta_extra or {})
    chemin_sidecar(chemin).write_text(
        json.dumps(meta), encoding="utf-8")
    return chemin


def test_comparer_refuse_une_ligne_manquante(tmp_path):
    reference = _ecrire_campagne(tmp_path, "ref", [True, False, True])
    candidat = _ecrire_campagne(tmp_path, "cand", [True, False])

    with pytest.raises(ValueError, match="memes lignes"):
        comparer(reference, candidat)


def test_comparer_refuse_un_doublon(tmp_path):
    reference = _ecrire_campagne(
        tmp_path, "ref", [True, False], index_force=[0, 0])
    candidat = _ecrire_campagne(tmp_path, "cand", [True, False])

    with pytest.raises(ValueError, match="doublon"):
        comparer(reference, candidat)


def test_comparer_refuse_une_ligne_en_erreur(tmp_path):
    reference = _ecrire_campagne(
        tmp_path, "ref", [True, False], erreurs={1: "solution_illegale"})
    candidat = _ecrire_campagne(tmp_path, "cand", [True, False])

    with pytest.raises(ValueError, match="erreur"):
        comparer(reference, candidat)


def test_comparer_refuse_des_metadonnees_incompatibles(tmp_path):
    reference = _ecrire_campagne(tmp_path, "ref", [True, False])
    candidat = _ecrire_campagne(
        tmp_path, "cand", [True, False],
        meta_extra={"modele_sha256": "c" * 64})

    with pytest.raises(ValueError, match="modele_sha256"):
        comparer(reference, candidat)


def test_comparer_refuse_un_sidecar_absent(tmp_path):
    from evaluation.puzzle_bench import chemin_sidecar

    reference = _ecrire_campagne(tmp_path, "ref", [True, False])
    candidat = _ecrire_campagne(tmp_path, "cand", [True, False])
    chemin_sidecar(candidat).unlink()

    with pytest.raises(FileNotFoundError):
        comparer(reference, candidat)


def test_comparer_refuse_une_colonne_absente(tmp_path):
    """Un CSV tronque ne doit pas produire une comparaison vide : sans la
    colonne de reussite, la lecture echoue clairement."""
    reference = _ecrire_campagne(tmp_path, "ref", [True])
    reference.write_text("ligne,erreur\n0,\n", encoding="utf-8")
    candidat = _ecrire_campagne(tmp_path, "cand", [True])

    with pytest.raises(ValueError, match="reussi_recherche"):
        comparer(reference, candidat)


@pytest.mark.parametrize("cle, valeur", [
    ("simulations", 64),
    ("search_seconds", 2.0),
])
def test_comparer_refuse_des_budgets_differents(tmp_path, cle, valeur):
    reference = _ecrire_campagne(tmp_path, "ref", [True, False])
    candidat = _ecrire_campagne(
        tmp_path, "cand", [True, False], meta_extra={cle: valeur})

    with pytest.raises(ValueError, match=cle):
        comparer(reference, candidat)


def test_comparer_non_inferiorite_et_paires_discordantes(tmp_path):
    reference_reussites = [i % 2 == 0 for i in range(100)]
    candidat_reussites = list(reference_reussites)
    candidat_reussites[1] = True          # un gain, aucune perte
    reference = _ecrire_campagne(tmp_path, "ref", reference_reussites)
    candidat = _ecrire_campagne(
        tmp_path, "cand", candidat_reussites, meta_extra={"search_workers": 8})

    comparaison, meta_ref, meta_cand = comparer(reference, candidat)

    assert comparaison.total == 100
    assert comparaison.verdict == "non-inferiorite"
    assert comparaison.discordantes_candidat_perd == 0
    assert comparaison.discordantes_candidat_gagne == 1
    assert meta_ref["search_workers"] == 1
    assert meta_cand["search_workers"] == 8


def test_comparer_regression_est_detectee(tmp_path):
    reference_reussites = [i % 2 == 0 for i in range(100)]
    candidat_reussites = list(reference_reussites)
    for position in range(0, 20, 2):       # dix pertes nettes
        candidat_reussites[position] = False
    reference = _ecrire_campagne(tmp_path, "ref", reference_reussites)
    candidat = _ecrire_campagne(tmp_path, "cand", candidat_reussites)

    comparaison, _, _ = comparer(reference, candidat)

    assert comparaison.verdict == "regression"
    assert comparaison.discordantes_candidat_perd == 10
    assert comparaison.discordantes_candidat_gagne == 0


def test_le_rapport_precise_la_limite_du_bootstrap(tmp_path):
    """Le bootstrap mesure l'echantillon de puzzles, pas la variabilite
    d'ordonnancement multicœur : le rapport doit le dire."""
    reference = _ecrire_campagne(tmp_path, "ref", [True, False])
    candidat = _ecrire_campagne(tmp_path, "cand", [True, True])

    comparaison, meta_ref, meta_cand = comparer(reference, candidat)
    texte = format_rapport(comparaison, meta_ref, meta_cand,
                           reference, candidat, seed=42)

    assert "pas toute la variabilite" in texte


def test_format_rapport_affiche_les_metadonnees_et_le_verdict(tmp_path):
    reference = _ecrire_campagne(tmp_path, "ref", [True, False, True])
    candidat = _ecrire_campagne(
        tmp_path, "cand", [True, True, True], meta_extra={"search_workers": 8})

    comparaison, meta_ref, meta_cand = comparer(reference, candidat)
    texte = format_rapport(comparaison, meta_ref, meta_cand,
                           reference, candidat, seed=42)

    assert "Comparaison appariee" in texte
    assert "8 workers" in texte
    assert "32 simulations" in texte
    assert "a" * 12 in texte
    assert comparaison.verdict.lower() in texte.lower()
    assert "—" not in texte
