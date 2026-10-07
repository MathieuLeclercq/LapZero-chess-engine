import csv
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src" / "dev_tools"))

import compare_bench_models

COLONNES = ["ligne", "reussi_recherche", "p_correct_reseau", "value_reseau",
            "part_visites_correct", "erreur"]


def ligne(numero, reussi, prior, value=0.0, visites=0.9, erreur=""):
    return {
        "ligne": numero,
        "reussi_recherche": str(reussi),
        "p_correct_reseau": str(prior),
        "value_reseau": str(value),
        "part_visites_correct": str(visites),
        "erreur": erreur,
    }


def ecrire(chemin, lignes):
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES)
        ecrivain.writeheader()
        for element in lignes:
            ecrivain.writerow(element)


def test_mcnemar_exact():
    assert compare_bench_models.mcnemar_exact(0, 0) == 1.0
    assert compare_bench_models.mcnemar_exact(1, 1) == 1.0
    assert compare_bench_models.mcnemar_exact(1, 4) == pytest.approx(0.375)
    assert compare_bench_models.mcnemar_exact(0, 5) == pytest.approx(0.0625)


def test_comparaison_appariee(tmp_path):
    reference = tmp_path / "reference.csv"
    candidat = tmp_path / "candidat.csv"
    ecrire(reference, [
        ligne(0, True, 0.2, value=0.10), ligne(1, True, 0.3, value=0.20),
        ligne(2, True, 0.4, value=0.30), ligne(3, True, 0.5, value=0.40),
        ligne(4, False, 0.1, value=0.50), ligne(5, False, 0.15, value=0.60),
    ])
    ecrire(candidat, [
        ligne(0, True, 0.25, value=0.20), ligne(1, True, 0.35, value=0.30),
        ligne(2, True, 0.45, value=0.40), ligne(3, False, 0.55, value=0.50),
        ligne(4, True, 0.2, value=0.60), ligne(5, False, 0.18, value=0.70),
    ])

    resultat = compare_bench_models.comparer(
        compare_bench_models.charger(reference),
        compare_bench_models.charger(candidat))

    assert resultat["communs"] == 6
    assert resultat["erreurs"] == 0
    assert resultat["reference_reussis"] == 4
    assert resultat["candidat_reussis"] == 4
    assert resultat["reference_seul"] == 1
    assert resultat["candidat_seul"] == 1
    assert resultat["p_mcnemar"] == 1.0

    prior_reference, prior_candidat, ecart = resultat["medias"][
        "p_correct_reseau"]
    assert prior_reference == pytest.approx(0.25)
    assert prior_candidat == pytest.approx(0.30)
    assert ecart == pytest.approx(0.05)

    value_reference, value_candidat, ecart_value = resultat["medias"][
        "value_reseau"]
    assert value_reference == pytest.approx(0.35)
    assert value_candidat == pytest.approx(0.45)
    assert ecart_value == pytest.approx(0.10)


def test_appariement_par_ligne_et_erreurs(tmp_path):
    reference = tmp_path / "reference.csv"
    candidat = tmp_path / "candidat.csv"
    ecrire(reference, [
        ligne(0, True, 0.2),
        ligne(1, False, 0.1),
        ligne(7, True, 0.4),
        ligne(9, True, 0.5, erreur="timeout"),
    ])
    ecrire(candidat, [
        ligne(0, False, 0.25),
        ligne(1, True, 0.15),
        ligne(7, True, 0.45),
        ligne(8, True, 0.3),
        ligne(9, True, 0.5),
    ])

    resultat = compare_bench_models.comparer(
        compare_bench_models.charger(reference),
        compare_bench_models.charger(candidat))

    assert resultat["communs"] == 4
    assert resultat["erreurs"] == 1
    assert resultat["reference_seul"] == 1
    assert resultat["candidat_seul"] == 1

    texte = compare_bench_models.formater(resultat, "ref.csv", "cand.csv")
    assert "Puzzles apparies : 4" in texte
    assert "McNemar exact bilateral" in texte


def test_message_tailles_differentes(tmp_path):
    reference = tmp_path / "reference.csv"
    candidat = tmp_path / "candidat.csv"
    ecrire(reference, [ligne(0, True, 0.2), ligne(1, True, 0.3),
                       ligne(2, False, 0.1)])
    ecrire(candidat, [ligne(0, True, 0.25), ligne(1, False, 0.35),
                      ligne(2, True, 0.15), ligne(3, False, 0.4),
                      ligne(4, True, 0.5)])

    resultat = compare_bench_models.comparer(
        compare_bench_models.charger(reference),
        compare_bench_models.charger(candidat))

    assert resultat["reference_lignes"] == 3
    assert resultat["candidat_lignes"] == 5
    assert resultat["communs"] == 3

    texte = compare_bench_models.formater(resultat, "ref.csv", "cand.csv")
    assert "Tailles differentes : ref.csv 3 lignes, cand.csv 5 lignes" in texte
    assert "comparaison sur les 3 puzzles communs" in texte


def test_message_lignes_ecartees_a_taille_egale(tmp_path):
    reference = tmp_path / "reference.csv"
    candidat = tmp_path / "candidat.csv"
    ecrire(reference, [ligne(0, True, 0.2), ligne(1, True, 0.3),
                       ligne(2, False, 0.1)])
    ecrire(candidat, [ligne(1, True, 0.25), ligne(2, False, 0.35),
                      ligne(3, True, 0.15)])

    resultat = compare_bench_models.comparer(
        compare_bench_models.charger(reference),
        compare_bench_models.charger(candidat))

    assert resultat["communs"] == 2
    texte = compare_bench_models.formater(resultat, "ref.csv", "cand.csv")
    assert "Puzzles communs : 2 sur 3 lignes" in texte


def test_aucun_message_quand_les_perimetres_sont_identiques(tmp_path):
    reference = tmp_path / "reference.csv"
    candidat = tmp_path / "candidat.csv"
    lignes = [ligne(0, True, 0.2), ligne(1, False, 0.3)]
    ecrire(reference, lignes)
    ecrire(candidat, lignes)

    resultat = compare_bench_models.comparer(
        compare_bench_models.charger(reference),
        compare_bench_models.charger(candidat))

    texte = compare_bench_models.formater(resultat, "ref.csv", "cand.csv")
    assert "Tailles differentes" not in texte
    assert "Puzzles communs" not in texte
