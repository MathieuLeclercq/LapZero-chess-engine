"""Tests du rapport de diagnostic self-play.

Aucun modele, aucun processus : le rapport est une fonction pure du rapport de
phases C++, des zones Python et des debits. Le timing factice imite le contrat
des champs exposes par le binding.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

from training.selfplay_diag import (
    formater_bilan,
    formater_diagnostic,
    metriques_wandb,
    part,
    resumer_passage,
)


def timing_factice(**remplacements):
    base = dict(
        enabled=True,
        mode=1,
        generation_wall_ns=1_000_000_000,
        phase_wall_ns=(10_000_000, 20_000_000, 500_000_000, 30_000_000,
                       300_000_000, 5_000_000, 100_000_000, 5_000_000),
        generation_other_wall_ns=30_000_000,
        onnx_run_ns=250_000_000,
        onnx_softmax_ns=40_000_000,
        root_expansion_onnx_run_ns=2_000_000,
        root_expansion_onnx_softmax_ns=1_000_000,
        loop_turns=1000,
        batch_calls=40,
        batch_rows=8000,
        unit_network_calls=16,
        unit_network_rows=16,
        max_batch_rows=256,
        batch_size_changes=3,
        deferred_turns=25,
        deferred_wall_ns=50_000_000,
        max_pending_age_ns=8_000_000,
        root_expansions=32,
        leaf_requests=8000,
        completed_sims=9000,
        no_network_sims=1000,
        terminal_sims=400,
        tt_hits=120,
        tt_misses=5000,
        slow_examples_saved=2000,
        batch_histogram=(0, 0, 0, 2, 4, 8, 10, 6, 8, 2),
        search_descents=9000,
        search_depth_sum=27000,
        search_depth_max=19,
        search_children_scored=540000,
        worker_count=8,
        worker_busy_sum_ns=900_000_000,
        worker_busy_max_ns=4_000_000,
        drain_wall_ns=100_000_000,
        drain_batch_calls=10,
        drain_batch_rows=500,
    )
    base.update(remplacements)
    return SimpleNamespace(**base)


ZONES = {
    "creation_evaluateur": 0.5,
    "appel_cpp": 10.0,
    "conversion": 1.0,
    "nettoyage": 0.5,
}


def test_resumer_passage_conserve_les_trois_debits():
    debit = {
        "coups_nouveaux_par_s": 450.0,
        "exemples_par_s": 80.0,
        "parties_par_s": 1.5,
        "evaluations_reseau_par_s": 801.6,
    }
    resume = resumer_passage(timing_factice(), ZONES, debit)

    assert resume["debit"] == debit
    assert resume["temps"]["total_python_s"] == pytest.approx(12.0)
    assert resume["temps"]["zones_python_s"]["conversion"] == pytest.approx(1.0)


def test_pourcentages_de_phase_utilisent_la_generation():
    resume = resumer_passage(timing_factice(), ZONES, {})
    phases = resume["temps"]["phases"]
    total_pct = sum(phase["part_generation_pct"] for phase in phases)
    assert total_pct == pytest.approx(97.0)
    assert resume["temps"]["residu_ns"] == 30_000_000
    assert resume["temps"]["part_onnx_de_l_appel_pct"] == pytest.approx(
        100.0 * 293_000_000 / 300_000_000)


def test_part_sur_total_nul_reste_nulle():
    assert part(10, 0) == 0.0
    assert part(0, 10) == 0.0


def test_reseau_rapporte_moyenne_max_et_histogramme():
    resume = resumer_passage(timing_factice(), ZONES, {})
    reseau = resume["reseau"]
    assert reseau["appels_batch"] == 40
    assert reseau["lignes_batch"] == 8000
    assert reseau["lignes_moyennes"] == pytest.approx(200.0)
    assert reseau["lignes_max"] == 256
    assert reseau["changements_taille"] == 3
    assert reseau["requetes_feuilles"] == 8000
    libelles = {element["borne"]: element["appels"]
                for element in reseau["histogramme"]}
    assert libelles["256"] == 8
    assert libelles["512 et +"] == 2
    assert libelles["1"] == 0


def test_attente_et_fin_sont_distinctes():
    resume = resumer_passage(timing_factice(), ZONES, {})
    assert resume["attente"]["tours_lot_pret_non_envoye"] == 25
    assert resume["attente"]["age_max_lot_en_attente_ns"] == 8_000_000
    assert resume["fin"]["sans_reseau"] == 1000
    assert resume["fin"]["terminales"] == 400
    assert resume["fin"]["expansions_racine"] == 32


def test_formater_diagnostic_cite_denominateurs_et_compteurs():
    resume = resumer_passage(timing_factice(), ZONES, {
        "coups_nouveaux_par_s": 450.0,
        "exemples_par_s": 80.0,
        "parties_par_s": 1.5,
        "evaluations_reseau_par_s": 801.6,
    })
    rapport = formater_diagnostic(resume)

    assert "de la generation" in rapport
    assert "softmax" in rapport
    assert "residu non classe" in rapport
    assert "histogramme" in rapport
    assert "256:8" in rapport
    assert "exemples sauves" in rapport
    assert "appel C++ complet" in rapport


def test_formater_diagnostic_affiche_le_detail_worker_en_mode_deux():
    resume = resumer_passage(timing_factice(mode=2), ZONES, {})
    rapport = formater_diagnostic(resume)
    assert "workers (8)" in rapport
    assert "pas une duree murale" in rapport

    sans_mode_deux = formater_diagnostic(
        resumer_passage(timing_factice(mode=1), ZONES, {}))
    assert "workers (8)" not in sans_mode_deux


def test_formater_bilan_liste_chaque_passage():
    resume = resumer_passage(timing_factice(), ZONES, {
        "coups_nouveaux_par_s": 450.0,
        "exemples_par_s": 80.0,
        "evaluations_reseau_par_s": 801.6,
    })
    bilan = formater_bilan([resume, resume])
    assert "passage 1" in bilan
    assert "passage 2" in bilan
    assert formater_bilan([]) == "[diag] aucun passage"


def test_la_vidange_est_resumee_et_affichee():
    resume = resumer_passage(timing_factice(), {}, {})

    assert resume["vidange"]["part_generation_pct"] == pytest.approx(10.0)
    assert resume["vidange"]["lignes_moyennes"] == pytest.approx(50.0)
    assert "vidange finale" in formater_diagnostic(resume)


def test_les_metriques_wandb_separent_cout_d_appel_et_remplissage():
    metriques = metriques_wandb(resumer_passage(timing_factice(), {}, {}))

    assert metriques["selfplay/reseau/lignes_par_appel"] == pytest.approx(200.0)
    assert metriques["selfplay/reseau/ms_par_appel"] == pytest.approx(6.25)
    assert metriques["selfplay/moteur/us_par_simulation"] == pytest.approx(1e6 / 9000)
    assert metriques["selfplay/moteur/sans_reseau_pct"] == pytest.approx(100 * 1000 / 9000)
    # Seaux 1, 2, 4 et 8 : 0 + 0 + 0 + 2 appels sur 40.
    assert metriques["selfplay/reseau/petits_lots_pct"] == pytest.approx(5.0)
    assert metriques["selfplay/vidange/duree_s"] == pytest.approx(0.1)
    assert all(cle.startswith("selfplay/") for cle in metriques)


def test_les_metriques_wandb_supportent_une_generation_vide():
    vide = timing_factice(batch_calls=0, batch_rows=0, completed_sims=0,
                          drain_batch_calls=0, drain_batch_rows=0,
                          batch_histogram=(0,) * 10)
    metriques = metriques_wandb(resumer_passage(vide, {}, {}))

    assert metriques["selfplay/reseau/ms_par_appel"] == 0.0
    assert metriques["selfplay/moteur/us_par_simulation"] == 0.0
