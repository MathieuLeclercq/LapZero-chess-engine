import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

from evaluation.match_modeles import bilan_du_match, elo_depuis_score, points_de_b


def test_un_score_de_moitie_vaut_zero_elo():
    assert elo_depuis_score(0.5) == pytest.approx(0.0)


def test_l_ecart_elo_est_antisymetrique():
    assert elo_depuis_score(0.64) == pytest.approx(-elo_depuis_score(0.36))
    assert elo_depuis_score(0.64) == pytest.approx(100.0, abs=1.0)


def test_un_score_parfait_reste_fini():
    assert elo_depuis_score(1.0) < 1300
    assert elo_depuis_score(0.0) > -1300


def test_les_points_suivent_la_couleur_de_b():
    assert points_de_b({"blancs": "B", "resultat": "1-0"}) == 1.0
    assert points_de_b({"blancs": "A", "resultat": "1-0"}) == 0.0
    assert points_de_b({"blancs": "A", "resultat": "0-1"}) == 1.0
    assert points_de_b({"blancs": "B", "resultat": "1/2-1/2"}) == 0.5


def test_le_bilan_moyenne_par_partie_et_non_par_paire():
    bilan = bilan_du_match([2.0, 1.0, 1.0, 0.0], victoires_b=3, nulles=2, defaites_b=3)

    assert bilan.parties == 8
    assert bilan.score_b == pytest.approx(0.5)
    assert bilan.elo_b == pytest.approx(0.0)
    assert bilan.elo_bas < 0 < bilan.elo_haut


def test_l_intervalle_se_resserre_avec_le_nombre_de_paires():
    petit = bilan_du_match([2.0, 1.0, 0.0, 1.0], 0, 0, 0)
    grand = bilan_du_match([2.0, 1.0, 0.0, 1.0] * 25, 0, 0, 0)

    assert (grand.elo_haut - grand.elo_bas) < (petit.elo_haut - petit.elo_bas) / 4


def test_un_bilan_sans_paire_leve():
    with pytest.raises(ValueError):
        bilan_du_match([], 0, 0, 0)
