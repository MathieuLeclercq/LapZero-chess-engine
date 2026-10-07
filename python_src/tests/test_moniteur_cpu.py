import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

from monitoring.moniteur_cpu import EchantillonneurCpu, LecteurPdh


class LecteurFactice:
    def __init__(self, lectures):
        self._lectures = list(lectures)
        self.ferme = False

    def lire(self):
        return self._lectures.pop(0)

    def fermer(self):
        self.ferme = True


def test_les_metriques_moyennent_et_gardent_le_minimum_de_frequence():
    lecteur = LecteurFactice([
        {"performance_pct": 110.0, "utilite_pct": 50.0, "charge_pct": 60.0},
        {"performance_pct": 90.0, "utilite_pct": 40.0, "charge_pct": 70.0},
    ])
    echantillonneur = EchantillonneurCpu(lecteur)
    echantillonneur.echantillonner()
    echantillonneur.echantillonner()

    metriques = echantillonneur.metriques("selfplay/cpu")

    assert metriques["selfplay/cpu/echantillons"] == 2
    assert metriques["selfplay/cpu/performance_pct"] == pytest.approx(100.0)
    assert metriques["selfplay/cpu/performance_min_pct"] == pytest.approx(90.0)
    assert metriques["selfplay/cpu/charge_pct"] == pytest.approx(65.0)


def test_la_reinitialisation_oublie_les_lectures_passees():
    lecteur = LecteurFactice([{"performance_pct": 50.0}, {"performance_pct": 120.0}])
    echantillonneur = EchantillonneurCpu(lecteur)
    echantillonneur.echantillonner()
    echantillonneur.reinitialiser()
    echantillonneur.echantillonner()

    metriques = echantillonneur.metriques("cpu")

    assert metriques["cpu/echantillons"] == 1
    assert metriques["cpu/performance_pct"] == pytest.approx(120.0)


def test_sans_lecture_seul_le_nombre_d_echantillons_est_rapporte():
    metriques = EchantillonneurCpu(LecteurFactice([])).metriques("cpu")

    assert metriques == {"cpu/echantillons": 0}


def test_le_fil_s_arrete_et_ferme_le_lecteur():
    lecteur = LecteurFactice([])
    echantillonneur = EchantillonneurCpu(lecteur, periode_s=60.0)
    echantillonneur.demarrer()
    echantillonneur.arreter()

    assert lecteur.ferme


@pytest.mark.skipif(sys.platform != "win32", reason="compteurs Windows")
def test_les_compteurs_windows_rendent_des_valeurs_plausibles():
    lecteur = LecteurPdh()
    try:
        valeurs = lecteur.lire()
    finally:
        lecteur.fermer()

    assert 1.0 < valeurs["performance_pct"] < 300.0
    assert 0.0 <= valeurs["charge_pct"] <= 100.0
