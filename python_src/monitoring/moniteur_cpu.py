"""Echantillonnage de la frequence effective et de la charge du CPU sous Windows.

Lit trois compteurs de performance Windows par l'API PDH (pdh.dll), sans
dependance a installer :

- % Processor Performance : frequence effective rapportee a la frequence
  nominale. Un bridage thermique ou energetique la fait passer sous 100 %.
- % Processor Utility : travail effectif, charge ponderee par la frequence.
- % Processor Time : charge classique.

psutil.cpu_freq() ne convient pas : sous Windows, il renvoie la frequence
nominale, constante.

Un fil secondaire lit les compteurs a intervalle fixe ; chaque lecture coute
quelques millisecondes, sans effet mesurable sur une generation de plusieurs
minutes. Hors Windows, ou si les compteurs sont indisponibles, l'echantillonneur
n'est pas cree et l'entrainement continue sans ces metriques.
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import ctypes
import statistics
import sys
import threading

COMPTEURS = {
    "performance_pct": r"\Processor Information(_Total)\% Processor Performance",
    "utilite_pct": r"\Processor Information(_Total)\% Processor Utility",
    "charge_pct": r"\Processor Information(_Total)\% Processor Time",
}

_PDH_FMT_DOUBLE = 0x00000200


class _ValeurPdh(ctypes.Structure):
    # PDH_FMT_COUNTERVALUE : un statut puis une union alignee sur 8 octets.
    _fields_ = [("statut", ctypes.c_ulong), ("valeur", ctypes.c_double)]


class LecteurPdh:
    """Lit les compteurs de COMPTEURS ; les compteurs de taux exigent deux collectes."""

    def __init__(self):
        from ctypes import wintypes
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._requete = wintypes.HANDLE()
        self._verifier(self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._requete)),
                       "PdhOpenQueryW")
        self._compteurs = {}
        for nom, chemin in COMPTEURS.items():
            poignee = wintypes.HANDLE()
            # Version anglaise : les noms de compteurs sont traduits selon la
            # langue de Windows, pas leurs equivalents anglais.
            self._verifier(self._pdh.PdhAddEnglishCounterW(
                self._requete, chemin, 0, ctypes.byref(poignee)),
                f"PdhAddEnglishCounterW({chemin})")
            self._compteurs[nom] = poignee
        self._verifier(self._pdh.PdhCollectQueryData(self._requete),
                       "PdhCollectQueryData")

    @staticmethod
    def _verifier(code, appel):
        if code != 0:
            raise OSError(f"{appel} a echoue (code PDH 0x{code & 0xFFFFFFFF:08X})")

    def lire(self):
        """Valeurs moyennes depuis la lecture precedente."""
        self._verifier(self._pdh.PdhCollectQueryData(self._requete),
                       "PdhCollectQueryData")
        valeurs = {}
        for nom, poignee in self._compteurs.items():
            valeur = _ValeurPdh()
            code = self._pdh.PdhGetFormattedCounterValue(
                poignee, _PDH_FMT_DOUBLE, None, ctypes.byref(valeur))
            if code == 0:
                valeurs[nom] = valeur.valeur
        return valeurs

    def fermer(self):
        self._pdh.PdhCloseQuery(self._requete)


class EchantillonneurCpu:
    """Accumule les lectures d'un lecteur a intervalle fixe, dans un fil secondaire."""

    def __init__(self, lecteur, periode_s=15.0):
        self._lecteur = lecteur
        self._periode_s = periode_s
        self._echantillons = []
        self._verrou = threading.Lock()
        self._arret = threading.Event()
        self._fil = None

    def demarrer(self):
        self._fil = threading.Thread(target=self._boucle, name="moniteur_cpu",
                                     daemon=True)
        self._fil.start()

    def _boucle(self):
        while not self._arret.wait(self._periode_s):
            self.echantillonner()

    def echantillonner(self):
        try:
            valeurs = self._lecteur.lire()
        except OSError:
            return
        with self._verrou:
            self._echantillons.append(valeurs)

    def reinitialiser(self):
        """Oublie les lectures passees, par exemple au debut d'une generation."""
        with self._verrou:
            self._echantillons.clear()

    def metriques(self, prefixe):
        """Moyenne de chaque compteur depuis la reinitialisation, et minimum de la frequence."""
        with self._verrou:
            echantillons = list(self._echantillons)
        resultat = {f"{prefixe}/echantillons": len(echantillons)}
        for nom in COMPTEURS:
            serie = [e[nom] for e in echantillons if nom in e]
            if serie:
                resultat[f"{prefixe}/{nom}"] = statistics.fmean(serie)
                if nom == "performance_pct":
                    resultat[f"{prefixe}/performance_min_pct"] = min(serie)
        return resultat

    def arreter(self):
        self._arret.set()
        if self._fil is not None:
            self._fil.join(timeout=2 * self._periode_s)
        self._lecteur.fermer()


def creer_echantillonneur(periode_s=15.0):
    """Echantillonneur demarre, ou None si les compteurs Windows sont indisponibles."""
    if sys.platform != "win32":
        return None
    try:
        lecteur = LecteurPdh()
    except OSError as erreur:
        print(f"[moniteur CPU] compteurs indisponibles, metriques ignorees : {erreur}")
        return None
    echantillonneur = EchantillonneurCpu(lecteur, periode_s)
    echantillonneur.demarrer()
    return echantillonneur
