"""Rapport du banc de self-play : compteurs distincts et denominateur honnete.

Le rapport doit distinguer les debuts, les fins, les plies reellement joues et
l'historique rejoue d'un puzzle, et n'utiliser que les plies nouveaux comme
denominateur du debit.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
sys.path.insert(0, str(RACINE / "python_src" / "dev_tools"))
os.add_dll_directory(str(RACINE / "python_src"))

import selfplay_refill_bench


def test_formater_bilan_distingue_debuts_fins_et_historique():
    resultats = [{
        "concurrent": 2,
        "total": 3,
        "duree": 10.0,
        "parties": 3,
        "debuts": 3,
        "fins": 3,
        "actives": 0,
        "nouveaux_plies": 120,
        "historique_rejoue": 40,
        "positions": 30,
        "parties_par_s": 0.3,
        "nouveaux_plies_par_s": 12.0,
        "ms_par_ply": 83.3,
        "positions_par_s": 3.0,
    }]

    lignes = selfplay_refill_bench.formater_bilan(resultats)

    assert len(lignes) == 2
    entete = lignes[0]
    for etiquette in ("debuts", "fins", "actives", "nouv. plies",
                      "hist. rejoue", "exemples"):
        assert etiquette in entete, etiquette

    corps = lignes[1]
    assert "(2,3)" in corps
    assert "120" in corps, "les plies nouveaux doivent apparaitre"
    assert "40" in corps, "l'historique rejoue garde sa colonne"


def test_mesurer_utilise_les_plies_nouveaux_comme_denominateur(monkeypatch):
    """total_real_moves inclut l'historique d'un puzzle : il ne doit jamais
    servir de denominateur au jeu produit."""

    class FaussesStats:
        games_started = 3
        games_completed = 3
        active_slots = 0
        new_plies = 100
        replayed_plies = 55

    class FaussePartie:
        state_tensors = type("T", (), {"shape": (7,)})()

    monkeypatch.setattr(
        selfplay_refill_bench.chess_engine,
        "generate_self_play_games_with_stats",
        lambda *args, **kwargs: ([FaussePartie(), FaussePartie()],
                                 FaussesStats()))

    resultat = selfplay_refill_bench.mesurer(object(), 2, 3, 2, 1, 0.5, 8192)

    assert resultat["debuts"] == 3
    assert resultat["fins"] == 3
    assert resultat["actives"] == 0
    assert resultat["nouveaux_plies"] == 100
    assert resultat["historique_rejoue"] == 55
    assert resultat["positions"] == 14
    assert resultat["ms_par_ply"] == pytest.approx(
        1000.0 * resultat["duree"] / 100)
