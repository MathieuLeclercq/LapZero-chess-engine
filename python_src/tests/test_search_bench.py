"""Logique pure du harnais de debit. Aucun moteur, aucun modele."""
import os
import sys
from pathlib import Path

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from dev_tools.search_bench import (
    Mesure,
    agreger,
    inferences_par_seconde,
    remplissage_moyen,
    sims_par_seconde,
    taux_table,
)


class FauxMCTSSetters:
    """Setters attendus par configurer_mcts, sans effet par defaut."""

    def set_fixed_batch(self, enabled):
        pass

    def set_tuning(self, virtual_loss, fpu_reduction,
                   collision_attempt_factor):
        pass

    def set_timing_enabled(self, enabled):
        pass


def _m(position="depart", chemin="mcts_search", simulations=400, duree_s=1.0,
       nn_calls=300, nn_batches=300, tt_hits=100, tt_misses=300,
       terminal_hits=0, batch_size=0, cache_history_depth=1,
       worker_count=1, pool_etat="froid",
       tt_position_matches=100, tt_rule50_rejects=0,
       tt_context_rejects=0, tt_history_rejects=0, repetition=0,
       timing=None):
    return Mesure(position=position, chemin=chemin, simulations=simulations,
                  batch_size=batch_size, duree_s=duree_s, nn_calls=nn_calls,
                  nn_batches=nn_batches, tt_hits=tt_hits,
                  tt_misses=tt_misses, terminal_hits=terminal_hits,
                  cache_history_depth=cache_history_depth,
                  worker_count=worker_count, pool_etat=pool_etat,
                  tt_position_matches=tt_position_matches,
                  tt_rule50_rejects=tt_rule50_rejects,
                  tt_context_rejects=tt_context_rejects,
                  tt_history_rejects=tt_history_rejects,
                  repetition=repetition, timing=timing)


def _timing(wall_ns: int) -> dict[str, int]:
    champs = (
        "wall_ns", "selection_ns", "tensor_key_ns", "tt_probe_store_ns",
        "tt_wait_ns", "board_copy_ns", "batch_assembly_ns", "evaluator_ns",
        "expansion_ns", "backup_ns", "worker_wait_ns",
    )
    return {champ: wall_ns if champ == "wall_ns" else 0
            for champ in champs}


def test_les_trois_grandeurs_sont_distinctes():
    """Une simulation n'est pas une inference : elle peut s'arreter sur un
    noeud terminal ou sur un succes de table. Mesure sur une position de mat,
    381 simulations sur 400 s'arretaient sans jamais appeler le reseau."""
    m = _m(simulations=400, duree_s=2.0, nn_calls=300, tt_hits=100, tt_misses=300)

    assert sims_par_seconde(m) == pytest.approx(200.0)
    assert inferences_par_seconde(m) == pytest.approx(150.0)
    assert taux_table(m) == pytest.approx(0.25)


def test_taux_table_sans_consultation_vaut_zero():
    assert taux_table(_m(tt_hits=0, tt_misses=0)) == 0.0


def test_duree_nulle_ne_divise_pas_par_zero():
    m = _m(duree_s=0.0)

    assert sims_par_seconde(m) == 0.0
    assert inferences_par_seconde(m) == 0.0


def test_remplissage_moyen_distingue_positions_et_appels_batch():
    assert remplissage_moyen(_m(nn_calls=320, nn_batches=10)) == pytest.approx(32.0)
    assert remplissage_moyen(_m(nn_calls=0, nn_batches=0)) == 0.0


def test_agreger_groupe_par_position_et_chemin():
    mesures = [
        _m(position="depart", chemin="mcts_search", duree_s=1.0),
        _m(position="depart", chemin="mcts_search", duree_s=2.0),
        _m(position="depart", chemin="step_analysis", duree_s=1.0),
        _m(position="finale", chemin="mcts_search", duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 0, 1, 1, "froid"),
                        ("depart", "step_analysis", 0, 1, 1, "froid"),
                        ("finale", "mcts_search", 0, 1, 1, "froid")}
    assert agr[("depart", "mcts_search", 0, 1, 1, "froid")]["passages"] == 2


def test_agreger_groupe_aussi_par_taille_de_batch():
    mesures = [
        _m(batch_size=0, duree_s=4.0),
        _m(batch_size=32, duree_s=1.0),
        _m(batch_size=32, duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 0, 1, 1, "froid"),
                        ("depart", "mcts_search", 32, 1, 1, "froid")}
    assert agr[("depart", "mcts_search", 32, 1, 1, "froid")]["passages"] == 2
    assert agr[("depart", "mcts_search", 32, 1, 1, "froid")]["sims_par_seconde_median"] == pytest.approx(400.0)


def test_agreger_separe_les_workers_et_l_etat_du_pool():
    """Un pool chaud et un pool froid ne mesurent pas la meme chose : le premier
    appel d'une configuration paie la creation des threads. Les melanger dans
    une mediane commune masquerait ce cout."""
    agr = agreger([
        _m(worker_count=1),
        _m(worker_count=4),
        _m(worker_count=4, pool_etat="chaud"),
    ])

    assert set(agr) == {
        ("depart", "mcts_search", 0, 1, 1, "froid"),
        ("depart", "mcts_search", 0, 1, 4, "froid"),
        ("depart", "mcts_search", 0, 1, 4, "chaud"),
    }


def test_agreger_separe_les_profondeurs_de_cache():
    agr = agreger([
        _m(cache_history_depth=0),
        _m(cache_history_depth=1),
    ])

    assert set(agr) == {
        ("depart", "mcts_search", 0, 0, 1, "froid"),
        ("depart", "mcts_search", 0, 1, 1, "froid"),
    }


def test_agreger_prend_la_mediane_des_ratios_et_non_le_ratio_des_medianes():
    agr = agreger([
        _m(tt_hits=10, tt_misses=10, tt_position_matches=20,
           tt_rule50_rejects=10),
        _m(tt_hits=0, tt_misses=100, tt_position_matches=10,
           tt_history_rejects=10),
    ])[("depart", "mcts_search", 0, 1, 1, "froid")]

    assert agr["tt_position_match_rate_median"] == pytest.approx(0.55)
    assert agr["tt_rule50_reject_rate_median"] == pytest.approx(0.25)
    assert agr["tt_context_reject_rate_median"] == 0.0
    assert agr["tt_history_reject_rate_median"] == pytest.approx(0.05)


def test_agreger_donne_mediane_et_etendue():
    """L'etendue est la raison d'etre des passages repetes : sans elle, on ne
    sait pas distinguer un gain de 5 pour cent d'un bruit de mesure."""
    mesures = [_m(duree_s=1.0), _m(duree_s=2.0), _m(duree_s=4.0)]

    a = agreger(mesures)[("depart", "mcts_search", 0, 1, 1, "froid")]

    assert a["sims_par_seconde_median"] == pytest.approx(200.0)
    assert a["sims_par_seconde_min"] == pytest.approx(100.0)
    assert a["sims_par_seconde_max"] == pytest.approx(400.0)


def test_agreger_supporte_une_liste_vide():
    assert agreger([]) == {}


from dev_tools.search_bench import format_report

CONTEXTE = {
    "modele": "iter316.onnx",
    "iteration": 316,
    "global_step": 19415,
    "passages": 5,
    "simulations": 400,
    "c_puct": 1.4,
}


def _agr():
    return agreger([_m(duree_s=1.0), _m(duree_s=1.1)])


def test_format_report_contient_le_contexte():
    texte = format_report(_agr(), CONTEXTE, [])

    assert "iter316.onnx" in texte
    assert "19415" in texte
    assert "simulations par seconde" in texte


def test_format_report_affiche_batch_et_remplissage():
    texte = format_report(
        agreger([_m(batch_size=32, nn_calls=320, nn_batches=10)]),
        CONTEXTE, [])

    assert "batch" in texte.lower()
    assert "32.0" in texte


def test_format_report_affiche_les_chronometrages_par_phase():
    timing = {
        "wall_ns": 100_000_000,
        "selection_ns": 10_000_000,
        "tensor_key_ns": 20_000_000,
        "tt_probe_store_ns": 30_000_000,
        "tt_wait_ns": 0,
        "board_copy_ns": 0,
        "batch_assembly_ns": 1_000_000,
        "evaluator_ns": 40_000_000,
        "expansion_ns": 5_000_000,
        "backup_ns": 4_000_000,
        "worker_wait_ns": 0,
    }

    texte = format_report(agreger([_m(timing=timing)]), CONTEXTE, [])

    assert "Chronometrages" in texte
    assert "evaluateur" in texte
    assert "40.000" in texte
    assert "selection" in texte


def test_rapport_affiche_les_rejets_semantiques():
    texte = format_report(agreger([_m(
        cache_history_depth=1,
        tt_hits=10,
        tt_misses=20,
        tt_position_matches=20,
        tt_rule50_rejects=3,
        tt_context_rejects=2,
        tt_history_rejects=5,
    )]), CONTEXTE, [])

    assert "h1" in texte
    assert "match position" in texte
    assert "50 coups" in texte
    assert "historique" in texte
    assert "66.7 %" in texte
    assert "10.0 %" in texte
    assert "6.7 %" in texte
    assert "16.7 %" in texte


def test_les_mesures_transmettent_la_profondeur_au_mcts(monkeypatch):
    import dev_tools.search_bench as search_bench

    constructions = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            constructions.append(depth)

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def mcts_search(self, *args):
            pass

        def step_analysis(self, *args):
            pass

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, cache_history_depth=3)
    search_bench.mesurer_step_analysis(
        object(), "fen", "position", 8, cache_history_depth=7)

    assert constructions == [3, 7]


def test_les_mesures_transmettent_batch_et_workers_au_mcts(monkeypatch):
    import dev_tools.search_bench as search_bench

    appels = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def mcts_search(self, board, simulations, c_puct, bruit,
                        batch_size, worker_count):
            appels.append(("mcts_search", batch_size, worker_count))
            return [0.0] * 4672

        def step_analysis(self, board, simulations, c_puct, batch_size,
                          worker_count):
            appels.append(("step_analysis", batch_size, worker_count))

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    mesure = search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, batch_size=8, worker_count=4)
    analyse = search_bench.mesurer_step_analysis(
        object(), "fen", "position", 8, batch_size=8, worker_count=4)

    assert appels == [("mcts_search", 8, 4), ("step_analysis", 8, 4)]
    assert mesure.worker_count == 4
    assert analyse.worker_count == 4
    assert mesure.batch_size == 8


def test_les_mesures_transmettent_le_lot_fixe(monkeypatch):
    import dev_tools.search_bench as search_bench

    appels = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def set_fixed_batch(self, enabled):
            appels.append(enabled)

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def mcts_search(self, *args):
            pass

        def step_analysis(self, *args):
            pass

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, fixed_batch=True)
    assert appels == [True]

    # Le defaut est reapplique explicitement : un MCTS reutilise ne doit pas
    # garder le lot fixe d'une configuration precedente.
    search_bench.mesurer_mcts_search(object(), "fen", "position", 8)
    assert appels == [True, False]


def test_les_mesures_transmettent_le_reglage_de_divergence(monkeypatch):
    import dev_tools.search_bench as search_bench

    appels = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def set_tuning(self, virtual_loss, fpu_reduction,
                       collision_attempt_factor):
            appels.append((virtual_loss, fpu_reduction,
                           collision_attempt_factor))

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def mcts_search(self, *args):
            pass

        def step_analysis(self, *args):
            pass

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0, leaf_collisions=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, virtual_loss=2, fpu_reduction=0.5,
        collision_attempts=8)
    assert appels == [(2, 0.5, 8)]

    # Les defauts sont reappliques explicitement : le tuning precedent ne
    # survit pas a la configuration suivante.
    search_bench.mesurer_mcts_search(object(), "fen", "position", 8)
    assert appels == [(2, 0.5, 8), (1, 0.30, 4)]


def test_les_mesures_decoupent_en_tranches(monkeypatch):
    import dev_tools.search_bench as search_bench

    appels = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def step_analysis(self, board, simulations, c_puct, batch_size,
                          worker_count):
            appels.append(simulations)

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0, leaf_collisions=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    search_bench.mesurer_step_analysis(
        object(), "fen", "position", 700, slices=64)

    assert appels == [64] * 10 + [60]


def test_la_mesure_capture_les_timings_et_la_taille_tt(monkeypatch):
    import dev_tools.search_bench as search_bench

    constructions = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            constructions.append((taille_tt, depth))

        def reset_counters(self):
            pass

        def set_timing_enabled(self, enabled):
            assert enabled is True

        def mcts_search(self, *args):
            pass

        def get_last_timing(self):
            return type("T", (), dict(
                wall_ns=101, selection_ns=11, tensor_key_ns=12,
                tt_probe_store_ns=13, tt_wait_ns=14, board_copy_ns=15,
                batch_assembly_ns=16, evaluator_ns=17, expansion_ns=18,
                backup_ns=19, worker_wait_ns=20))()

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    mesure = search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, cache_history_depth=0,
        tt_size=123, timings=True, repetition=7)

    assert constructions == [(123, 0)]
    assert mesure.repetition == 7
    assert mesure.timing["wall_ns"] == 101
    assert mesure.timing["evaluator_ns"] == 17


def test_configurer_mcts_reapplique_les_defauts():
    """Un MCTS reutilise ne doit garder ni le lot fixe, ni le tuning, ni le
    chronometrage d'une configuration precedente."""
    import dev_tools.search_bench as search_bench

    journal = []

    class FauxMCTS:
        def set_fixed_batch(self, enabled):
            journal.append(("fixed_batch", enabled))

        def set_tuning(self, virtual_loss, fpu_reduction, tentatives):
            journal.append(("tuning", virtual_loss, fpu_reduction, tentatives))

        def set_timing_enabled(self, enabled):
            journal.append(("timing", enabled))

    mcts = FauxMCTS()
    search_bench.configurer_mcts(mcts, search_bench.ReglagesMCTS(
        fixed_batch=True, virtual_loss=3, fpu_reduction=0.5,
        collision_attempts=8, timings=True))
    search_bench.configurer_mcts(mcts, search_bench.ReglagesMCTS())

    assert journal == [
        ("fixed_batch", True), ("tuning", 3, 0.5, 8), ("timing", True),
        ("fixed_batch", False), ("tuning", 1, 0.30, 4), ("timing", False),
    ]


class _Rapport:
    def __init__(self, nodes=10, max_depth=3, violations=0, messages=(),
                 en_vol=0, pending=0):
        self.nodes = nodes
        self.max_depth = max_depth
        self.violations = violations
        self.messages = list(messages)
        self.en_vol = en_vol
        self.pending = pending


class _FauxMCTSInvariants:
    """Faux MCTS du mode invariants : journalise setters et tranches."""

    instances = []
    rapports_preprogrammes = []

    def __init__(self, evaluateur, taille_tt, depth):
        self.depth = depth
        self.journal = []
        self.recherches = []
        self.inspections = 0
        self.rapports = list(_FauxMCTSInvariants.rapports_preprogrammes)
        self.dernier_batch = None
        self.dernier_workers = None
        _FauxMCTSInvariants.instances.append(self)

    def set_fixed_batch(self, enabled):
        self.journal.append(("fixed_batch", enabled))

    def set_tuning(self, virtual_loss, fpu_reduction, tentatives):
        self.journal.append(("tuning", virtual_loss, fpu_reduction, tentatives))

    def set_timing_enabled(self, enabled):
        self.journal.append(("timing", enabled))

    def reset_analysis(self):
        self.journal.append(("reset_analysis",))

    def mcts_search(self, *args):
        pass

    def step_analysis(self, board, simulations, c_puct, batch_size,
                      worker_count):
        self.journal.append(("step_analysis", simulations))
        self.recherches.append(simulations)
        self.dernier_batch = batch_size
        self.dernier_workers = worker_count

    def inspect_tree(self):
        self.inspections += 1
        if self.rapports:
            return self.rapports.pop(0)
        return _Rapport()


def _preparer_invariants(monkeypatch, tmp_path, argv):
    """Monte un main() en mode invariants, sans modele ni GPU."""
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    _FauxMCTSInvariants.instances = []
    _FauxMCTSInvariants.rapports_preprogrammes = []
    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS",
                        _FauxMCTSInvariants)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), *argv])
    return model


def test_les_invariants_recoivent_les_reglages_demandes(monkeypatch, tmp_path):
    """Le mode invariants doit verifier la configuration annoncee, pas les
    defauts : setters avant la recherche, batch et workers reels."""
    import json
    import dev_tools.search_bench as search_bench

    brut = tmp_path / "invariants.json"
    _preparer_invariants(monkeypatch, tmp_path, [
        "--invariants", "--simulations", "8", "--batch-sizes", "8",
        "--worker-counts", "4", "--fixed-batch", "--virtual-loss", "3",
        "--fpu", "0.5", "--collision-attempts", "8",
        "--out-json", str(brut), "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert search_bench.main() == 0

    # La premiere instance est la recherche de rodage, sans configuration.
    mcts = _FauxMCTSInvariants.instances[-1]
    assert mcts.journal == [
        ("fixed_batch", True), ("tuning", 3, 0.5, 8), ("timing", False),
        ("reset_analysis",), ("step_analysis", 8),
    ]
    assert mcts.dernier_batch == 8
    assert mcts.dernier_workers == 4
    donnees = json.loads(brut.read_text(encoding="utf-8"))
    assert donnees["contexte"]["fixed_batch"] is True
    assert donnees["contexte"]["virtual_loss"] == 3
    assert donnees["contexte"]["fpu_reduction"] == 0.5
    assert donnees["contexte"]["collision_attempts"] == 8
    assert donnees["mesures"] == []
    assert donnees["invariants"][0]["violations"] == 0
    assert donnees["invariants"][0]["tranches"] == 1
    assert donnees["invariants"][0]["en_vol"] == 0
    assert donnees["invariants"][0]["pending"] == 0


def test_les_invariants_controllent_chaque_tranche(monkeypatch, tmp_path):
    import dev_tools.search_bench as search_bench

    _preparer_invariants(monkeypatch, tmp_path, [
        "--invariants", "--simulations", "17", "--slices", "8",
        "--batch-sizes", "8", "--worker-counts", "1",
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert search_bench.main() == 0

    mcts = _FauxMCTSInvariants.instances[-1]
    assert mcts.recherches == [8, 8, 1]
    assert mcts.inspections == 3


def test_les_invariants_echouent_avec_un_code_non_nul(monkeypatch, tmp_path,
                                                      capsys):
    import dev_tools.search_bench as search_bench

    _preparer_invariants(monkeypatch, tmp_path, [
        "--invariants", "--simulations", "8", "--batch-sizes", "8",
        "--worker-counts", "1",
        "--out-rapport", str(tmp_path / "r.md"),
    ])
    _FauxMCTSInvariants.rapports_preprogrammes = [
        _Rapport(violations=2,
                 messages=["noeud en vol au repos, n_in_flight 1"],
                 en_vol=1, pending=1)]

    assert search_bench.main() == 1
    sortie = capsys.readouterr().out
    assert "2 VIOLATIONS" in sortie
    assert "en vol 1" in sortie
    assert "Pending 1" in sortie

    _FauxMCTSInvariants.rapports_preprogrammes = []
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model",
        str(tmp_path / "model.onnx"), "--invariants", "--simulations", "8",
        "--batch-sizes", "8", "--worker-counts", "1",
        "--out-rapport", str(tmp_path / "propre.md"),
    ])
    assert search_bench.main() == 0


def test_main_ecrit_les_mesures_individuelles_en_json(
        monkeypatch, tmp_path):
    import json
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    rapport = tmp_path / "rapport.md"
    brut = tmp_path / "mesures.json"
    appels = []

    class Chauffe:
        def __init__(self, evaluateur, taille_tt, depth):
            assert taille_tt == 123

        def mcts_search(self, *args):
            pass

    def mesurer_recherche(*args, **kwargs):
        appels.append(("mcts_search", kwargs["repetition"]))
        return _m(cache_history_depth=0, repetition=kwargs["repetition"],
                  timing=_timing(101))

    def mesurer_analyse(*args, **kwargs):
        appels.append(("step_analysis", kwargs["repetition"]))
        return _m(chemin="step_analysis", cache_history_depth=0,
                  repetition=kwargs["repetition"],
                  timing=_timing(102))

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", Chauffe)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(search_bench, "mesurer_mcts_search", mesurer_recherche)
    monkeypatch.setattr(search_bench, "mesurer_step_analysis", mesurer_analyse)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "1",
        "--repetitions", "2", "--simulations", "8",
        "--cache-history-depths", "0", "--tt-size", "123",
        "--timings", "--out-json", str(brut),
        "--out-rapport", str(rapport),
    ])

    assert search_bench.main() == 0
    assert appels == [
        ("mcts_search", 0), ("step_analysis", 0),
        ("mcts_search", 1), ("step_analysis", 1),
    ]
    donnees = json.loads(brut.read_text(encoding="utf-8"))
    assert donnees["contexte"]["tt_size"] == 123
    assert donnees["contexte"]["timings"] is True
    assert donnees["contexte"]["worker_counts"] == [1]
    assert donnees["contexte"]["warmup_pool"] is False
    assert len(donnees["mesures"]) == 4
    assert donnees["mesures"][0]["timing"]["wall_ns"] == 101
    assert donnees["mesures"][0]["worker_count"] == 1
    assert donnees["mesures"][0]["pool_etat"] == "froid"


def test_main_balaie_toutes_les_profondeurs_demandees(
        monkeypatch, tmp_path):
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    rapport = tmp_path / "rapport.md"
    appels = []

    class Chauffe:
        def __init__(self, evaluateur, taille_tt, depth):
            appels.append(("chauffe", depth))

        def mcts_search(self, *args):
            pass

    def mesurer_recherche(*args, **kwargs):
        depth = args[-1]
        appels.append(("mcts_search", depth))
        return _m(cache_history_depth=depth)

    def mesurer_analyse(*args, **kwargs):
        depth = args[-1]
        appels.append(("step_analysis", depth))
        return _m(chemin="step_analysis", cache_history_depth=depth)

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", Chauffe)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(search_bench, "mesurer_mcts_search", mesurer_recherche)
    monkeypatch.setattr(search_bench, "mesurer_step_analysis", mesurer_analyse)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "1",
        "--simulations", "8", "--cache-history-depths", "0", "3",
        "--out-rapport", str(rapport),
    ])

    assert search_bench.main() == 0
    assert appels == [
        ("chauffe", 0),
        ("mcts_search", 0), ("step_analysis", 0),
        ("mcts_search", 3), ("step_analysis", 3),
    ]


def test_main_alterne_l_ordre_des_workers_entre_passages(monkeypatch, tmp_path):
    """Passage 1 puis passage 2 dans l'ordre inverse : une derive thermique ne
    doit pas se confondre avec l'effet du nombre de workers."""
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    appels = []

    class Chauffe:
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def mcts_search(self, *args):
            pass

    def mesurer_recherche(*args, **kwargs):
        appels.append(("mcts_search", kwargs["worker_count"]))
        return _m(worker_count=kwargs["worker_count"])

    def mesurer_analyse(*args, **kwargs):
        appels.append(("step_analysis", kwargs["worker_count"]))
        return _m(chemin="step_analysis", worker_count=kwargs["worker_count"])

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", Chauffe)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(search_bench, "mesurer_mcts_search", mesurer_recherche)
    monkeypatch.setattr(search_bench, "mesurer_step_analysis", mesurer_analyse)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "2",
        "--repetitions", "1", "--simulations", "8", "--batch-sizes", "8",
        "--worker-counts", "1", "2",
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert search_bench.main() == 0
    assert appels == [
        ("mcts_search", 1), ("step_analysis", 1),
        ("mcts_search", 2), ("step_analysis", 2),
        ("mcts_search", 2), ("step_analysis", 2),
        ("mcts_search", 1), ("step_analysis", 1),
    ]


def test_main_reutilise_le_pool_chaud_et_efface_la_tt(monkeypatch, tmp_path):
    """Le pool chaud doit garder ses threads et oublier la TT entre deux
    mesures : sans clear, la seconde mesure serait un banc de hits de table,
    pas un banc de recherche."""
    import json
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    brut = tmp_path / "mesures.json"
    instances = []

    class FauxMCTS(FauxMCTSSetters):
        def __init__(self, evaluateur, taille_tt, depth):
            instances.append(self)
            self.clears = 0

        def set_timing_enabled(self, enabled):
            pass

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def clear_evaluation_cache(self):
            self.clears += 1

        def mcts_search(self, *args):
            return [0.0] * 4672

        def step_analysis(self, *args):
            pass

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "1",
        "--repetitions", "2", "--simulations", "8",
        "--cache-history-depths", "0", "--warmup-pool",
        "--out-json", str(brut), "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert search_bench.main() == 0

    # chauffe, puis un objet persistant par chemin.
    assert len(instances) == 3
    assert [m.clears for m in instances[1:]] == [1, 1]
    donnees = json.loads(brut.read_text(encoding="utf-8"))
    assert donnees["contexte"]["warmup_pool"] is True
    assert donnees["contexte"]["worker_counts"] == [1]
    etats = {(m["chemin"], m["pool_etat"]) for m in donnees["mesures"]}
    assert etats == {
        ("mcts_search", "froid"), ("mcts_search", "chaud"),
        ("step_analysis", "froid"), ("step_analysis", "chaud"),
    }


def test_le_defaut_de_profondeur_est_h0(monkeypatch, tmp_path):
    """La CLI harmonise son defaut sur h0, comme le defaut C++."""
    import json
    import dev_tools.search_bench as search_bench
    import evaluation.puzzle_bench as puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    brut = tmp_path / "mesures.json"
    profondeurs = []

    class Chauffe:
        def __init__(self, evaluateur, taille_tt, depth):
            pass

        def mcts_search(self, *args):
            pass

    def mesurer_recherche(*args, **kwargs):
        profondeurs.append(args[6])
        return _m(cache_history_depth=args[6])

    def mesurer_analyse(*args, **kwargs):
        profondeurs.append(args[6])
        return _m(chemin="step_analysis", cache_history_depth=args[6])

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", Chauffe)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(search_bench, "mesurer_mcts_search", mesurer_recherche)
    monkeypatch.setattr(search_bench, "mesurer_step_analysis", mesurer_analyse)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "1",
        "--simulations", "8", "--out-json", str(brut),
        "--out-rapport", str(tmp_path / "r.md"),
    ])

    assert search_bench.main() == 0
    assert profondeurs == [0, 0]
    donnees = json.loads(brut.read_text(encoding="utf-8"))
    assert donnees["contexte"]["cache_history_depths"] == [0]


def test_format_report_n_ecrit_jamais_noeuds_par_seconde():
    """Le perft mesure 1,6 million de noeuds par seconde, la recherche 295
    simulations par seconde : confondre les deux serait une erreur d'un facteur
    5000."""
    texte = format_report(_agr(), CONTEXTE, []).lower()

    assert "noeuds par seconde" not in texte
    assert "nps" not in texte


def test_format_report_ne_contient_pas_de_tiret_cadratin():
    assert "—" not in format_report(_agr(), CONTEXTE, [])


def test_format_report_signale_les_violations():
    invariants = [("depart", 1, 0, 8, 120, 5, 3,
                   ["enfant duplique, move_idx 42"])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" in texte
    assert "3" in texte


def test_format_report_associe_les_invariants_a_leur_batch():
    invariants = [("depart", 1, 32, 4, 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "| depart | h1 | 32 | 4 | 120 | 5 | 0 |" in texte


def test_format_report_affiche_workers_et_etat_du_pool():
    texte = format_report(
        agreger([_m(worker_count=4, pool_etat="chaud")]), CONTEXTE, [])

    assert "workers" in texte.lower()
    assert "chaud" in texte
    assert "4" in texte


def test_format_report_rappelle_que_le_total_est_le_temps_mur():
    """Les phases sont cumulees par thread : leur somme peut depasser le mur.
    Le rapport doit dire lequel des deux est la reference."""
    timing = _timing(100_000_000)
    timing["evaluator_ns"] = 400_000_000

    texte = format_report(agreger([_m(timing=timing)]), CONTEXTE, [])

    assert "temps mur" in texte
    assert "100.000" in texte


def test_format_report_est_muet_quand_aucune_violation():
    invariants = [("depart", 1, 0, 8, 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" not in texte
    assert "aucune violation" in texte.lower()


def test_format_report_produit_des_tables_markdown_valides():
    """Garde fou repris du banc de puzzles : un separateur dont le nombre de
    cellules ne correspond pas a l'en-tete casse le rendu en silence."""
    lignes = format_report(_agr(), CONTEXTE, []).split("\n")
    separateurs = [i for i, l in enumerate(lignes)
                   if set(l) <= set("|-") and "-" in l]

    assert separateurs
    for i in separateurs:
        cols = lignes[i].count("|") - 1
        assert lignes[i - 1].count("|") - 1 == cols, lignes[i - 1]
        j = i + 1
        while j < len(lignes) and lignes[j].startswith("|"):
            assert lignes[j].count("|") - 1 == cols, lignes[j]
            j += 1
