"""Transmission du bruit rapide et du ratio de réutilisation.

Tests locaux : argparse, pipeline avec dépendances factices, branche
diagnostic de generate_games. Aucun GPU, aucun moteur, aucun modèle.
Contrat : docs/superpowers/specs/2026-10-05-selfplay-exploration-ratio-design.md.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import training.run_selftrain as run_selftrain
import training.train_self_play as train_self_play
from test_position_bench_training import JournalWandb, _appels_vides
from test_selfplay_diag import timing_factice


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

STATS_VIDES = {
    "checkmates": 2, "stalemates": 0, "repetition": 0,
    "50_moves": 0, "insuff_mat": 0, "max_moves": 0,
}


def _dependances(journal, nouvelles_positions, tirages_reels=None, appels=None):
    """Dépendances de pipeline avec génération et entraînement instrumentés."""
    from test_position_bench_training import _dependances as base

    deps = base(_appels_vides(), journal)

    def generer(onnx, games, concurrent, slow, fast, ratio, tt_size=0,
                diagnostics=0, fast_dirichlet_noise=True):
        if appels is not None:
            appels.append({"etape": "generation", "diagnostics": diagnostics,
                           "fast_dirichlet_noise": fast_dirichlet_noise})
        return [0] * nouvelles_positions, 100.0, dict(STATS_VIDES), None

    def entrainer(model, optimizer, scaler, device, folder, lr, **kwargs):
        if appels is not None:
            appels.append({"etape": "entrainement",
                           "samples_per_epoch": kwargs["samples_per_epoch"]})
        cible = kwargs.get("training_stats")
        if cible is not None:
            cible["samples_seen"] = (
                kwargs["samples_per_epoch"] if tirages_reels is None
                else tirages_reels)
        return kwargs["global_step"] + 1

    deps["generer_parties"] = generer
    deps["entrainer"] = entrainer
    return deps


def _etape(appels, nom):
    return [a for a in appels if a["etape"] == nom]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_defauts_historiques(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py"])
    args = run_selftrain.analyser_arguments()
    assert args.sampling_ratio == 14.0
    assert args.fast_dirichlet_noise is True

    args = train_self_play.analyser_arguments([])
    assert args.sampling_ratio == 14.0
    assert args.fast_dirichlet_noise is True


def test_cli_options_de_campagne(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_selftrain.py", "--sampling-ratio", "6",
        "--no-fast-dirichlet-noise"])
    args = run_selftrain.analyser_arguments()
    assert args.sampling_ratio == 6.0
    assert args.fast_dirichlet_noise is False

    args = train_self_play.analyser_arguments(
        ["--sampling-ratio", "6", "--no-fast-dirichlet-noise"])
    assert args.sampling_ratio == 6.0
    assert args.fast_dirichlet_noise is False


def test_cli_option_positive_retablit_le_bruit(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_selftrain.py", "--no-fast-dirichlet-noise",
        "--fast-dirichlet-noise"])
    assert run_selftrain.analyser_arguments().fast_dirichlet_noise is True

    args = train_self_play.analyser_arguments(
        ["--no-fast-dirichlet-noise", "--fast-dirichlet-noise"])
    assert args.fast_dirichlet_noise is True


def test_run_selftrain_transmet_au_pipeline(monkeypatch):
    recu = {}

    def faux_pipeline(**kwargs):
        recu.update(kwargs)

    monkeypatch.setattr(train_self_play, "pipeline", faux_pipeline)
    monkeypatch.setattr(sys, "argv", [
        "run_selftrain.py", "--sampling-ratio", "6",
        "--no-fast-dirichlet-noise", "--wandb-mode", "online"])
    run_selftrain.main()
    assert recu["target_sampling_ratio"] == 6.0
    assert recu["fast_dirichlet_noise"] is False


# ---------------------------------------------------------------------------
# Pipeline : budget, configuration et ratio effectif
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio, budget", [(6.0, 120000), (14.0, 280000)])
def test_ratio_transmet_le_budget_de_tirages(ratio, budget):
    journal = JournalWandb()
    appels = []
    train_self_play.pipeline(
        num_iterations=1, eval_every=0, target_sampling_ratio=ratio,
        fast_dirichlet_noise=False,
        dependances=_dependances(journal, 20000, appels=appels))

    generation = _etape(appels, "generation")
    entrainement = _etape(appels, "entrainement")
    assert len(generation) == 1 and len(entrainement) == 1
    assert entrainement[0]["samples_per_epoch"] == budget
    assert generation[0]["fast_dirichlet_noise"] is False
    config = journal.initialisations[0]["config"]
    assert config["target_sampling_ratio"] == ratio
    assert config["fast_dirichlet_noise"] is False


def test_ratio_effectif_utilise_le_compteur_reel():
    journal = JournalWandb()
    appels = []
    train_self_play.pipeline(
        num_iterations=1, eval_every=0, target_sampling_ratio=6.0,
        fast_dirichlet_noise=True,
        dependances=_dependances(journal, 20000, tirages_reels=119000,
                                 appels=appels))

    point = journal.points[0][0]
    assert point["train/replay_ratio_target"] == 6.0
    assert point["train/replay_ratio_effective"] == pytest.approx(5.95)


def test_sans_nouvelles_positions_le_ratio_effectif_est_omis():
    journal = JournalWandb()
    appels = []
    train_self_play.pipeline(
        num_iterations=1, eval_every=0, target_sampling_ratio=6.0,
        fast_dirichlet_noise=False,
        dependances=_dependances(journal, 0, appels=appels))

    point = journal.points[0][0]
    assert point["train/replay_ratio_target"] == 6.0
    assert "train/replay_ratio_effective" not in point
    assert not any(
        isinstance(valeur, float) and valeur != valeur  # NaN
        for valeur in point.values())
    assert _etape(appels, "entrainement")[0]["samples_per_epoch"] == 0


@pytest.mark.parametrize("mauvais", [0.0, -1.0, float("nan"), float("inf")])
def test_ratio_invalide_rejete_avant_generation(mauvais):
    with pytest.raises(ValueError, match="target_sampling_ratio"):
        train_self_play.pipeline(
            num_iterations=1, eval_every=0, target_sampling_ratio=mauvais,
            dependances={"cuda_disponible": lambda: True})


# ---------------------------------------------------------------------------
# generate_games : branche normale et branches diagnostic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diagnostics", [0, 1, 2])
def test_generate_games_transmet_le_bruit_par_nom(monkeypatch, diagnostics):
    appels = []

    def faux_normal(*args, **kwargs):
        appels.append(("normal", kwargs))
        return [SimpleNamespace(total_real_moves=30)]

    def faux_diagnostic(*args, **kwargs):
        appels.append(("diagnostic", kwargs))
        return (
            [SimpleNamespace(total_real_moves=30)],
            SimpleNamespace(new_plies=3, games_completed=1, replayed_plies=0),
            timing_factice(),
        )

    monkeypatch.setattr(train_self_play.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(train_self_play.chess_engine,
                        "generate_self_play_games", faux_normal)
    monkeypatch.setattr(train_self_play.chess_engine,
                        "generate_self_play_games_with_diagnostics",
                        faux_diagnostic)
    monkeypatch.setattr(train_self_play, "convert_game_results",
                        lambda resultats: ([], dict(STATS_VIDES)))
    monkeypatch.setattr(train_self_play, "mesurer_memoire", lambda: {})

    train_self_play.generate_games(
        "faux.onnx", 1, 1, 2, 1, 0.5, diagnostics=diagnostics,
        fast_dirichlet_noise=False)

    assert len(appels) == 1
    nom, kwargs = appels[0]
    assert kwargs == {"fast_dirichlet_noise": False}
    if diagnostics == 0:
        assert nom == "normal"
    else:
        assert nom == "diagnostic"
