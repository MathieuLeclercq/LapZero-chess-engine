"""Tests de l'integration du banc de positions a la boucle self-play.

Generation, entrainement, export, sauvegarde, evaluation et W&B sont des
doubles : aucun GPU, aucun modele, aucun dataset externe.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import training.run_selftrain as run_selftrain
import training.train_self_play as train_self_play
from training.train_self_play import pipeline


class JournalWandb:
    def __init__(self):
        self.initialisations = []
        self.points = []
        self.fins = 0

    def init(self, **kwargs):
        self.initialisations.append(kwargs)

    def log(self, donnees, **kwargs):
        self.points.append((donnees, kwargs))

    def finish(self):
        self.fins += 1


class Modele:
    def to(self, device):
        return self

    def parameters(self):
        return []

    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


class Optimiseur:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]

    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


class Scaler:
    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


def _appels_vides():
    return {cle: [] for cle in (
        "generation", "entrainement", "export", "sauvegarde", "evaluation")}


def _dependances(appels, journal):
    def generer(onnx, games, concurrent, slow, fast, ratio, tt_size=0,
                diagnostics=0, fast_dirichlet_noise=True):
        appels["generation"].append({"onnx": str(onnx), "games": games})
        stats = {"checkmates": 2, "stalemates": 0, "repetition": 0,
                 "50_moves": 0, "insuff_mat": 0, "max_moves": 0}
        return [], 100.0, stats, None

    def entrainer(model, optimizer, scaler, device, folder, lr, **kwargs):
        appels["entrainement"].append(dict(kwargs))
        return kwargs["global_step"] + 1

    def exporter(model, chemin, device):
        appels["export"].append(str(chemin))

    def sauver(chemin, charge):
        appels["sauvegarde"].append((str(chemin), charge["iteration"]))

    def evaluer(onnx, banc, sortie, config, **kwargs):
        appels["evaluation"].append({
            "onnx": str(onnx), "banc": str(banc), "config": config,
            **kwargs})
        return {
            "status": "ok", "duration_s": 310.0,
            "completed_positions": 10000, "completed_search_positions": 256,
            "dataset_version": "v1", "protocol_id": "p" * 64,
            "dataset_sha256": "d" * 64, "model_sha256": "m" * 64,
            "metrics": {}, "comparison": None, "result_path": "x",
            "error": None, "target_s": 300.0, "over_target": True,
        }

    def charger_banc(chemin):
        assert Path(chemin).name == "banc"
        return ({"dataset_sha256": "d" * 64, "dataset_version": "v1",
                 "search_ids": [], "counts": {}}, [])

    def ecrire_buffer(new_data, folder, plafond):
        return 0

    def charger_checkpoint(chemin):
        return {"model_state_dict": {}, "optimizer_state_dict": {},
                "scaler_state_dict": {}, "iteration": 3, "global_step": 100}

    return {
        "generer_parties": generer,
        "entrainer": entrainer,
        "exporter": exporter,
        "sauver_checkpoint": sauver,
        "evaluer": evaluer,
        "charger_banc": charger_banc,
        "ecrire_buffer": ecrire_buffer,
        "charger_checkpoint": charger_checkpoint,
        "creer_modele": lambda: Modele(),
        "creer_optimiseur": lambda parametres: Optimiseur(),
        "creer_scaler": lambda: Scaler(),
        "cuda_disponible": lambda: True,
        "chercher_precedent": lambda *arguments: None,
        "protocol_id": lambda config: "p" * 64,
        "wandb": journal,
    }


def test_memoire_aux_frontieres_et_absente_sans_diagnostic(monkeypatch, tmp_path,
                                                            capsys):
    import monitoring.moniteur_cpu as moniteur_cpu
    monkeypatch.setattr(moniteur_cpu, "creer_echantillonneur", lambda: None)
    for mode in (0, 1):
        appels = _appels_vides()
        journal = JournalWandb()
        deps = _dependances(appels, journal)
        generer = deps["generer_parties"]
        def generer_avec_memoire(*args, **kwargs):
            resultats = generer(*args, **kwargs)
            if kwargs["diagnostics"] == 0:
                return resultats
            from training.selfplay_diag import resumer_passage
            from test_selfplay_diag import timing_factice
            diagnostic = resumer_passage(timing_factice(), {}, {})
            diagnostic["memoire"] = {
                "apres_nettoyage_selfplay": {"privee_mio": 777.0}}
            return (*resultats[:3], diagnostic)
        deps["generer_parties"] = generer_avec_memoire
        mesures = []
        def mesurer():
            mesures.append(len(mesures) + 1)
            return {"privee_mio": float(mesures[-1]), "rss_mio": 10.0}
        deps["mesurer_memoire"] = mesurer
        pipeline(num_iterations=1, eval_every=1, selfplay_diagnostics=mode,
                 position_bench_path=tmp_path / "banc",
                 eval_output_dir=tmp_path / "resultats", dependances=deps)
        points = [d for d, _ in journal.points if d]
        if mode == 0:
            assert mesures == []
            assert not any(k.startswith("memoire/") for k in points[0])
        else:
            assert mesures == [1, 2, 3, 4, 5, 6]
            assert points[0]["memoire/apres_nettoyage_selfplay/privee_mio"] == 777.0
            assert points[0]["memoire/avant_selfplay/privee_mio"] == 1.0
            assert points[0]["memoire/apres_liberation_donnees/privee_mio"] == 2.0
            assert points[0]["memoire/apres_entrainement/privee_mio"] == 3.0
            assert points[0]["memoire/apres_sauvegarde_pt/privee_mio"] == 4.0
            assert points[0]["memoire/apres_export_onnx/privee_mio"] == 5.0
            assert points[0]["memoire/apres_evaluation/privee_mio"] == 6.0
            # Les mesures restent dans W&B et ne polluent plus la console.
            assert "memoire" not in capsys.readouterr().out


def test_generation_releve_la_memoire_apres_les_liberations(monkeypatch):
    from types import SimpleNamespace
    from test_selfplay_diag import timing_factice
    mesures = []
    def mesurer():
        mesures.append(len(mesures) + 1)
        return {"privee_mio": float(mesures[-1])}
    monkeypatch.setattr(train_self_play, "mesurer_memoire", mesurer, raising=False)
    monkeypatch.setattr(train_self_play.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    games = [SimpleNamespace(total_real_moves=3)]
    stats = dict(checkmates=1, stalemates=0, repetition=0,
                 insuff_mat=0, max_moves=0, **{"50_moves": 0})
    monkeypatch.setattr(train_self_play, "convert_game_results",
                        lambda results: ([1], stats))
    engine_stats = SimpleNamespace(new_plies=3, games_completed=1, replayed_plies=0)
    monkeypatch.setattr(train_self_play, "run_with_interrupt",
                        lambda *args: (games, engine_stats, timing_factice()))
    _, _, _, diagnostic = train_self_play.generate_games(
        "fake.onnx", 1, 1, 2, 1, 0.5, diagnostics=1)
    assert "memoire" in diagnostic
    assert diagnostic["memoire"] == {
        "apres_creation_evaluateur": {"privee_mio": 1.0},
        "apres_generation_cpp": {"privee_mio": 2.0},
        "apres_conversion": {"privee_mio": 3.0},
        "apres_nettoyage_selfplay": {"privee_mio": 4.0},
    }


def test_selfplay_range_les_modeles_dans_data_meme_depuis_un_autre_dossier(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    appels = _appels_vides()
    pipeline(num_iterations=1, checkpoint_path="faux.pt", eval_every=1,
             position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats",
             dependances=_dependances(appels, JournalWandb()))
    assert Path(appels["sauvegarde"][0][0]).parent == RACINE / "data" / "checkpoints"
    assert all(Path(p).parent == RACINE / "data" / "checkpoints" / "onnx"
               for p in appels["export"])
    assert appels["generation"][0]["onnx"] == appels["export"][0]
    assert appels["evaluation"][0]["onnx"] == appels["export"][-1]
    assert not (tmp_path / "checkpoints").exists()


def test_cadence_globale_apres_reprise():
    iterations = range(3, 9)
    evaluated = [i + 1 for i in iterations if (i + 1) % 4 == 0]

    assert evaluated == [4, 8]


def test_pipeline_evalue_aux_iterations_4_et_8(tmp_path):
    appels = _appels_vides()
    journal = JournalWandb()

    pipeline(num_iterations=6, checkpoint_path="faux.pt", eval_every=4,
             position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats",
             dependances=_dependances(appels, journal))

    assert [appel["iteration"] for appel in appels["evaluation"]] == [4, 8]
    assert [iteration for _, iteration in appels["sauvegarde"]] == [
        4, 5, 6, 7, 8, 9]
    assert appels["evaluation"][0]["onnx"].endswith(
        "iter4_unsupervised.onnx")
    assert appels["evaluation"][1]["onnx"].endswith(
        "iter8_unsupervised.onnx")
    assert appels["evaluation"][0]["config"].search.workers == 8
    assert appels["evaluation"][0]["config"].target_s == 300.0
    # Un seul log par iteration, et il porte le statut du banc les bonnes
    # iterations.
    points = [donnees for donnees, _ in journal.points if donnees]
    assert any("eval/position/status" in point for point in points)
    assert len(appels["generation"]) == 6
    assert len(appels["entrainement"]) == 6


def test_le_precedent_compatible_est_passe_a_l_evaluation(tmp_path):
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())
    dependances["chercher_precedent"] = (
        lambda *arguments: tmp_path / "precedent.json")

    pipeline(num_iterations=4, eval_every=4,
             position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats", dependances=dependances)

    assert len(appels["evaluation"]) == 1
    assert Path(appels["evaluation"][0]["previous"]).name == "precedent.json"


def test_eval_every_zero_desactive_le_banc():
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())

    def charger_interdit(chemin):
        raise AssertionError("le banc ne doit pas etre charge")

    dependances["charger_banc"] = charger_interdit

    pipeline(num_iterations=1, eval_every=0, dependances=dependances)

    assert appels["evaluation"] == []


def test_eval_every_negatif_refuse():
    with pytest.raises(ValueError):
        pipeline(eval_every=-1, dependances={"cuda_disponible": lambda: True})


def test_les_parametres_stockfish_sont_refuses():
    cas: list[dict] = [{"stockfish_path": "x"}, {"stockfish_elo": 2600},
                       {"stockfish_nodes": 200_000},
                       {"num_sim_eval_sf": 700}]
    for parametres in cas:
        with pytest.raises(ValueError, match="Stockfish"):
            pipeline(dependances={"cuda_disponible": lambda: True},
                     **parametres)


def test_eval_stockfish_every_devient_eval_every(tmp_path, capsys):
    appels = _appels_vides()

    pipeline(num_iterations=10, eval_stockfish_every=5,
             eval_every=4, position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats",
             dependances=_dependances(appels, JournalWandb()))

    assert [appel["iteration"] for appel in appels["evaluation"]] == [5, 10]
    assert "eval_stockfish_every" in capsys.readouterr().out


def test_un_banc_absent_arrete_avant_la_generation(tmp_path):
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())
    del dependances["charger_banc"]  # load_dataset reel

    with pytest.raises((FileNotFoundError, ValueError)):
        pipeline(num_iterations=1, eval_every=4,
                 position_bench_path=tmp_path / "absent",
                 eval_output_dir=tmp_path / "resultats",
                 dependances=dependances)

    assert appels["generation"] == []


def test_wandb_mode_disabled_reste_transmis(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    appels = _appels_vides()

    pipeline(num_iterations=1, eval_every=0, wandb_mode="disabled",
             dependances=_dependances(appels, JournalWandb()))

    assert os.environ["WANDB_MODE"] == "disabled"


def test_la_cli_refuse_les_drapeaux_stockfish():
    with pytest.raises(SystemExit) as info:
        train_self_play.principal(["--stockfish", "stockfish.exe"])

    assert "Stockfish" in str(info.value)


def test_run_selftrain_cadence_par_defaut_et_transmission(monkeypatch):
    capture = {}
    monkeypatch.setattr(run_selftrain, "etat_du_buffer",
                        lambda dossier: ([], 0))
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py", "--iterations", "1"])

    def pipeline_capture(**kwargs):
        capture.update(kwargs)

    monkeypatch.setattr(train_self_play, "pipeline", pipeline_capture)

    run_selftrain.main()

    assert capture["eval_every"] == 4
    assert capture["eval_search_workers"] == 8
    assert capture["eval_target_s"] == 300.0
    assert capture["position_bench_path"] is None
    assert capture["eval_output_dir"] is None


def test_run_selftrain_n_expose_plus_les_options_stockfish(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py"])

    args = run_selftrain.analyser_arguments()

    assert not hasattr(args, "stockfish")
    assert not hasattr(args, "stockfish_elo")


@pytest.mark.parametrize("extra,expected", [
    ([], str(RACINE / "data/checkpoints/2026_04_30_09h53_iter436_unsupervised.pt")),
    (["--checkpoint", "chemin_personnel.pt"], "chemin_personnel.pt"),
])
def test_lanceur_respecte_le_checkpoint(monkeypatch, extra, expected):
    capture = {}
    monkeypatch.setattr(run_selftrain, "etat_du_buffer", lambda path: ([], 0))
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py", "--iterations", "1", *extra])
    monkeypatch.setattr(train_self_play, "pipeline", lambda **kw: capture.update(kw))
    run_selftrain.main()
    assert capture["checkpoint_path"] == expected
    assert capture["eval_every"] == 4
