"""Tests du chargeur replay integre a `train_on_shards`.

CPU uniquement : petit modele, journal W&B factice, scaler double.
Contrat : docs/superpowers/specs/2026-10-05-selfplay-loader-design.md (lot 2).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import training.train_self_play as train_self_play
from training.train_self_play import train_on_shards

PLANS = 119
TAILLE_POLICY = 4672
COTE = 8


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class JournalWandb:
    def __init__(self):
        self.points = []

    def log(self, donnees, **kwargs):
        self.points.append((dict(donnees), dict(kwargs)))


class ScalerFactice:
    """Scaler AMP minimal ; `saute=True` simule un pas non applique."""

    def __init__(self, saute=False):
        self.saute = saute
        self.etapes = 0
        self.sauts = 0

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        self.etapes += 1
        if self.saute:
            self.sauts += 1
        else:
            optimizer.step()

    def update(self):
        pass


class ModeleJournaliseur(torch.nn.Module):
    """Modele constant qui enregistre les marqueurs de chaque lot."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(TAILLE_POLICY))
        self.valeur = torch.nn.Parameter(torch.zeros(1))
        self.appels = []

    def forward(self, x):
        n = x.size(0)
        self.appels.append(x[:, 0, 0, 0].tolist())
        return (self.logits.unsqueeze(0).expand(n, -1),
                self.valeur.unsqueeze(0).expand(n, 1))


class ModeleDeuxActions(torch.nn.Module):
    """Logits nuls sur deux actions, tres negatifs ailleurs, value nulle."""

    def __init__(self):
        super().__init__()
        logits = torch.full((TAILLE_POLICY,), -1e9)
        logits[0] = 0.0
        logits[1] = 0.0
        self.logits = torch.nn.Parameter(logits)
        self.valeur = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.size(0)
        return (self.logits.unsqueeze(0).expand(n, -1),
                self.valeur.unsqueeze(0).expand(n, 1))


def ecrire_shard(dossier, nom, marqueurs):
    """Mini-archive d'entrainement : policy one-hot en 0, value nulle."""
    n = len(marqueurs)
    states = np.zeros((n, PLANS, COTE, COTE), dtype=np.float16)
    policies = np.zeros((n, TAILLE_POLICY), dtype=np.float16)
    values = np.zeros((n,), dtype=np.float32)
    for i, marqueur in enumerate(marqueurs):
        states[i].fill(np.float16(marqueur))
        policies[i, 0] = 1.0
    np.savez_compressed(Path(dossier) / nom, states=states, policies=policies,
                        values=values)


def _model_et_optimiseur(lr=0.01):
    modele = ModeleJournaliseur()
    return modele, torch.optim.SGD(modele.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# Lots, tentatives et compteurs
# ---------------------------------------------------------------------------

def test_une_tentative_par_lot_et_compteurs_wandb(monkeypatch, tmp_path):
    for numero in range(4):
        ecrire_shard(tmp_path, f"shard_{numero}.npz",
                     [numero * 10 + 1, numero * 10 + 2, numero * 10 + 3])
    journal = JournalWandb()
    monkeypatch.setattr(train_self_play, "wandb", journal)
    modele, optimiseur = _model_et_optimiseur()
    scaler = ScalerFactice()
    statistiques = {}
    retour = train_on_shards(
        modele, optimiseur, scaler, torch.device("cpu"), tmp_path, 1e-3,
        batch_size=8, global_step=100, samples_per_epoch=19,
        training_stats=statistiques)
    assert retour == 103
    assert statistiques == {"samples_seen": 19}
    assert [len(appel) for appel in modele.appels] == [7, 6, 6]
    assert scaler.etapes == 3
    assert len(journal.points) == 1
    donnees, kwargs = journal.points[0]
    assert donnees["train/samples_seen"] == 19
    assert donnees["train/num_batches"] == 3
    assert donnees["train/batch_size_min"] == 6
    assert donnees["train/batch_size_max"] == 7
    assert donnees["train/global_step"] == 103
    assert math.isfinite(donnees["train/epoch_loss"])
    assert kwargs["step"] == 103


def test_scaler_qui_saute_compte_quand_meme_les_tentatives(monkeypatch, tmp_path):
    ecrire_shard(tmp_path, "shard_0.npz", [1, 2, 3])
    journal = JournalWandb()
    monkeypatch.setattr(train_self_play, "wandb", journal)
    modele, optimiseur = _model_et_optimiseur()
    avant = {cle: valeur.clone() for cle, valeur in modele.state_dict().items()}
    scaler = ScalerFactice(saute=True)
    retour = train_on_shards(
        modele, optimiseur, scaler, torch.device("cpu"), tmp_path, 1e-3,
        batch_size=8, global_step=100, samples_per_epoch=19)
    assert retour == 103
    assert scaler.etapes == 3
    assert scaler.sauts == 3
    apres = modele.state_dict()
    for cle, valeur in avant.items():
        assert torch.equal(valeur, apres[cle])


def test_frontiere_de_fichier_sans_tentative_supplementaire(monkeypatch, tmp_path):
    for numero in range(5):
        ecrire_shard(tmp_path, f"shard_{numero}.npz",
                     [numero * 10 + 1, numero * 10 + 2])
    monkeypatch.setattr(train_self_play, "wandb", JournalWandb())
    modele, optimiseur = _model_et_optimiseur()
    retour = train_on_shards(
        modele, optimiseur, ScalerFactice(), torch.device("cpu"), tmp_path,
        1e-3, batch_size=8, global_step=0, samples_per_epoch=17)
    nombre_lots = math.ceil(17 / 8)
    assert retour == nombre_lots
    assert len(modele.appels) == nombre_lots


# ---------------------------------------------------------------------------
# Ponderation des losses
# ---------------------------------------------------------------------------

def test_loss_ponderee_par_effectif(monkeypatch, tmp_path):
    ecrire_shard(tmp_path, "shard_0.npz", [1, 2, 3])

    def faux_flux(shards, total, batch_size, **kwargs):
        assert total == 5
        cibles = torch.zeros(2, TAILLE_POLICY)
        cibles[:, 0] = 0.5
        cibles[:, 1] = 0.5
        yield (torch.zeros(3, PLANS, COTE, COTE),
               cibles[:1].expand(3, -1).contiguous(),
               torch.zeros(3))
        yield (torch.zeros(2, PLANS, COTE, COTE), cibles, torch.ones(2))

    monkeypatch.setattr(train_self_play, "iter_replay_batches", faux_flux)
    journal = JournalWandb()
    monkeypatch.setattr(train_self_play, "wandb", journal)
    modele = ModeleDeuxActions()
    avant = {cle: valeur.clone() for cle, valeur in modele.state_dict().items()}
    optimiseur = torch.optim.SGD(modele.parameters(), lr=0.0)
    retour = train_on_shards(
        modele, optimiseur, ScalerFactice(), torch.device("cpu"), tmp_path,
        1e-3, batch_size=4, global_step=0, samples_per_epoch=5)
    assert retour == 2
    donnees, _ = journal.points[0]
    assert donnees["train/epoch_policy_loss"] == pytest.approx(
        math.log(2), rel=1e-5)
    assert donnees["train/epoch_value_loss"] == pytest.approx(0.4, rel=1e-5)
    assert donnees["train/epoch_loss"] == pytest.approx(
        math.log(2) + 0.4, rel=1e-5)
    apres = modele.state_dict()
    for cle, valeur in avant.items():
        assert torch.equal(valeur, apres[cle])


# ---------------------------------------------------------------------------
# Cas limites et erreurs
# ---------------------------------------------------------------------------

def test_budget_nul_et_dossier_vide(monkeypatch, tmp_path):
    journal = JournalWandb()
    monkeypatch.setattr(train_self_play, "wandb", journal)
    modele, optimiseur = _model_et_optimiseur()
    scaler = ScalerFactice()
    statistiques = {}
    retour = train_on_shards(
        modele, optimiseur, scaler, torch.device("cpu"),
        tmp_path / "vide", 1e-3, batch_size=8, global_step=7,
        samples_per_epoch=19, training_stats=statistiques)
    assert retour == 7
    assert statistiques == {"samples_seen": 0}
    assert modele.appels == []
    assert scaler.etapes == 0
    assert journal.points == []

    ecrire_shard(tmp_path, "shard_0.npz", [1, 2])
    statistiques = {}
    retour = train_on_shards(
        modele, optimiseur, scaler, torch.device("cpu"), tmp_path, 1e-3,
        batch_size=8, global_step=7, samples_per_epoch=0,
        training_stats=statistiques)
    assert retour == 7
    assert statistiques == {"samples_seen": 0}
    assert modele.appels == []
    assert scaler.etapes == 0
    assert journal.points == []


def test_data_workers_non_nul_est_rejete(tmp_path):
    ecrire_shard(tmp_path, "shard_0.npz", [1])
    modele, optimiseur = _model_et_optimiseur()
    with pytest.raises(ValueError, match="data_workers"):
        train_on_shards(
            modele, optimiseur, ScalerFactice(), torch.device("cpu"),
            tmp_path, 1e-3, batch_size=4, global_step=0,
            samples_per_epoch=4, data_workers=1)
    assert modele.appels == []


# ---------------------------------------------------------------------------
# Reproductibilite
# ---------------------------------------------------------------------------

def test_manual_seed_reproduit_tirages_et_mise_a_jour(monkeypatch, tmp_path):
    for numero in range(4):
        ecrire_shard(tmp_path, f"shard_{numero}.npz",
                     [numero * 10 + 1, numero * 10 + 2, numero * 10 + 3])
    resultats = []
    for _ in range(2):
        modele, optimiseur = _model_et_optimiseur(lr=0.005)
        journal = JournalWandb()
        with monkeypatch.context() as contexte:
            contexte.setattr(train_self_play, "wandb", journal)
            torch.manual_seed(123)
            train_on_shards(
                modele, optimiseur, ScalerFactice(), torch.device("cpu"),
                tmp_path, 1e-3, batch_size=8, global_step=0,
                samples_per_epoch=19)
        etat = {cle: valeur.detach().clone()
                for cle, valeur in modele.state_dict().items()}
        resultats.append((modele.appels, etat, journal.points))
    assert resultats[0][0] == resultats[1][0]
    for cle, valeur in resultats[0][1].items():
        assert torch.equal(valeur, resultats[1][1][cle])
    assert resultats[0][2] == resultats[1][2]
