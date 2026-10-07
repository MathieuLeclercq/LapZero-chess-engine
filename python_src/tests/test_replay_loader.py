"""Tests du chargeur du replay buffer (`replay_loader`).

Contrat : docs/superpowers/specs/2026-10-05-selfplay-loader-design.md.
Aucun GPU, aucun modele, aucun acces reseau.
"""
import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import training.replay_loader as replay_loader
from training.replay_loader import (
    ShardInfo,
    capacite_melange,
    iter_replay_batches,
    lister_shards,
    tailles_batches,
    tirer_indices,
)

PLANS = 119
TAILLE_POLICY = 4672
COTE = 8


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class RngFactice:
    """RNG minimal qui rend une suite d'indices imposee."""

    def __init__(self, valeurs):
        self.valeurs = np.asarray(valeurs, dtype=np.int64)
        self.appels = []

    def integers(self, low, high, size):
        self.appels.append((low, high, size))
        return self.valeurs


def _fabriquer_contenu(identifiants):
    """Mini-archive en memoire : le meme identifiant dans les trois tableaux."""
    n = len(identifiants)
    states = np.zeros((n, PLANS, COTE, COTE), dtype=np.float16)
    policies = np.zeros((n, TAILLE_POLICY), dtype=np.float16)
    values = np.zeros((n,), dtype=np.float32)
    for i, ident in enumerate(identifiants):
        states[i].fill(np.float16(ident))
        policies[i, 0] = ident
        values[i] = float(ident)
    return {"states": states, "policies": policies, "values": values}


def ecrire_shard(dossier, nom, identifiants):
    """Ecrit une vraie mini-archive .npz et renvoie son chemin."""
    chemin = Path(dossier) / nom
    contenu = _fabriquer_contenu(identifiants)
    np.savez_compressed(chemin, **contenu)
    return chemin


class JournalChargement:
    """Compte ouvertures, fermetures et chargements lourds par fichier."""

    def __init__(self, identifiants):
        self.identifiants = {Path(c): list(v) for c, v in identifiants.items()}
        self.ouverts = {}
        self.fermes = {}
        self.chargements_lourds = {}
        self.references = {}
        self.actifs = 0
        self.max_actifs = 0

    def ouvrir(self, chemin):
        return ArchiveFactice(Path(chemin), self)


class ArchiveFactice:
    def __init__(self, chemin, journal):
        self._chemin = chemin
        self._journal = journal
        self._contenu = _fabriquer_contenu(journal.identifiants[chemin])
        journal.references.setdefault(chemin, []).append(
            weakref.ref(self._contenu["states"]))

    @property
    def files(self):
        return list(self._contenu)

    def __getitem__(self, cle):
        journal = self._journal
        journal.chargements_lourds.setdefault(self._chemin, {})
        journal.chargements_lourds[self._chemin][cle] = (
            journal.chargements_lourds[self._chemin].get(cle, 0) + 1)
        return self._contenu[cle]

    def __enter__(self):
        self._journal.ouverts[self._chemin] = (
            self._journal.ouverts.get(self._chemin, 0) + 1)
        self._journal.actifs += 1
        self._journal.max_actifs = max(self._journal.max_actifs,
                                       self._journal.actifs)
        return self

    def __exit__(self, *args):
        self._journal.actifs -= 1
        self._journal.fermes[self._chemin] = (
            self._journal.fermes.get(self._chemin, 0) + 1)
        return False


# ---------------------------------------------------------------------------
# Tailles des batches
# ---------------------------------------------------------------------------

def test_tailles_batches_cas_du_contrat():
    assert tailles_batches(0, 4096) == []
    assert tailles_batches(5, 4096) == [5]
    assert tailles_batches(4096, 4096) == [4096]
    assert tailles_batches(4097, 4096) == [2049, 2048]
    assert tailles_batches(8197, 4096) == [2733, 2732, 2732]


def test_tailles_batches_rejette_les_entrees_invalides():
    with pytest.raises(ValueError):
        tailles_batches(-1, 4096)
    with pytest.raises(ValueError):
        tailles_batches(10, 0)
    with pytest.raises(ValueError):
        tailles_batches(10, -2)


# ---------------------------------------------------------------------------
# Tirages
# ---------------------------------------------------------------------------

def test_tirer_indices_total_bornes_et_fichier_vide():
    rng = np.random.default_rng(7)
    indices = tirer_indices([4, 0, 6], 25, rng)
    assert len(indices) == 3
    assert sum(len(a) for a in indices) == 25
    assert all(0 <= int(i) < 4 for i in indices[0])
    assert len(indices[1]) == 0
    assert all(0 <= int(i) < 6 for i in indices[2])


def test_tirer_indices_graine_reproductible():
    premiere = tirer_indices([10, 10], 50, np.random.default_rng(123))
    seconde = tirer_indices([10, 10], 50, np.random.default_rng(123))
    assert len(premiere) == len(seconde)
    for a, b in zip(premiere, seconde):
        assert np.array_equal(a, b)


def test_tirer_indices_fichier_unique():
    indices = tirer_indices([1], 9, np.random.default_rng(1))
    assert len(indices) == 1
    assert np.array_equal(indices[0], np.zeros(9, dtype=np.int64))


def test_tirer_indices_regroupement_par_fichier():
    rng = RngFactice([0, 2, 2, 4, 1, 3])
    indices = tirer_indices([2, 3], 6, rng)
    assert rng.appels == [(0, 5, 6)]
    assert np.array_equal(indices[0], [0, 1])
    assert np.array_equal(indices[1], [0, 0, 2, 1])


def test_tirer_indices_sans_fichier_non_vide():
    assert tirer_indices([0, 0], 5, np.random.default_rng(1)) == []
    assert tirer_indices([1], 0, np.random.default_rng(1)) == []


def test_tirer_indices_rejette_un_total_negatif():
    with pytest.raises(ValueError):
        tirer_indices([1], -1, np.random.default_rng(1))


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def test_lister_shards_taille_reelle_pas_le_nom(tmp_path):
    ecrire_shard(tmp_path, "shard_20260101_000000_999.npz", [1, 2, 3])
    infos = lister_shards(tmp_path)
    assert len(infos) == 1
    assert infos[0].size == 3
    assert infos[0].path.name.startswith("shard_")


def test_lister_shards_ignore_les_autres_fichiers(tmp_path):
    ecrire_shard(tmp_path, "shard_a.npz", [0])
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "autre.npz").write_bytes(b"")
    assert [i.path.name for i in lister_shards(tmp_path)] == ["shard_a.npz"]


def test_lister_shards_dossier_absent(tmp_path):
    assert lister_shards(tmp_path / "absent") == []


def test_lister_shards_archive_illisible(tmp_path):
    (tmp_path / "shard_casse.npz").write_text("pas une archive")
    with pytest.raises(ValueError, match="shard_casse"):
        lister_shards(tmp_path)


def test_lister_shards_cle_manquante(tmp_path):
    np.savez_compressed(
        tmp_path / "shard_sans_policies.npz",
        states=np.zeros((1, PLANS, COTE, COTE), np.float16),
        values=np.zeros(1, np.float32))
    with pytest.raises(ValueError, match="policies"):
        lister_shards(tmp_path)


def test_lister_shards_values_mal_forme(tmp_path):
    np.savez_compressed(
        tmp_path / "shard_values_2d.npz",
        states=np.zeros((1, PLANS, COTE, COTE), np.float16),
        policies=np.zeros((1, TAILLE_POLICY), np.float16),
        values=np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError, match="values"):
        lister_shards(tmp_path)


def test_chargement_rejette_dimensions_incoherentes(tmp_path):
    np.savez_compressed(
        tmp_path / "shard_dims.npz",
        states=np.zeros((2, PLANS, COTE, COTE), np.float16),
        policies=np.zeros((2, TAILLE_POLICY), np.float16),
        values=np.zeros(3, np.float32))
    infos = lister_shards(tmp_path)
    assert infos[0].size == 3
    with pytest.raises(ValueError, match="shard_dims"):
        list(iter_replay_batches(infos, 5, 4, rng=np.random.default_rng(0)))


# ---------------------------------------------------------------------------
# Flux de batches
# ---------------------------------------------------------------------------

def test_flux_lots_equilibres_total_et_alignement(tmp_path):
    for numero in range(4):
        ecrire_shard(tmp_path, f"shard_{numero}.npz",
                     list(range(numero * 10 + 1, numero * 10 + 4)))
    infos = lister_shards(tmp_path)
    lots = list(iter_replay_batches(infos, 19, 8, rng=np.random.default_rng(123)))
    assert [x.shape[0] for x, _, _ in lots] == [7, 6, 6]
    total = 0
    for x, pi, v in lots:
        assert x.dtype == torch.float32
        assert pi.dtype == torch.float32
        assert v.dtype == torch.float32
        assert x.shape[1:] == (PLANS, COTE, COTE)
        assert pi.shape[1] == TAILLE_POLICY
        marqueurs = x[:, 0, 0, 0]
        assert np.array_equal(marqueurs, pi[:, 0])
        assert np.array_equal(marqueurs, v)
        attendus = {1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}
        assert set(marqueurs.tolist()) <= attendus
        total += x.shape[0]
    assert total == 19


def test_flux_melange_plusieurs_origines(tmp_path):
    for numero in range(4):
        ecrire_shard(tmp_path, f"shard_{numero}.npz",
                     list(range(numero * 100 + 1, numero * 100 + 6)))
    infos = lister_shards(tmp_path)
    lots = list(iter_replay_batches(infos, 19, 8, rng=np.random.default_rng(123)))
    origines = []
    for x, _, _ in lots:
        origines.append({int(m) // 100 for m in x[:, 0, 0, 0].tolist()})
    assert any(len(o) >= 2 for o in origines)


def test_flux_ne_tire_jamais_dans_un_fichier_vide(tmp_path):
    ecrire_shard(tmp_path, "shard_vide.npz", [])
    ecrire_shard(tmp_path, "shard_plein.npz", [50, 51, 52])
    infos = lister_shards(tmp_path)
    vide = [i for i in infos if i.path.name == "shard_vide.npz"]
    assert len(vide) == 1 and vide[0].size == 0
    marqueurs = []
    for x, _, _ in iter_replay_batches(infos, 12, 5, rng=np.random.default_rng(5)):
        marqueurs.extend(x[:, 0, 0, 0].tolist())
    assert set(marqueurs) == {50.0, 51.0, 52.0}


def test_flux_budget_nul_ou_buffer_vide(tmp_path):
    ecrire_shard(tmp_path, "shard_1.npz", [1, 2])
    infos = lister_shards(tmp_path)
    vide = ShardInfo(tmp_path / "absent.npz", 0)
    assert list(iter_replay_batches(infos, 0, 4,
                                    rng=np.random.default_rng(0))) == []
    assert list(iter_replay_batches([], 10, 4,
                                    rng=np.random.default_rng(0))) == []
    assert list(iter_replay_batches([vide], 10, 4,
                                    rng=np.random.default_rng(0))) == []


def test_flux_rejette_les_entrees_invalides(tmp_path):
    ecrire_shard(tmp_path, "shard_1.npz", [1, 2])
    infos = lister_shards(tmp_path)
    with pytest.raises(ValueError):
        list(iter_replay_batches(infos, -1, 4, rng=np.random.default_rng(0)))
    with pytest.raises(ValueError):
        list(iter_replay_batches(infos, 4, 0, rng=np.random.default_rng(0)))
    with pytest.raises(ValueError):
        list(iter_replay_batches(infos, 4, 4, rng=np.random.default_rng(0),
                                 shuffle_batches=0))


def test_capacite_melange_bornee():
    assert capacite_melange(19, 8) == 19
    assert capacite_melange(100_000, 4096) == 4096 * 8
    assert capacite_melange(100_000, 4096, shuffle_batches=2) == 8192
    with pytest.raises(ValueError):
        capacite_melange(10, 4, shuffle_batches=0)


# ---------------------------------------------------------------------------
# Memoire structurelle
# ---------------------------------------------------------------------------

def test_flux_un_seul_shard_actif_et_un_chargement_lourd(monkeypatch, tmp_path):
    identifiants = {}
    for numero in range(4):
        ids = list(range(numero * 10 + 1, numero * 10 + 4))
        chemin = ecrire_shard(tmp_path, f"shard_{numero}.npz", ids)
        identifiants[chemin] = ids
    infos = lister_shards(tmp_path)
    journal = JournalChargement(identifiants)
    monkeypatch.setattr(replay_loader, "_ouvrir_archive", journal.ouvrir)
    lots = list(iter_replay_batches(infos, 9, 4, rng=np.random.default_rng(3)))
    assert sum(x.shape[0] for x, _, _ in lots) == 9
    assert journal.max_actifs == 1
    assert journal.ouverts
    for comptes in journal.chargements_lourds.values():
        assert comptes.get("states") == 1
        assert comptes.get("policies") == 1
    assert journal.actifs == 0


def test_flux_fermeture_anticipee_libere_les_archives(monkeypatch, tmp_path):
    identifiants = {}
    for numero in range(20):
        ids = list(range(numero * 10 + 1, numero * 10 + 6))
        chemin = ecrire_shard(tmp_path, f"shard_{numero:02d}.npz", ids)
        identifiants[chemin] = ids
    infos = lister_shards(tmp_path)
    journal = JournalChargement(identifiants)
    monkeypatch.setattr(replay_loader, "_ouvrir_archive", journal.ouvrir)
    flux = iter_replay_batches(infos, 7, 4, rng=np.random.default_rng(11))
    premier = next(flux)
    assert premier[0].shape[0] == 4
    flux.close()
    assert sum(journal.ouverts.values()) == sum(journal.fermes.values())
    assert journal.actifs == 0
    assert sum(journal.ouverts.values()) < 20


def test_flux_ne_retient_pas_les_tableaux_des_shards_quittes(monkeypatch, tmp_path):
    identifiants = {}
    infos = []
    for numero in range(2):
        ids = list(range(numero * 10 + 1, numero * 10 + 4))
        chemin = tmp_path / f"factice_{numero}.npz"
        identifiants[chemin] = ids
        infos.append(ShardInfo(chemin, len(ids)))
    journal = JournalChargement(identifiants)
    monkeypatch.setattr(replay_loader, "_ouvrir_archive", journal.ouvrir)
    flux = iter_replay_batches(infos, 6, 3, rng=np.random.default_rng(2))
    premier = next(flux)
    assert premier[0].shape[0] == 3
    gc.collect()
    assert journal.references[infos[0].path][0]() is None
    flux.close()
    gc.collect()
    assert journal.references[infos[1].path][0]() is None
