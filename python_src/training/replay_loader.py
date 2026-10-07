"""Chargeur du replay buffer : lots équilibrés et mélange borné.

Contrat complet : docs/superpowers/specs/2026-10-05-selfplay-loader-design.md.

Pour un buffer non vide et S > 0 : exactement S tirages uniformes avec remise,
regroupés par fichier, puis `ceil(S/B)` lots de tailles équilibrées émis depuis
un réservoir borné. Un seul shard est décompressé à la fois et chaque fichier
sélectionné n'est lu qu'une fois. Le module est indépendant du moteur C++, de
W&B et de `train_self_play`.
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

PLANS = 119
COTE = 8
TAILLE_POLICY = 4672
SHUFFLE_BATCHES_DEFAUT = 8

_CLES = ("states", "policies", "values")


@dataclass(frozen=True)
class ShardInfo:
    """Entrée du catalogue : chemin et effectif réel lu dans `values`."""

    path: Path
    size: int


def _ouvrir_archive(chemin):
    """Point d'ouverture d'une archive .npz, substituable en test."""
    return np.load(chemin)


def lister_shards(dossier: str | Path) -> list[ShardInfo]:
    """Catalogue les `shard_*.npz` d'un dossier par leur effectif réel.

    L'effectif est lu dans le petit tableau `values` ; `states` et `policies`
    ne sont pas décompressés pour le catalogue. Une archive illisible ou sans
    les clés attendues lève une erreur mentionnant son chemin.
    """
    dossier = Path(dossier)
    if not dossier.is_dir():
        return []
    infos: list[ShardInfo] = []
    for chemin in sorted(dossier.glob("shard_*.npz")):
        try:
            archive = _ouvrir_archive(chemin)
        except Exception as erreur:
            raise ValueError(
                f"{chemin} : archive illisible ({erreur})") from erreur
        try:
            with archive:
                manquantes = [cle for cle in _CLES if cle not in archive.files]
                if manquantes:
                    raise ValueError(
                        f"{chemin} : clés manquantes {manquantes}")
                valeurs = np.asarray(archive["values"])
        except ValueError:
            raise
        except Exception as erreur:
            raise ValueError(
                f"{chemin} : archive illisible ({erreur})") from erreur
        if valeurs.ndim != 1:
            raise ValueError(
                f"{chemin} : values doit être un vecteur, forme "
                f"{valeurs.shape}")
        infos.append(ShardInfo(chemin, int(valeurs.shape[0])))
    return infos


def tailles_batches(total: int, batch_size: int) -> list[int]:
    """Tailles des `ceil(total / batch_size)` lots, à un près.

    Les `total % K` premiers lots ont la taille supérieure, les autres un
    exemplaire de moins. Aucun tirage n'est ajouté ni retiré.
    """
    if total < 0:
        raise ValueError("total doit être positif ou nul")
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif")
    if total == 0:
        return []
    nombre_lots = -(-total // batch_size)
    base, reste = divmod(total, nombre_lots)
    return [base + 1] * reste + [base] * (nombre_lots - reste)


def tirer_indices(tailles: Sequence[int], total: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Tire `total` indices globaux avec remise et les regroupe par fichier.

    Renvoie, dans l'ordre des fichiers, les indices locaux de chaque fichier
    (tableau vide pour un fichier jamais tiré). Un buffer vide ne produit
    aucun tirage.
    """
    if total < 0:
        raise ValueError("total doit être positif ou nul")
    nombre = int(sum(int(t) for t in tailles))
    if total == 0 or nombre == 0:
        return []
    tirages = np.asarray(rng.integers(0, nombre, size=total), dtype=np.int64)
    cumul = np.cumsum(np.asarray([int(t) for t in tailles], dtype=np.int64))
    fichiers = np.searchsorted(cumul, tirages, side="right")
    debuts = np.concatenate(([0], cumul[:-1]))
    locaux = tirages - debuts[fichiers]
    return [locaux[fichiers == numero] for numero in range(len(tailles))]


def capacite_melange(total: int, batch_size: int,
                     shuffle_batches: int = SHUFFLE_BATCHES_DEFAUT) -> int:
    """Capacité du réservoir : `shuffle_batches * B`, ou S s'il est plus petit."""
    if shuffle_batches < 1:
        raise ValueError("shuffle_batches doit être au moins 1")
    return min(int(shuffle_batches) * int(batch_size), int(total))


def _charger_tableaux(archive, info: ShardInfo):
    """Valide et renvoie les tableaux d'une archive ouverte."""
    manquantes = [cle for cle in _CLES if cle not in archive.files]
    if manquantes:
        raise ValueError(f"{info.path} : clés manquantes {manquantes}")
    etats = archive["states"]
    politiques = archive["policies"]
    valeurs = archive["values"]
    if valeurs.ndim != 1:
        raise ValueError(
            f"{info.path} : values doit être un vecteur, forme {valeurs.shape}")
    n = int(valeurs.shape[0])
    if (etats.ndim != 4 or etats.shape[0] != n
            or tuple(etats.shape[1:]) != (PLANS, COTE, COTE)):
        raise ValueError(
            f"{info.path} : states de forme {etats.shape}, attendu "
            f"({n}, {PLANS}, {COTE}, {COTE})")
    if politiques.ndim != 2 or tuple(politiques.shape) != (n, TAILLE_POLICY):
        raise ValueError(
            f"{info.path} : policies de forme {politiques.shape}, attendu "
            f"({n}, {TAILLE_POLICY})")
    if n != info.size:
        raise ValueError(
            f"{info.path} : effectif {n} différent du catalogue {info.size}")
    return etats, politiques, valeurs


class _Flux:
    """Sert les exemples tirés, un seul shard décompressé à la fois.

    Chaque fichier sélectionné est ouvert et lu une seule fois ; ses tableaux
    sont libérés dès qu'on passe au fichier suivant ou que le flux est fermé.
    Les blocs servis sont des copies de `taille_morceau` lignes au plus, donc
    le réservoir ne retient jamais les tableaux complets d'un shard.
    """

    def __init__(self, shards: Sequence[ShardInfo], indices, ordre,
                 taille_morceau: int, ouvrir=None):
        self._shards = shards
        self._indices = indices
        self._ordre = [int(numero) for numero in ordre]
        self._taille_morceau = max(1, int(taille_morceau))
        self._ouvrir = ouvrir if ouvrir is not None else _ouvrir_archive
        self._rang_fichier = 0
        self._etats = None
        self._politiques = None
        self._valeurs = None
        self._locaux = None
        self._rang_local = 0
        self._bloc = None
        self._offset = 0
        self._avancer()

    def fermer(self) -> None:
        """Libère le fichier courant et le bloc en cours."""
        self._liberer_fichier()
        self._bloc = None

    def _liberer_fichier(self) -> None:
        self._etats = None
        self._politiques = None
        self._valeurs = None
        self._locaux = None
        self._rang_local = 0

    def _avancer(self) -> bool:
        """Prépare le prochain bloc, en changeant de fichier si nécessaire."""
        while True:
            if (self._locaux is not None
                    and self._rang_local < len(self._locaux)):
                bloc = self._locaux[self._rang_local:
                                    self._rang_local + self._taille_morceau]
                self._rang_local += len(bloc)
                self._bloc = (self._etats[bloc],
                              self._politiques[bloc],
                              self._valeurs[bloc])
                self._offset = 0
                return True
            self._liberer_fichier()
            if self._rang_fichier >= len(self._ordre):
                self._bloc = None
                return False
            numero = self._ordre[self._rang_fichier]
            self._rang_fichier += 1
            locaux = self._indices[numero]
            if len(locaux) == 0:
                continue
            info = self._shards[numero]
            try:
                with self._ouvrir(info.path) as archive:
                    etats, politiques, valeurs = _charger_tableaux(archive,
                                                                   info)
            except ValueError:
                raise
            except Exception as erreur:
                raise ValueError(
                    f"{info.path} : archive illisible ({erreur})") from erreur
            self._etats = etats
            self._politiques = politiques
            self._valeurs = valeurs
            self._locaux = locaux
            self._rang_local = 0

    def remplir(self, etats, politiques, valeurs, debut: int,
                maximum: int) -> int:
        """Copie au plus `maximum` exemples dans le réservoir à partir de `debut`.

        Renvoie le nombre d'exemples copiés, zéro quand le flux est épuisé.
        """
        copie = 0
        while copie < maximum and self._bloc is not None:
            bloc_etats, bloc_politiques, bloc_valeurs = self._bloc
            prendre = min(maximum - copie, len(bloc_valeurs) - self._offset)
            cible = slice(debut + copie, debut + copie + prendre)
            source = slice(self._offset, self._offset + prendre)
            etats[cible] = bloc_etats[source]
            politiques[cible] = bloc_politiques[source]
            valeurs[cible] = bloc_valeurs[source]
            copie += prendre
            self._offset += prendre
            if self._offset >= len(bloc_valeurs):
                self._avancer()
        return copie


def iter_replay_batches(
        shards: Sequence[ShardInfo], total: int, batch_size: int, *,
        rng: np.random.Generator,
        shuffle_batches: int = SHUFFLE_BATCHES_DEFAUT,
        ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Émet `ceil(S/B)` lots équilibrés, FP32, à partir de S = `total` tirages.

    Le réservoir de `shuffle_batches * B` exemples (au plus S) mélange les
    fichiers et les générations ; un seul shard est chargé à la fois. Aucun
    lot n'est émis si le budget est nul ou si le buffer ne contient aucun
    exemple.
    """
    if total < 0:
        raise ValueError("total doit être positif ou nul")
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif")
    tailles_fichiers = [int(shard.size) for shard in shards]
    if total == 0 or sum(tailles_fichiers) == 0:
        return
    capacite = capacite_melange(total, batch_size, shuffle_batches)
    indices = tirer_indices(tailles_fichiers, total, rng)
    utilises = [numero for numero in range(len(shards))
                if len(indices[numero]) > 0]
    ordre = rng.permutation(utilises)
    flux = _Flux(shards, indices, ordre, batch_size)
    etats = np.empty((capacite, PLANS, COTE, COTE), dtype=np.float16)
    politiques = np.empty((capacite, TAILLE_POLICY), dtype=np.float16)
    valeurs = np.empty((capacite,), dtype=np.float32)
    remplis = 0
    try:
        while remplis < capacite:
            ajout = flux.remplir(etats, politiques, valeurs, remplis,
                                 capacite - remplis)
            if ajout == 0:
                break
            remplis += ajout
        for taille in tailles_batches(total, batch_size):
            while remplis < taille:
                ajout = flux.remplir(etats, politiques, valeurs, remplis,
                                     capacite - remplis)
                if ajout == 0:
                    break
                remplis += ajout
            if remplis < taille:
                raise RuntimeError(
                    "replay_loader : flux épuisé avant la fin des lots")
            choix = rng.choice(remplis, size=taille, replace=False)
            yield (
                torch.from_numpy(etats[choix].astype(np.float32)),
                torch.from_numpy(politiques[choix].astype(np.float32)),
                torch.from_numpy(valeurs[choix].astype(np.float32)),
            )
            garde = np.ones(remplis, dtype=bool)
            garde[choix] = False
            survivants = np.nonzero(garde)[0]
            nb_survivants = int(survivants.shape[0])
            etats[:nb_survivants] = etats[survivants]
            politiques[:nb_survivants] = politiques[survivants]
            valeurs[:nb_survivants] = valeurs[survivants]
            remplis = nb_survivants
            while remplis < capacite:
                ajout = flux.remplir(etats, politiques, valeurs, remplis,
                                     capacite - remplis)
                if ajout == 0:
                    break
                remplis += ajout
    finally:
        flux.fermer()
