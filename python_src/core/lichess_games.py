"""
Couche de recuperation des parties Lichess par lots, avec cache disque.

Isolee du reste du pipeline pour deux raisons : les preoccupations reseau
(limites de debit, reprise, jeton) ne concernent personne d'autre, et le
recuperateur est injectable, ce qui rend le module testable sans reseau.
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import io
import re
import time
from pathlib import Path
from typing import Callable, Iterator

import chess.pgn
import requests

EXPORT_URL = "https://lichess.org/api/games/export/_ids"

# Verifie le 2026-08-07 contre la specification OpenAPI de Lichess :
# "300 IDs can be submitted."
MAX_IDS_PER_REQUEST = 300

# Lichess demande de ne faire qu'une requete a la fois.
PAUSE_BETWEEN_REQUESTS_S = 1.0
MAX_RETRIES = 5

# Les identifiants de partie Lichess font exactement 8 caracteres.
_ID_RE = re.compile(r"lichess\.org/([A-Za-z0-9]{8})")
_PLY_RE = re.compile(r"#(\d+)\s*$")


def game_id_from_url(url: str) -> str | None:
    """Extrait l'identifiant de partie a 8 caracteres d'une GameUrl."""
    match = _ID_RE.search(url or "")
    return match.group(1) if match else None


def ply_hint_from_url(url: str) -> int | None:
    """Numero de ply de l'ancre, s'il y en a une. Sert uniquement a lever une
    ambiguite d'appariement, jamais a determiner la position directement."""
    match = _PLY_RE.search(url or "")
    return int(match.group(1)) if match else None


def batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _cache_path(game_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{game_id}.pgn"


def cached_pgn(game_id: str, cache_dir: Path) -> str | None:
    path = _cache_path(game_id, cache_dir)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _http_fetcher(ids: list[str], token: str | None) -> str:
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(MAX_RETRIES):
        response = requests.post(
            EXPORT_URL, data=",".join(ids), headers=headers, timeout=120)
        if response.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  429 recu, attente de {wait} s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.text
    raise RuntimeError("limite de debit non levee apres plusieurs tentatives")


def _split_pgn_by_game_id(all_pgn: str) -> dict[str, str]:
    """Decoupe une reponse multi-parties et indexe par identifiant.

    L'en-tete Site porte l'URL de la partie, donc son identifiant. On ne se fie
    pas a l'ordre de la reponse.
    """
    result: dict[str, str] = {}
    stream = io.StringIO(all_pgn)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        game_id = game_id_from_url(game.headers.get("Site", ""))
        if game_id:
            result[game_id] = str(game)
    return result


def fetch_games(game_ids: list[str],
                cache_dir: Path,
                token: str | None = None,
                fetcher: Callable[[list[str], str | None], str] | None = None) -> None:
    """Remplit le cache disque pour les identifiants demandes.

    Ne retelecharge jamais ce qui est deja en cache, donc une relance apres
    interruption ne coute que ce qui manque. Les parties absentes de la reponse
    (supprimees, privees) ne sont pas mises en cache et seront simplement
    manquantes a l'etape d'appariement.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or _http_fetcher

    missing = [gid for gid in game_ids if not _cache_path(gid, cache_dir).exists()]
    if not missing:
        return

    total_batches = (len(missing) + MAX_IDS_PER_REQUEST - 1) // MAX_IDS_PER_REQUEST
    for index, batch in enumerate(batched(missing, MAX_IDS_PER_REQUEST), start=1):
        print(f"  lot {index}/{total_batches} ({len(batch)} parties)", flush=True)
        for game_id, pgn in _split_pgn_by_game_id(fetcher(batch, token)).items():
            _cache_path(game_id, cache_dir).write_text(pgn, encoding="utf-8")
        if index < total_batches:
            time.sleep(PAUSE_BETWEEN_REQUESTS_S)
