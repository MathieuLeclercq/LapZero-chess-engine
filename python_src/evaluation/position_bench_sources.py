"""Sources externes du banc de positions : archives, PGN et candidats.

Ce module lit les archives de broadcasts Lichess (CC BY-SA 4.0), filtre les
parties, extrait au plus une position par phase et par partie, puis rejoue
chaque candidat dans le moteur C++ pour construire l'entree reseau reelle.

Regles structurantes :

- provenance : seules les archives de broadcasts de la base officielle
  Lichess, jamais les corpus de preentrainement, les puzzles ou le self-play ;
- historique : chaque candidat conserve la position initiale et tous les coups
  UCI joues, si bien que le tenseur a 8 plies est celui d'une vraie partie ;
- identite : deux positions sont identiques quand le reseau et le generateur
  de coups ne peuvent pas les distinguer, c'est-a-dire quand le tenseur, la
  liste triee des index legaux, le trait, le compteur de demi-coups et le
  nombre de repetitions coincident. Le SHA-256 de cette serialisation est le
  position_id ;
- rejet explicite : toute partie refusee est comptee par cause dans un
  dictionnaire de bilan, jamais ignoree en silence.

Regles de normalisation des identites. Joueur : identifiant FIDE si present,
sinon nom replie en minuscules avec espaces reduits (la regle est celle de
`_normaliser_nom`, un espace unique entre les mots). Evenement : identifiant
explicite si present, sinon nom normalise combine au slug de diffusion extrait
de l'URL. Un joueur sans metadonnee reste de type « unknown » : il n'est
jamais presente comme humain certain.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md (sections 2
et 3) et docs/superpowers/plans/2026-09-30-position-bench.md (tache 3).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import hashlib
import io
import json
import os
import re
import struct
import sys
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast

import chess
import chess.pgn
import chess_engine
import numpy as np
import requests
import zstandard as zstd
from evaluation.bench_metrics import index_to_uci, uci_to_index
from evaluation.position_bench_metrics import Position, SourceInfo, validate_position

VARIANTE_STANDARD = "Standard"
ELO_MINIMUM = 2000
PLY_MINIMUM = 16
DATE_MINIMUM = date(2026, 3, 1)

# Selection des positions dans une partie, spec 3.2.
OUVERTURE_PLY_MIN = 8
OUVERTURE_PLY_MAX = 24
OUVERTURE_POIDS_MIN = 18
FINALE_POIDS_MAX = 8
POIDS_PHASE = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}

# Sels des hachages, figes par version du banc.
SEL_PHASE = "lapzero-position-bench-v1"
DOMAINE_IDENTITE = b"lapzero-position-bench-v1-identity"

LICENCE_BROADCASTS = "CC BY-SA 4.0"
ATTRIBUTION_BROADCASTS = "Lichess"

PHASES = ("ouverture", "milieu", "finale")
TYPES_JOUEUR = ("human", "bot", "unknown")


class DonneesInvalides(Exception):
    """La partie ou le candidat ne peut pas entrer dans le banc."""

    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


# ============================================================
#                     LECTURE DES ARCHIVES
# ============================================================

def read_games(path, source: SourceInfo):
    """Itere les parties d'un PGN ou d'un PGN.zst, en flux.

    La SourceInfo est attachee a chaque partie dans l'attribut `source_info`,
    ce qui permet un traitement par archive. Le contenu n'est jamais charge
    entierement en memoire.
    """
    chemin = Path(path)
    if chemin.suffix == ".zst":
        decompresseur = zstd.ZstdDecompressor()
        with open(chemin, "rb") as brut, io.TextIOWrapper(
                decompresseur.stream_reader(brut),
                encoding="utf-8", errors="replace") as flux:
            yield from _lire_parties(flux, source)
    else:
        with open(chemin, encoding="utf-8", errors="replace") as flux:
            yield from _lire_parties(flux, source)


def _lire_parties(flux, source: SourceInfo):
    while True:
        partie = chess.pgn.read_game(flux)
        if partie is None:
            return
        # Attribut libre : la classe Game de python-chess n'en declare aucun.
        cast(Any, partie).source_info = source
        yield partie


def download_archive(url: str, destination, expected_sha256: str | None,
                     *, fetch=None, progress=None) -> SourceInfo:
    """Telecharge une archive, verifie son hash et ne publie qu'apres controle.

    Le telechargement s'ecrit dans `<destination>.part` ; le fichier final
    n'apparait qu'apres verification du SHA-256 attendu. Toute interruption
    supprime le `.part` : un fichier partiel ne peut jamais etre pris pour une
    archive valide. `fetch` est injectable pour les tests ; `progress` recoit
    `(recus, total, octets_par_s, eta_s)` a chaque bloc recu.
    """
    chemin = Path(destination)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    partiel = chemin.with_name(chemin.name + ".part")
    if fetch is not None:
        telecharger = fetch
    else:
        telecharger = lambda url_source: _telecharger_http(url_source, progress)

    digest = hashlib.sha256()
    publie = False
    try:
        with open(partiel, "wb") as sortie:
            for bloc in telecharger(url):
                sortie.write(bloc)
                digest.update(bloc)
        somme = digest.hexdigest()
        if expected_sha256 is not None and somme != expected_sha256.lower():
            raise ValueError(
                f"hash de l'archive incorrect : {somme} au lieu de "
                f"{expected_sha256.lower()}")
        os.replace(partiel, chemin)
        publie = True
    finally:
        if not publie and partiel.exists():
            partiel.unlink()

    mois = _mois_depuis_url(url)
    return SourceInfo(
        source_id=f"lichess-broadcasts-{mois}",
        url=url,
        month=mois,
        archive_sha256=somme,
        license=LICENCE_BROADCASTS,
        attribution=ATTRIBUTION_BROADCASTS,
    )


def _telecharger_http(url: str, progress=None):
    with requests.get(url, stream=True, timeout=60) as reponse:
        reponse.raise_for_status()
        entete = reponse.headers.get("content-length")
        total = int(entete) if entete and entete.isdigit() else None
        debut = time.monotonic()
        recus = 0
        for bloc in reponse.iter_content(chunk_size=1024 * 1024):
            if not bloc:
                continue
            recus += len(bloc)
            if progress is not None:
                ecoule = time.monotonic() - debut
                debit = recus / ecoule if ecoule > 0 else 0.0
                eta = ((total - recus) / debit
                       if total is not None and debit > 0 else None)
                progress(recus, total, debit, eta)
            yield bloc


def formater_progression(nom: str, recus: int, total: int | None,
                         debit: float, eta: float | None) -> str:
    """Ligne de progression : pourcentage, volume, debit et ETA."""
    if total:
        avancement = (f"{100.0 * recus / total:5.1f} % "
                      f"({recus / 1e6:7.1f}/{total / 1e6:.1f} Mo)")
    else:
        avancement = f"{recus / 1e6:7.1f} Mo"
    reste = (f" ETA {int(eta) // 60:02d}:{int(eta) % 60:02d}"
             if eta is not None else "")
    return f"{nom} : {avancement} {debit / 1e6:6.2f} Mo/s{reste}"


def barre_progression(nom: str, *, intervalle_s: float = 0.5, flot=None,
                      fichier=None) -> Callable:
    """Fabrique un callback de progression, limite en frequence.

    Ecrit la ligne sur `flot` (stderr par defaut) et, si `fichier` est fourni,
    ecrase son contenu a chaque mise a jour : un suivi depuis un autre terminal
    lit ainsi l'etat courant du telechargement.
    """
    dernier = [0.0]

    def suivi(recus: int, total: int | None, debit: float,
              eta: float | None) -> None:
        maintenant = time.monotonic()
        if maintenant - dernier[0] < intervalle_s:
            return
        dernier[0] = maintenant
        ligne = formater_progression(nom, recus, total, debit, eta)
        sortie = sys.stderr if flot is None else flot
        print(f"\r{ligne}", end="", file=sortie, flush=True)
        if fichier is not None:
            Path(fichier).write_text(ligne + "\n", encoding="utf-8")

    return suivi


def _mois_depuis_url(url: str) -> str:
    correspondance = re.search(r"(\d{4})-(\d{2})", url)
    if correspondance is None:
        raise ValueError(f"mois illisible dans l'URL : {url}")
    return f"{correspondance[1]}-{correspondance[2]}"


# ============================================================
#                  EXTRACTION DES CANDIDATS
# ============================================================

def extract_candidates(games, source: SourceInfo, *, compteurs=None,
                       date_min: date = DATE_MINIMUM, progress=None,
                       progress_every: int = 2500) -> list[Position]:
    """Extrait au plus une position par phase et par partie valide.

    `compteurs` recoit par cause le nombre de parties ou de candidats rejetes.
    `progress` est appele tous les `progress_every` parties avec l'etat courant
    (`parties_lues`, `candidats`, `rejets`), pour suivre une extraction longue.
    Le resultat est trie par ordre canonique `(game_id, ply, source_id)` et
    dedoublonne par identite reseau : deux candidats indiscernables pour le
    reseau ne sont presentes qu'une fois, quelle que soit l'ordre de lecture.
    """
    compteur = {} if compteurs is None else compteurs
    candidats: list[Position] = []
    empreintes_vues = set()
    for lues, partie in enumerate(games, start=1):
        try:
            candidats_partie = _extraire_partie(
                partie, source, compteur, date_min, empreintes_vues)
        except DonneesInvalides as erreur:
            _compter(compteur, erreur.cause)
        else:
            candidats.extend(candidats_partie)
        if progress is not None and lues % progress_every == 0:
            rejets = sum(valeur for cause, valeur in compteur.items()
                         if cause != "date_hors_mois")
            progress({"parties_lues": lues, "candidats": len(candidats),
                      "rejets": rejets})

    return dedupliquer_candidats(candidats)


def dedupliquer_candidats(candidats: list[Position]) -> list[Position]:
    """Ne garde qu'une occurrence par identite reseau, en ordre canonique.

    L'occurrence retenue est celle de plus petite cle `(game_id, ply,
    source_id)`, ce qui rend le resultat independant de l'ordre de lecture et
    du mois d'archive. La sortie est triee par cette meme cle.
    """
    uniques: dict[str, tuple[tuple[str, int, str], Position]] = {}
    for candidat in candidats:
        cle = (candidat["game_id"], candidat["ply"], candidat["source_id"])
        connu = uniques.get(candidat["position_id"])
        if connu is None or cle < connu[0]:
            uniques[candidat["position_id"]] = (cle, candidat)
    # Le position_id departage les cles canoniques egales, cas reel des
    # parties qui partagent l'identifiant de leur diffusion.
    return [candidat for _, candidat in sorted(
        uniques.values(),
        key=lambda item: (*item[0], item[1]["position_id"]))]


def filtrer_candidats_non_terminaux(candidats: list[Position]) -> list[Position]:
    """Retire les nulles de regle des anciens caches d'extraction.

    Ces compteurs proviennent du rejeu C++ integral. Python-chess ne considere
    pas la troisieme repetition ni les 50 coups comme automatiquement terminaux,
    alors que le MCTS le fait. Les autres fins de partie etaient deja exclues.
    """
    return [c for c in candidats
            if c["halfmove_clock"] < 100 and c["repetition_count"] < 3]


def _extraire_partie(partie, source: SourceInfo, compteur: dict,
                     date_min: date, empreintes_vues: set) -> list[Position]:
    en_tete = partie.headers

    if en_tete.get("Variant", VARIANTE_STANDARD) != VARIANTE_STANDARD:
        raise DonneesInvalides("variante_non_standard")
    if "SetUp" in en_tete or "FEN" in en_tete:
        raise DonneesInvalides("depart_non_standard")
    if en_tete.get("Result", "") not in ("1-0", "0-1", "1/2-1/2"):
        raise DonneesInvalides("resultat_incomplet")
    if partie.errors:
        raise DonneesInvalides("partie_illegale")

    elo_blanc = _elo(en_tete.get("WhiteElo"))
    elo_noir = _elo(en_tete.get("BlackElo"))
    if elo_blanc is None or elo_noir is None:
        raise DonneesInvalides("elo_absent")
    if elo_blanc < ELO_MINIMUM or elo_noir < ELO_MINIMUM:
        raise DonneesInvalides("elo_insuffisant")

    jour = _date_partie(en_tete.get("Date"))
    if jour is None:
        raise DonneesInvalides("date_absente")
    if jour < date_min:
        raise DonneesInvalides("date_ancienne")

    identite_blanc, type_blanc = _identite_joueur(en_tete, "White")
    identite_noir, type_noir = _identite_joueur(en_tete, "Black")
    evenement = _identite_evenement(en_tete)

    # Premiere passe : la ligne principale, legalite comprise. L'identifiant
    # de partie peut se replier sur l'empreinte, donc deux passes.
    coups = []
    plateau = chess.Board()
    for coup in partie.mainline_moves():
        if not plateau.is_legal(coup):
            raise DonneesInvalides("partie_illegale")
        plateau.push(coup)
        coups.append(coup.uci())
    if len(coups) < PLY_MINIMUM:
        raise DonneesInvalides("partie_trop_courte")

    empreinte = game_fingerprint(coups)
    if empreinte in empreintes_vues:
        raise DonneesInvalides("partie_dupliquee")
    empreintes_vues.add(empreinte)
    game_id = _game_id(en_tete, empreinte)

    # Seconde passe : au plus un candidat par phase, choisi par hash stable.
    meilleurs: dict[str, tuple[int, chess.Board] | None] = {
        phase: None for phase in PHASES}
    plateau = chess.Board()
    for ply, coup in enumerate(coups, start=1):
        plateau.push_uci(coup)
        if plateau.is_game_over():
            break
        if plateau.halfmove_clock >= 100 or plateau.is_repetition(3):
            # Ignorer cette racine, mais une vraie partie peut continuer apres
            # une nulle reclamable : les positions suivantes restent candidates.
            _compter(compteur, "position_terminale")
            continue
        phase = _phase(ply, plateau)
        courant = meilleurs[phase]
        if (courant is None
                or _hash_phase(game_id, ply) < _hash_phase(game_id, courant[0])):
            meilleurs[phase] = (ply, plateau.copy())

    if jour.strftime("%Y-%m") != source["month"]:
        _compter(compteur, "date_hors_mois")

    infos = {
        "game_id": game_id,
        "game_fingerprint": empreinte,
        "source_id": source["source_id"],
        "event_id": evenement,
        "white_id": identite_blanc,
        "black_id": identite_noir,
        "white_elo": elo_blanc,
        "black_elo": elo_noir,
        "white_type": type_blanc,
        "black_type": type_noir,
        "game_date": jour.isoformat(),
    }

    candidats = []
    for phase in PHASES:
        choix = meilleurs[phase]
        if choix is None:
            continue
        ply, plateau_candidat = choix
        enregistrement = _construire_candidat(
            infos, phase, ply, plateau_candidat, coups, compteur)
        if enregistrement is not None:
            candidats.append(enregistrement)
    return candidats


def _construire_candidat(infos: dict, phase: str, ply: int,
                         plateau_python: chess.Board, coups: list[str],
                         compteur: dict) -> Position | None:
    moves_uci = list(coups[:ply])
    fen_attendu = plateau_python.fen(en_passant="fen")
    try:
        board = replay_position(
            {"start_fen": chess.STARTING_FEN, "moves_uci": moves_uci})
    except DonneesInvalides:
        _compter(compteur, "historique_illegal")
        return None
    if board.to_fen() != fen_attendu:
        _compter(compteur, "fen_incoherente")
        return None
    try:
        indices = _valider_coups(board, plateau_python)
    except DonneesInvalides as erreur:
        _compter(compteur, erreur.cause)
        return None

    cle = board.get_evaluation_cache_key(0)
    if board.half_move_clock >= 100 or int(cle.repetition_category) >= 2:
        _compter(compteur, "position_terminale")
        return None
    enregistrement: Position = {
        "position_id": position_identity(board),
        "game_id": infos["game_id"],
        "game_fingerprint": infos["game_fingerprint"],
        "source_id": infos["source_id"],
        "event_id": infos["event_id"],
        "white_id": infos["white_id"],
        "black_id": infos["black_id"],
        "white_elo": infos["white_elo"],
        "black_elo": infos["black_elo"],
        "white_type": infos["white_type"],
        "black_type": infos["black_type"],
        "game_date": infos["game_date"],
        "start_fen": chess.STARTING_FEN,
        "moves_uci": moves_uci,
        "fen": fen_attendu,
        "ply": ply,
        "turn": 1 if board.turn == chess_engine.Color.BLACK else 0,
        "halfmove_clock": plateau_python.halfmove_clock,
        "repetition_count": min(3, int(cle.repetition_category) + 1),
        "phase": phase,
        "legal_indices": indices,
    }
    validate_position(enregistrement)
    return enregistrement


def _valider_coups(board: chess_engine.Chessboard,
                   plateau_python: chess.Board) -> list[int]:
    """Controle la bijection entre index C++ et coups UCI de python-chess.

    Le controle porte sur la position apres rejeu, pas sur la position de la
    partie python-chess : les deux doivent decrire exactement les memes coups
    legaux, sous-promotions et prise en passant comprises.
    """
    # Le C++ rend les coups dans l'ordre de generation : le contrat du banc
    # impose la liste triee par index, c'est donc au stockage de trier.
    indices = sorted(board.get_legal_move_indices())
    if len(set(indices)) != len(indices):
        raise DonneesInvalides("coups_dupliques")
    attendus = sorted(move.uci() for move in plateau_python.legal_moves)
    obtenus = sorted(index_to_uci(board, index) for index in indices)
    if attendus != obtenus:
        raise DonneesInvalides("coups_incoherents")
    for index in indices:
        if uci_to_index(board, index_to_uci(board, index)) != index:
            raise DonneesInvalides("bijection_invalide")
    return indices


def replay_position(record: Mapping[str, object]) -> chess_engine.Chessboard:
    """Rejoue l'historique d'un enregistrement, FEN de depart comprise."""
    board = chess_engine.Chessboard()
    board.load_fen(str(record["start_fen"]))
    for coup in cast(list[str], record["moves_uci"]):
        if not board.move_piece_uci(coup):
            raise DonneesInvalides("historique_illegal")
    return board


def position_identity(board: chess_engine.Chessboard) -> str:
    """Cle d'identite reseau d'une position : SHA-256 du tenseur, des coups
    legaux, du trait, du compteur de demi-coups et des repetitions.

    Les frontieres de la serialisation sont explicites : un domaine et les
    longueurs des trois blocs sont prepends, si bien que deux concatenations
    differentes ne peuvent pas produire la meme charge. La cle h0 seule ne
    remplace pas le tenseur historique : elle ignore l'historique, que le
    tenseur encode.
    """
    tenseur = np.asarray(board.get_alphazero_tensor(), dtype="<f4", order="C")
    indices = np.asarray(sorted(board.get_legal_move_indices()), dtype="<u4")
    repetitions = int(board.get_evaluation_cache_key(0).repetition_category) + 1
    suffixe = json.dumps(
        [int(board.turn), board.half_move_clock, repetitions],
        separators=(",", ":")).encode("ascii")

    tenseur_octets = tenseur.tobytes()
    indices_octets = indices.tobytes()
    entete = struct.pack("<3I", len(tenseur_octets), len(indices_octets),
                         len(suffixe))
    charge = (DOMAINE_IDENTITE + entete + tenseur_octets + indices_octets
              + suffixe)
    return hashlib.sha256(charge).hexdigest()


def game_fingerprint(moves_uci, depart: str = chess.STARTING_FEN) -> str:
    """Empreinte d'une partie : SHA-256 de la position de depart et des coups.

    Elle ne depend ni des commentaires, ni des variantes, ni des en-tetes, ni
    de l'ordre de lecture des archives.
    """
    charge = json.dumps([depart, list(moves_uci)],
                        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(charge).hexdigest()


# ============================================================
#                     FILTRES ET IDENTITES
# ============================================================

def _elo(valeur) -> int | None:
    if valeur is None:
        return None
    try:
        return int(str(valeur).strip())
    except ValueError:
        return None


def _date_partie(valeur) -> date | None:
    if not valeur:
        return None
    morceaux = str(valeur).split(".")
    if len(morceaux) != 3:
        return None
    try:
        return date(int(morceaux[0]), int(morceaux[1]), int(morceaux[2]))
    except ValueError:
        return None


def _normaliser_nom(nom: str) -> str:
    """Regle de normalisation des noms : minuscules, espaces reduits."""
    return " ".join(str(nom).strip().casefold().split())


def _valeur_en_tete(en_tete, cle: str) -> str:
    """Valeur d'un en-tete, vide si absente.

    python-chess pre-remplit ses en-tetes avec « ? » (et la date avec
    « ????.??.?? ») quand le PGN ne les fournit pas : ces valeurs sentinelles
    comptent comme absentes.
    """
    valeur = str(en_tete.get(cle, "")).strip()
    return "" if valeur == "?" else valeur


def _type_joueur(en_tete: dict, couleur: str) -> str:
    for cle in (f"{couleur}Type", f"{couleur}Title"):
        if _valeur_en_tete(en_tete, cle).upper() == "BOT":
            return "bot"
    return "unknown"


def _identite_joueur(en_tete: dict, couleur: str) -> tuple[str, str]:
    fide = _valeur_en_tete(en_tete, f"{couleur}FideId")
    if fide and fide != "0":
        if not fide.isdigit():
            raise DonneesInvalides("identite_joueur")
        return f"fide:{fide}", _type_joueur(en_tete, couleur)
    nom = _normaliser_nom(_valeur_en_tete(en_tete, couleur))
    if not nom:
        raise DonneesInvalides("identite_joueur")
    return f"nom:{nom}", _type_joueur(en_tete, couleur)


def _identite_evenement(en_tete: dict) -> str:
    identifiant = _valeur_en_tete(en_tete, "EventId")
    if identifiant:
        return identifiant
    evenement = _normaliser_nom(_valeur_en_tete(en_tete, "Event"))
    contexte = _contexte_diffusion(_valeur_en_tete(en_tete, "Site"))
    if not evenement and not contexte:
        raise DonneesInvalides("identite_evenement")
    charge = f"{evenement}|{contexte}".encode()
    return "evt:" + hashlib.sha256(charge).hexdigest()[:32]


def _contexte_diffusion(site: str) -> str:
    correspondance = re.search(r"/broadcast/([^/#?]+)", site)
    if correspondance is not None:
        return correspondance.group(1).casefold()
    return _normaliser_nom(site)


def _game_id(en_tete: dict, empreinte: str) -> str:
    for cle in ("GameId", "LichessGameId", "BroadcastGameId"):
        valeur = _valeur_en_tete(en_tete, cle)
        if valeur:
            return valeur
    segments = [s for s in _valeur_en_tete(en_tete, "Site").rstrip("/").split("/")
                if s]
    if segments:
        dernier = segments[-1]
        if 8 <= len(dernier) <= 12 and dernier.isalnum():
            return dernier
    return empreinte


def _phase(ply: int, plateau: chess.Board) -> str:
    poids = 0
    for type_piece, valeur in POIDS_PHASE.items():
        poids += valeur * len(plateau.pieces(type_piece, chess.WHITE))
        poids += valeur * len(plateau.pieces(type_piece, chess.BLACK))
    if (OUVERTURE_PLY_MIN <= ply <= OUVERTURE_PLY_MAX
            and poids >= OUVERTURE_POIDS_MIN):
        return "ouverture"
    if poids <= FINALE_POIDS_MAX:
        return "finale"
    return "milieu"


def _hash_phase(game_id: str, ply: int) -> str:
    charge = f"{game_id}|{ply}|{SEL_PHASE}".encode()
    return hashlib.sha256(charge).hexdigest()


def _compter(compteur: dict, cause: str) -> None:
    compteur[cause] = compteur.get(cause, 0) + 1
