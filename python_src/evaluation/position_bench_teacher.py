"""Annotation Stockfish du banc de positions, cachee et reprenable.

L'annotation se fait hors ligne, en deux passes (conception, section 4) : un
criblage a 50 000 noeuds donne une esperance de resultat par position, puis
chaque coup legal de la reserve est analyse a 200 000 noeuds avec root_moves.
Le score est toujours ramene au point de vue du joueur au trait.

Le cache de travail est un fichier SQLite : une seule connexion en ecriture,
tenue par le coordinateur, et une transaction par resultat complet. A la
reouverture, la configuration (moteur, reseaux, budgets, schema) est comparee
a celle du cache ; une difference est refusee, jamais reinterprettee.

Un reseau NNUE embarque n'est hache qu'apres une extraction reelle par la
commande d'export du binaire ; si l'identification est impossible, la fonction
leve une erreur precise plutot que d'inventer un hash.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md (section 4) et
docs/superpowers/plans/2026-09-30-position-bench.md (tache 4).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chess
import chess.engine
from evaluation.bench_metrics import index_to_uci, uci_to_index
from evaluation.position_bench_metrics import (
    TOLERANCE,
    AnnotatedPosition,
    MoveLabel,
    validate_move_label,
    validate_position,
    wdl_category,
    wdl_score,
)
from evaluation.position_bench_sources import replay_position

NODES_CRIBLAGE = 50_000
NODES_ANNOTATION = 200_000
NODES_AUDIT = 400_000
SCHEMA_STORE = 1

SEUIL_REGRET_AUDIT = 0.02
SEUIL_VARIATION_AUDIT = 0.01
FRACTION_AUDIT_MIN = 0.95

OPTIONS_REQUISES = (
    "Threads",
    "Hash",
    "UCI_LimitStrength",
    "UCI_ShowWDL",
    "SyzygyPath",
)
# python-chess remet MultiPV a 1 et Ponder a false pour chaque analyse : ces
# options sont verifiees mais jamais configurees ici.
OPTIONS_GEREES = ("MultiPV", "Ponder")

OPTIONS_REPRODUCTIBLES = {
    "Threads": 1,
    "Hash": 128,
    "UCI_LimitStrength": False,
    "UCI_ShowWDL": True,
    "SyzygyPath": "",
}

# Jeton de partie unique : python-chess envoie ucinewgame quand le jeton
# change. Le Clear Hash explicite avant chaque recherche resetant deja l'etat,
# reutiliser le meme jeton evite un envoi redondant. Mesure du 2026-09-30 sur
# des positions reelles : etiquettes identiques au bit pres, environ 5 pour
# cent de temps gagne.
JEU_ANNOTATION = object()


def configurer_moteur(engine) -> None:
    """Fixe une fois les options de reproductibilite, apres controle.

    Leve si une option requise manque : mieux vaut arreter le build que
    mesurer un moteur dont on ne maitrise pas les reglages.
    """
    options = engine.options
    manquantes = [nom for nom in OPTIONS_REQUISES if nom not in options]
    manquantes += [nom for nom in OPTIONS_GEREES if nom not in options]
    if manquantes:
        raise RuntimeError(
            "options Stockfish absentes : " + ", ".join(sorted(manquantes)))
    engine.configure(dict(OPTIONS_REPRODUCTIBLES))


def identifier_stockfish(engine, chemin_moteur, *,
                         dossier_export=None) -> dict:
    """Nom UCI, hash du binaire et hash de chaque reseau NNUE utilise."""
    chemin = Path(chemin_moteur)
    if not chemin.is_file():
        raise ValueError(f"binaire Stockfish introuvable : {chemin}")
    identite = dict(engine.id)
    if not identite.get("name"):
        raise RuntimeError(
            "Stockfish sans 'id name' : identification impossible")
    return {
        "name": identite["name"],
        "author": identite.get("author", ""),
        "binary_sha256": _sha256_fichier(chemin),
        "networks": _identifier_reseaux(engine, chemin, dossier_export),
    }


def _identifier_reseaux(engine, chemin_moteur: Path, dossier_export) -> list:
    options = engine.options
    noms = sorted(nom for nom in options if nom.startswith("EvalFile"))
    if not noms:
        raise RuntimeError(
            "aucune option EvalFile : le reseau NNUE ne peut pas etre "
            "identifie")
    reseaux = []
    for nom in noms:
        defaut = str(getattr(options[nom], "default", "") or "").strip()
        fichier = _resoudre_reseau(defaut, chemin_moteur)
        if fichier is not None:
            reseaux.append({"option": nom, "name": defaut or fichier.name,
                            "sha256": _sha256_fichier(fichier),
                            "origin": "fichier"})
            continue
        if not defaut:
            raise RuntimeError(
                f"{nom} sans valeur par defaut : reseau non identifiable")
        if dossier_export is None:
            raise RuntimeError(
                f"{nom} ({defaut}) est embarque : fournir dossier_export "
                "pour l'extraire avant de le hacher")
        cible = Path(dossier_export) / f"{nom}-{defaut}"
        cible.parent.mkdir(parents=True, exist_ok=True)
        protocole = getattr(engine, "protocol", None)
        if protocole is None or not hasattr(protocole, "send_line"):
            raise RuntimeError(
                f"{nom} embarque et aucun protocole brut pour l'export")
        protocole.send_line(f"export_net {cible}")
        engine.ping()
        if not cible.is_file():
            raise RuntimeError(
                f"export du reseau {defaut} refuse par le binaire ; "
                "identification impossible")
        reseaux.append({"option": nom, "name": defaut,
                        "sha256": _sha256_fichier(cible), "origin": "export"})
    return reseaux


def _resoudre_reseau(defaut: str, chemin_moteur: Path) -> Path | None:
    if not defaut:
        return None
    for candidat in (Path(defaut), chemin_moteur.parent / defaut):
        if candidat.is_file():
            return candidat
    return None


def _sha256_fichier(chemin, taille_bloc: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(chemin, "rb") as flux:
        for bloc in iter(lambda: flux.read(taille_bloc), b""):
            digest.update(bloc)
    return digest.hexdigest()


# ============================================================
#                        ANNOTATION
# ============================================================

def _vers_plateau(position: Mapping[str, object]) -> chess.Board:
    """Rejoue l'historique complet dans un plateau python-chess."""
    board = chess.Board(str(position["start_fen"]))
    for uci in cast(list[str], position["moves_uci"]):
        board.push_uci(uci)
    return board


def _valider_wdl(wdl: chess.engine.Wdl) -> tuple[int, int, int]:
    trois = (int(wdl.wins), int(wdl.draws), int(wdl.losses))
    if any(valeur < 0 or valeur > 1000 for valeur in trois) \
            or sum(trois) != 1000:
        raise RuntimeError(f"WDL Stockfish invalide : {trois}")
    return trois


def _valider_info(info: Mapping, uci: str) -> None:
    if info.get("lowerbound") or info.get("upperbound"):
        raise RuntimeError("score borne : annotation refusee")
    if "score" not in info or "wdl" not in info:
        raise RuntimeError("annotation sans score ou sans WDL")
    pv = info.get("pv")
    if not pv or pv[0] != chess.Move.from_uci(uci):
        raise RuntimeError(
            "PV absente ou ne commencant pas par le coup impose")
    if "depth" not in info or "nodes" not in info:
        raise RuntimeError("annotation sans profondeur ou sans noeuds")


def _derniere_info_exacte(engine, board, limit, *,
                          root_moves=None,
                          uci_impose: str | None = None,
                          jeu=None) -> Mapping:
    """Derniere ligne exacte d'une analyse bornee par des noeuds.

    Une recherche coupee a la limite de noeuds se termine par une ligne
    bornee (upperbound/lowerbound) : l'iteration en cours n'est pas finie.
    La derniere ligne exacte est celle de l'iteration complete precedente,
    a une profondeur presque identique. On la conserve donc, et une analyse
    sans aucune ligne exacte est refusee plutot que remplacee par une borne.

    `jeu` est le jeton de partie de python-chess : un jeton neuf fait envoyer
    `ucinewgame` avant l'analyse, un jeton stable non. Le `Clear Hash`
    explicite de l'appelant resetant deja l'etat de recherche, un jeton
    stable evite un envoi redondant.
    """
    meilleure = None
    with engine.analysis(board, limit, root_moves=root_moves,
                         game=jeu if jeu is not None else object()) as analyse:
        for info in analyse:
            if "score" not in info or "wdl" not in info:
                continue
            if info.get("lowerbound") or info.get("upperbound"):
                continue
            if uci_impose is not None:
                pv = info.get("pv")
                if not pv or pv[0] != chess.Move.from_uci(uci_impose):
                    continue
            meilleure = info
    if meilleure is None:
        raise RuntimeError(
            "analyse sans ligne exacte : annotation refusee")
    return meilleure


def _annoter_coup(engine, position: Mapping[str, object], uci: str,
                  index: int, nodes: int) -> MoveLabel:
    board = _vers_plateau(position)
    coup = chess.Move.from_uci(uci)
    if coup not in board.legal_moves:
        raise ValueError(f"coup {uci} illegal dans la position a annoter")

    # Clear Hash avant chaque recherche : le moteur repart d'un etat propre,
    # quel que soit l'ordre des taches. Le jeton de partie reste stable, le
    # ucinewgame qu'il declencherait etant redondant avec ce Clear Hash.
    engine.configure({"Clear Hash": None})
    info = _derniere_info_exacte(
        engine, board, chess.engine.Limit(nodes=nodes),
        root_moves=[coup], uci_impose=uci, jeu=JEU_ANNOTATION)
    _valider_info(info, uci)

    score = info["score"].pov(board.turn)
    wdl = _valider_wdl(info["wdl"].pov(board.turn))
    if score.is_mate():
        cp, mate = None, int(score.mate())
    else:
        cp, mate = int(score.score()), None

    label: MoveLabel = {
        "uci": uci,
        "index": index,
        "wdl": wdl,
        "score": wdl_score(wdl),
        "cp": cp,
        "mate": mate,
        "depth": int(info["depth"]),
        "nodes": int(info["nodes"]),
    }
    validate_move_label(label)
    return label


def annotate_move(engine, position: Mapping[str, object], uci: str,
                  nodes: int) -> MoveLabel:
    """Analyse un coup legal impose, du point de vue du joueur au trait."""
    board = replay_position(position)
    index = uci_to_index(board, uci)
    if index not in set(board.get_legal_move_indices()):
        raise ValueError(f"coup {uci} absent des coups legaux du plateau")
    return _annoter_coup(engine, position, uci, index, nodes)


def annotate_position(engine, position: Mapping[str, object],
                      nodes: int) -> AnnotatedPosition:
    """Analyse tous les coups legaux et rend la position annotee."""
    board = replay_position(position)
    indices = sorted(board.get_legal_move_indices())
    labels = [
        _annoter_coup(engine, position, index_to_uci(board, index), index,
                      nodes)
        for index in indices
    ]
    s_best = max(label["score"] for label in labels)
    resultat = dict(position)
    resultat["labels"] = labels
    resultat["s_best"] = s_best
    resultat["wdl_bucket"] = wdl_category(s_best)
    annotee = cast(AnnotatedPosition, resultat)
    validate_position(annotee)
    return annotee


def screen_position(engine, position: Mapping[str, object], *,
                    nodes: int = NODES_CRIBLAGE) -> float:
    """Esperance de resultat du criblage, du point de vue du joueur au trait."""
    board = _vers_plateau(position)
    engine.configure({"Clear Hash": None})
    info = _derniere_info_exacte(engine, board,
                                 chess.engine.Limit(nodes=nodes),
                                 jeu=JEU_ANNOTATION)
    wdl = _valider_wdl(info["wdl"].pov(board.turn))
    return wdl_score(wdl)


def audit_labels(base: list[AnnotatedPosition],
                 deeper: list[AnnotatedPosition]) -> dict:
    """Controle de stabilite des etiquettes entre deux budgets.

    Le meilleur coup de base est departage par le plus petit index. Son regret
    est mesure sur l'analyse profonde, et la variation de S* est prise en
    valeur absolue : aucune compensation entre hausses et baisses.
    """
    par_id = {position["position_id"]: position for position in deeper}
    positions = 0
    regret_ok = 0
    variations = []
    for reference in base:
        profonde = par_id.get(reference["position_id"])
        if profonde is None:
            raise ValueError("audit : position absente de la passe profonde")
        scores_base = {label["index"]: float(label["score"])
                       for label in reference["labels"]}
        scores_profond = {label["index"]: float(label["score"])
                          for label in profonde["labels"]}
        if scores_base.keys() != scores_profond.keys():
            raise ValueError("audit : couverture des coups differente")
        meilleur = min(index for index, score in scores_base.items()
                       if abs(score - max(scores_base.values())) <= TOLERANCE)
        regret = max(scores_profond.values()) - scores_profond[meilleur]
        regret_ok += regret <= SEUIL_REGRET_AUDIT + TOLERANCE
        variations.append(
            abs(float(reference["s_best"]) - float(profonde["s_best"])))
        positions += 1

    fraction = regret_ok / positions if positions else 0.0
    variation = sum(variations) / positions if positions else 0.0
    return {
        "count": positions,
        "fraction_regret_ok": fraction,
        "variation_moyenne": variation,
        "passed": bool(positions and fraction >= FRACTION_AUDIT_MIN
                       and variation <= SEUIL_VARIATION_AUDIT + TOLERANCE),
    }


# ============================================================
#                     CACHE DE TRAVAIL
# ============================================================

def cle_criblage(position: Mapping[str, object]) -> str:
    return f"screen|{position['position_id']}|{NODES_CRIBLAGE}"


def cle_position(position: Mapping[str, object], nodes: int) -> str:
    return f"position|{position['position_id']}|{nodes}"


def cle_coup(position: Mapping[str, object], uci: str, nodes: int) -> str:
    return f"move|{position['position_id']}|{uci}|{nodes}"


def _hash_config(config: Mapping) -> str:
    charge = json.dumps(config, sort_keys=True, separators=(",", ":"),
                        default=str).encode()
    return hashlib.sha256(charge).hexdigest()


class AnnotationStore:
    """Cache SQLite des annotations, repris apres interruption.

    Une seule connexion en ecriture est tenue par le coordinateur ; les
    travailleurs renvoient leurs resultats complets, jamais des morceaux. La
    configuration du cache (moteur, reseaux, budgets, schema) est figee a la
    creation : la reouvrir avec une autre configuration leve.
    """

    def __init__(self, work_dir, config: Mapping):
        self._dossier = Path(work_dir)
        self._dossier.mkdir(parents=True, exist_ok=True)
        self._chemin = self._dossier / "annotations.sqlite"
        self._config_hash = _hash_config(config)
        self._connection = sqlite3.connect(self._chemin)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "cle TEXT PRIMARY KEY, valeur TEXT NOT NULL)")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS annotation ("
            "cle TEXT PRIMARY KEY, valeur TEXT NOT NULL)")
        self._verifier_config()

    def _verifier_config(self) -> None:
        ligne = self._connection.execute(
            "SELECT valeur FROM meta WHERE cle = 'config'").fetchone()
        if ligne is None:
            self._connection.execute(
                "INSERT INTO meta (cle, valeur) VALUES ('config', ?)",
                (self._config_hash,))
            self._connection.commit()
        elif ligne[0] != self._config_hash:
            self._connection.close()
            raise ValueError(
                "cache d'annotation ouvert avec une autre configuration ; "
                "reprendre avec les memes parametres ou repartir d'un "
                "dossier neuf")

    def get(self, key: str):
        ligne = self._connection.execute(
            "SELECT valeur FROM annotation WHERE cle = ?", (key,)).fetchone()
        if ligne is None:
            return None
        return json.loads(ligne[0])

    def put(self, key: str, result: Any) -> None:
        charge = json.dumps(result, sort_keys=True, separators=(",", ":"))
        self._connection.execute(
            "INSERT OR REPLACE INTO annotation (cle, valeur) VALUES (?, ?)",
            (key, charge))
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False
