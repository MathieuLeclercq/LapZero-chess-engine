"""Construction, selection et publication du banc externe de positions.

Ce module contient le noyau pur : selection par quotas (plus grand deficit
relatif, plafonds evenement et joueur, equilibre des couleurs), sous-banc de
recherche par plus grands restes, echographie d'audit, ecriture canonique
JSONL/zstd et manifeste. La CLI orchestre les cinq etapes : extract, screen,
annotate, finalize, build.

Regles de la conception (sections 5 et 6) :

- 10 000 positions, une seule par partie, neuf quotas phase x categorie WDL ;
- au moins 45 % de positions avec les blancs au trait, au plus 55 % ;
- au plus 100 positions par evenement, 20 apparitions par joueur, toutes
  couleurs confondues ;
- sous-banc MCTS de 256 identifiants, proportions arrondies par plus grands
  restes, hash independant ;
- dataset v1 immuable : une version differente ne peut pas ecraser la
  precedente ; le JSONL est trie, compresse a parametres fixes, et le
  manifeste est publie en dernier.

Voir docs/etudes/progression-banc-positions/protocole/2026-09-29-position-bench-design.md (sections 4
a 6) et docs/superpowers/plans/2026-09-30-position-bench.md (tache 5).
"""

from __future__ import annotations

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import chess.engine
import zstandard as zstd
from evaluation.position_bench_metrics import (
    SearchConfig,
    SourceInfo,
    validate_manifest,
    wdl_category,
)
from evaluation.position_bench_sources import (
    ATTRIBUTION_BROADCASTS,
    DATE_MINIMUM,
    LICENCE_BROADCASTS,
    barre_progression,
    dedupliquer_candidats,
    download_archive,
    extract_candidates,
    filtrer_candidats_non_terminaux,
    read_games,
)
from evaluation.position_bench_teacher import (
    NODES_ANNOTATION,
    NODES_AUDIT,
    NODES_CRIBLAGE,
    SCHEMA_STORE,
    AnnotationStore,
    annotate_position,
    audit_labels,
    cle_criblage,
    cle_position,
    configurer_moteur,
    identifier_stockfish,
    screen_position,
)

TAILLE_BANC = 10_000
TAILLE_SOUS_BANC = 256
FACTEUR_RESERVE = 1.25
EVENT_CAP = 100
PLAYER_CAP = 20
AUDIT_COUNT = 500
ZSTD_NIVEAU = 19
BUILDER_REVISION = "lapzero-position-bench-builder-v1-cached-audit-repair-20261001"

PHASES = ("ouverture", "milieu", "finale")
BUCKETS = ("disputee", "avantage", "decisive")
CELLULES = tuple((phase, bucket) for phase in PHASES for bucket in BUCKETS)

QUOTAS = {
    ("ouverture", "disputee"): 1100,
    ("ouverture", "avantage"): 500,
    ("ouverture", "decisive"): 400,
    ("milieu", "disputee"): 3600,
    ("milieu", "avantage"): 1000,
    ("milieu", "decisive"): 1400,
    ("finale", "disputee"): 1250,
    ("finale", "avantage"): 250,
    ("finale", "decisive"): 500,
}

SEL_FINAL = "lapzero-position-bench-v1-final"
SEL_RESERVE = "lapzero-position-bench-v1-reserve"
SEL_RECHERCHE = "lapzero-position-bench-v1-search"
SEL_AUDIT = "lapzero-position-bench-v1-audit"
SEL_PILOTE = "lapzero-position-bench-v1-pilot"

URL_BROADCASTS = ("https://database.lichess.org/broadcast/"
                  "lichess_db_broadcast_{mois}.pgn.zst")


class SelectionIncomplete(RuntimeError):
    """Aucune selection complete ne peut etre formee avec ce vivier."""


class AuditEchoue(RuntimeError):
    """L'audit de stabilite des etiquettes a echoue ; v1 n'est pas publie."""


def _hash_cle(position_id: str, sel: str) -> str:
    return hashlib.sha256(f"{position_id}|{sel}".encode()).hexdigest()


def select_positions(records, quotas, *, salt: str, event_cap: int,
                     player_cap: int, rapport: dict | None = None) -> list:
    """Selection gloutonne par plus grand deficit relatif.

    A chaque tour, la case dont le deficit relatif est le plus grand recoit le
    premier candidat admissible dans son ordre de hash. Un candidat est refuse
    s'il fait depasser un plafond d'evenement ou de joueur, s'il apporte une
    seconde position de la meme partie, ou s'il fait depasser 55 % du total
    vise a une couleur. Le rapport distingue une selection complete d'une
    selection ou il manque des positions : aucun filtre n'est relache en
    silence.
    """
    par_case = {}
    for record in records:
        cle = (record["phase"], record["wdl_bucket"])
        if cle in quotas:
            par_case.setdefault(cle, []).append(record)
    for liste in par_case.values():
        liste.sort(key=lambda r: _hash_cle(r["position_id"], salt))

    total_vise = sum(quotas.values())
    curseurs = {cle: 0 for cle in quotas}
    epuisees: set = set()
    comptes = {cle: 0 for cle in quotas}
    par_evenement: dict[str, int] = {}
    par_joueur: dict[str, int] = {}
    par_couleur = {0: 0, 1: 0}
    jeux_utilises: set = set()
    choisis = []

    def reste_relatif(cle):
        if cle in epuisees or quotas[cle] == 0:
            return -1.0
        return (quotas[cle] - comptes[cle]) / quotas[cle]

    while len(choisis) < total_vise:
        cle = max(CELLULES, key=reste_relatif)
        if reste_relatif(cle) < 0.0:
            break

        retenu = None
        while curseurs[cle] < len(par_case.get(cle, [])):
            candidat = par_case[cle][curseurs[cle]]
            curseurs[cle] += 1
            if candidat["game_fingerprint"] in jeux_utilises:
                continue
            if par_evenement.get(candidat["event_id"], 0) >= event_cap:
                continue
            if par_joueur.get(candidat["white_id"], 0) >= player_cap:
                continue
            if par_joueur.get(candidat["black_id"], 0) >= player_cap:
                continue
            couleur = candidat["turn"]
            if par_couleur[couleur] + 1 > 0.55 * total_vise:
                continue
            retenu = candidat
            break
        if retenu is None:
            epuisees.add(cle)
            continue

        choisis.append(retenu)
        comptes[cle] += 1
        jeux_utilises.add(retenu["game_fingerprint"])
        par_evenement[retenu["event_id"]] = (
            par_evenement.get(retenu["event_id"], 0) + 1)
        par_joueur[retenu["white_id"]] = (
            par_joueur.get(retenu["white_id"], 0) + 1)
        par_joueur[retenu["black_id"]] = (
            par_joueur.get(retenu["black_id"], 0) + 1)
        par_couleur[retenu["turn"]] += 1

    manquants = {
        f"{phase}/{bucket}": quotas[(phase, bucket)] - comptes[(phase, bucket)]
        for phase, bucket in CELLULES
        if quotas[(phase, bucket)] > comptes[(phase, bucket)]
    }
    total = len(choisis)
    couleur_ok = bool(total) and all(
        par_couleur[couleur] >= 0.45 * total for couleur in (0, 1))
    if rapport is not None:
        rapport["complet"] = not manquants and couleur_ok
        rapport["manquants"] = manquants
        rapport["couleurs"] = {"blancs": par_couleur[0], "noirs": par_couleur[1]}
        rapport["couleur_ok"] = couleur_ok
    return choisis


def quotas_sous_banc() -> dict:
    """Neuf quotas arrondis a 256 par plus grands restes, ordre des cases."""
    total = sum(QUOTAS.values())
    exacts = {cle: quota * TAILLE_SOUS_BANC / total
              for cle, quota in QUOTAS.items()}
    arrondis = {cle: int(valeur) for cle, valeur in exacts.items()}
    reste = TAILLE_SOUS_BANC - sum(arrondis.values())
    ordre = sorted(
        CELLULES,
        key=lambda cle: (-(exacts[cle] - arrondis[cle]), CELLULES.index(cle)))
    for cle in ordre[:reste]:
        arrondis[cle] += 1
    return arrondis


def select_search_ids(records) -> list[str]:
    """256 identifiants du sous-banc, par cellule puis hash independant."""
    quotas = quotas_sous_banc()
    par_case = {}
    for record in records:
        cle = (record["phase"], record["wdl_bucket"])
        par_case.setdefault(cle, []).append(record)
    resultat = []
    for cle in CELLULES:
        liste = sorted(
            par_case.get(cle, []),
            key=lambda r: _hash_cle(r["position_id"], SEL_RECHERCHE))
        if len(liste) < quotas[cle]:
            raise SelectionIncomplete(
                f"sous-banc : {cle[0]}/{cle[1]} a {len(liste)} candidats "
                f"pour {quotas[cle]}")
        resultat.extend(r["position_id"] for r in liste[:quotas[cle]])
    return resultat


def select_audit_ids(records, count: int = AUDIT_COUNT) -> list[str]:
    """Positions de l'echantillon d'audit, par hash independant."""
    ordonnes = sorted(records,
                      key=lambda r: _hash_cle(r["position_id"], SEL_AUDIT))
    return [r["position_id"] for r in ordonnes[:count]]


def finaliser(selection, deeper, *,
              audit_count: int = AUDIT_COUNT) -> tuple[list, dict]:
    """Sous-banc et audit de la selection ; leve si l'audit echoue.

    `deeper` contient les memes positions annotees au budget d'audit.
    """
    search_ids = select_search_ids(selection)
    ids_audit = select_audit_ids(selection, audit_count)
    # La base garde ses scores de production, la passe profonde les siens :
    # le controle croise les deux jeux d'etiquettes position par position.
    base = {position["position_id"]: position for position in selection}
    profonde = {position["position_id"]: position for position in deeper}
    audit = audit_labels([base[i] for i in ids_audit],
                         [profonde[i] for i in ids_audit])
    if not audit["passed"]:
        raise AuditEchoue(json.dumps(audit, sort_keys=True))
    return search_ids, audit


def assembler_manifeste(*, dataset_version: str, sources, selection,
                        search_ids, stockfish: dict, audit: dict,
                        quotas=None, created: str | None = None) -> dict:
    """Manifeste de publication ; le hash du dataset est pose a l'ecriture."""
    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "dataset_sha256": "0" * 64,
        "training_forbidden": True,
        "sources": [dict(source) for source in sources],
        "counts": {"positions": len(selection),
                   "search": len(search_ids)},
        "quotas": _quotas_json(QUOTAS if quotas is None else quotas),
        "stockfish": stockfish,
        "audit": audit,
        "search_ids": list(search_ids),
        "protocol_defaults": {
            "search": asdict(SearchConfig()),
            "policy_batch_size": 1024,
            "target_s": 300.0,
        },
        "builder_revision": BUILDER_REVISION,
        "created": created or datetime.now(UTC).isoformat(),
    }


def _selection_complete(records) -> list:
    rapport: dict = {}
    selection = select_positions(records, QUOTAS, salt=SEL_FINAL,
                                 event_cap=EVENT_CAP, player_cap=PLAYER_CAP,
                                 rapport=rapport)
    if not rapport["complet"]:
        raise SelectionIncomplete(
            "selection incomplete : "
            + json.dumps(rapport["manquants"], sort_keys=True))
    return selection


def finaliser_depuis_audit(records, deeper) -> tuple[list, list, dict]:
    """Promeut l'audit existant sans le presenter comme un nouvel audit reussi.

    Les scores d'origine servent au rapport de stabilite. La selection finale
    est ensuite recalculee avec les annotations profondes, et le sous-banc de
    recherche est stratifie uniquement dans les positions ainsi corrigees.
    Les listes fournies ne sont pas modifiees.
    """
    base = _selection_complete(records)
    ids_audit = select_audit_ids(base, AUDIT_COUNT)
    profonds = {position["position_id"]: position for position in deeper}
    if (len(ids_audit) != AUDIT_COUNT or len(profonds) != len(deeper)
            or set(profonds) != set(ids_audit)):
        raise ValueError("reparation : audit incomplet, duplique ou incompatible")
    par_id = {position["position_id"]: position for position in base}
    champs_annotation = {"labels", "s_best", "wdl_bucket", "annotation_budget_nodes"}
    for position_id in ids_audit:
        reference = par_id[position_id]
        profonde = profonds[position_id]
        for cle, valeur in reference.items():
            if cle not in champs_annotation and profonde.get(cle) != valeur:
                raise ValueError(f"reparation : provenance differente ({cle})")

    # Ne pas comparer les etiquettes corrigees avec elles-memes : ce rapport
    # reste celui du controle independant effectue AVANT leur promotion.
    audit = audit_labels([par_id[i] for i in ids_audit],
                         [profonds[i] for i in ids_audit])
    enrichis = [dict(profonds.get(position["position_id"], position),
                     annotation_budget_nodes=(
                         NODES_AUDIT if position["position_id"] in profonds
                         else NODES_ANNOTATION))
                for position in records]
    selection = _selection_complete(enrichis)
    pool = [position for position in selection
            if position["position_id"] in profonds]
    search_ids = select_search_ids(pool)
    audit.update({
        "accepted_with_warning": not audit["passed"],
        "acceptance_reason": "explicit_cached_audit_repair",
        "comparison": "original_labels_vs_cached_audit",
        "post_repair_stability_verified": False,
        "original_audit_ids": ids_audit,
        "source_annotation_nodes": NODES_ANNOTATION,
        "audit_nodes": NODES_AUDIT,
        "repaired_positions": len(profonds),
        "repaired_positions_in_dataset": len(pool),
        "search_reference_nodes": NODES_AUDIT,
        "search_pool_count": len(pool),
    })
    return selection, search_ids, audit


def _quotas_json(quotas) -> dict:
    return {f"{phase}/{bucket}": quotas[(phase, bucket)]
            for phase, bucket in CELLULES}


def est_productif(manifeste) -> bool:
    """Vrai seulement pour un banc complet, donc publiable et entrainable."""
    counts = manifeste.get("counts", {})
    return (counts.get("positions") == TAILLE_BANC
            and counts.get("search") == TAILLE_SOUS_BANC)


def _jsonl_compresse(records) -> bytes:
    lignes = [
        json.dumps(record, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False)
        for record in sorted(records, key=lambda r: r["position_id"])
    ]
    charge = ("\n".join(lignes) + "\n").encode("utf-8")
    return zstd.ZstdCompressor(level=ZSTD_NIVEAU).compress(charge)


def _ecrire_atomique(chemin: Path, donnees: bytes) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    with open(temporaire, "wb") as flux:
        flux.write(donnees)
    os.replace(temporaire, chemin)


def ecrire_jsonl_zst(records, chemin) -> None:
    _ecrire_atomique(Path(chemin), _jsonl_compresse(records))


def lire_jsonl_zst(chemin) -> list:
    decompresseur = zstd.ZstdDecompressor()
    with open(chemin, "rb") as brut, io.TextIOWrapper(
            decompresseur.stream_reader(brut), encoding="utf-8") as flux:
        return [json.loads(ligne) for ligne in flux if ligne.strip()]


def _readme(manifeste) -> str:
    sources = ", ".join(source["source_id"] for source in manifeste["sources"])
    audit = manifeste["audit"]
    avertissement = ""
    if audit.get("acceptance_reason") == "explicit_cached_audit_repair":
        avertissement = (
            "\n## Reparation depuis l'audit en cache\n\n"
            "Le rapport d'audit conserve les mesures sur les etiquettes "
            "d'origine, avant correction. Son champ `passed` n'est pas "
            "falsifie : un echec est accepte explicitement avec avertissement. "
            "Aucun nouvel audit independant apres correction n'a ete execute.\n\n"
            f"- Positions corrigees dans le banc : {audit['repaired_positions_in_dataset']}\n"
            f"- Budget des references MCTS : {audit['search_reference_nodes']} noeuds par coup legal\n"
            "- Budget demande par position : `annotation_budget_nodes` ; "
            "les `nodes` des coups sont les nombres de noeuds effectivement examines.\n"
        )
    return (
        "# Banc externe de positions, "
        f"{manifeste['dataset_version']}\n\n"
        "Positions extraites des archives de broadcasts Lichess "
        "(CC BY-SA 4.0, attribution Lichess), annotees par Stockfish hors "
        "ligne. Ce banc est interdit a l'entrainement : "
        "`training_forbidden` vaut true dans le manifeste, et aucun chargeur "
        "de donnees ne le decouvre automatiquement.\n\n"
        f"- Sources : {sources}\n"
        f"- Positions : {manifeste['counts']['positions']}\n"
        f"- Identifiants de recherche : {manifeste['counts']['search']}\n"
        f"- SHA-256 du JSONL compresse : {manifeste['dataset_sha256']}\n"
        f"- Moteur d'annotation : {manifeste['stockfish'].get('name', '?')}\n\n"
        "Les identifiants de recherche sont la liste explicite du manifeste ; "
        "l'ordre des positions du fichier est celui des `position_id` "
        "croissants.\n"
    ) + avertissement


def write_dataset(records, manifest, output) -> Path:
    """Ecrit le JSONL, le README puis le manifeste, dans cet ordre.

    Une version existante avec le meme hash est rendue telle quelle ; une
    version differente est refusee, jamais ecrasee.
    """
    dossier = Path(output)
    compresse = _jsonl_compresse(records)
    digest = hashlib.sha256(compresse).hexdigest()
    manifeste = dict(manifest)
    manifeste["dataset_sha256"] = digest
    validate_manifest(manifeste)

    chemin_manifeste = dossier / "manifest.json"
    if chemin_manifeste.is_file():
        existant = json.loads(chemin_manifeste.read_text(encoding="utf-8"))
        if existant.get("dataset_sha256") != digest:
            raise FileExistsError(
                f"une version differente existe deja : {dossier}")
        return chemin_manifeste

    _ecrire_atomique(dossier / "positions.jsonl.zst", compresse)
    _ecrire_atomique(dossier / "README.md", _readme(manifeste).encode("utf-8"))
    _ecrire_atomique(chemin_manifeste, json.dumps(
        manifeste, sort_keys=True, indent=2, ensure_ascii=False).encode(
            "utf-8"))
    return chemin_manifeste


# ============================================================
#                            CLI
# ============================================================

def _parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_position_bench")
    parser.add_argument(
        "commande",
        choices=["extract", "screen", "annotate", "finalize", "build"])
    parser.add_argument("--work-dir", default="data/position_bench_work")
    parser.add_argument("--output", default="data/position_bench/v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--months", nargs="+",
                        default=["2026-08", "2026-07", "2026-06"],
                        help="mois initiaux ; build ajoute les mois precedents "
                             "si les quotas sont incomplets (depuis 2026-03)")
    parser.add_argument("--stockfish")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-count", type=int, default=0)
    parser.add_argument("--stage", choices=["reserve", "final"], default="final")
    parser.add_argument(
        "--repair-from-cached-audit", action="store_true",
        help="finalize --stage final uniquement : reutiliser l'audit complet "
             "en cache, accepter explicitement son avertissement et choisir "
             "le sous-banc MCTS dans les positions corrigees ; aucune recherche")
    return parser


def main(argv=None) -> int:
    args = _parseur().parse_args(argv)
    try:
        if args.repair_from_cached_audit and (
                args.commande != "finalize" or args.stage != "final"):
            raise ValueError(
                "--repair-from-cached-audit exige finalize --stage final")
        if args.commande == "extract":
            _cmd_extract(args)
        elif args.commande == "screen":
            _cmd_screen(args)
        elif args.commande == "annotate":
            _cmd_annotate(args)
        elif args.commande == "finalize":
            _cmd_finalize(args)
        else:
            _cmd_build(args)
    except (ValueError, RuntimeError, OSError, chess.engine.EngineError) as erreur:
        print(f"erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


def _cmd_build(args) -> None:
    """Etend le vivier sans relacher les quotas ni perdre les analyses.

    Les mois effectifs sont memorises avant chaque extraction pour reprendre
    aussi une interruption au milieu d'un mois ajoute automatiquement.
    Les commandes individuelles gardent leur comportement explicite.
    """
    demandes = sorted(set(args.months), reverse=True)
    _valider_mois(demandes)
    etat = Path(args.work_dir) / "build_months.json"
    args.months = list(demandes)
    if args.resume and etat.is_file():
        sauvegarde = json.loads(etat.read_text(encoding="utf-8"))
        if not isinstance(sauvegarde, dict):
            raise ValueError(f"etat des mois invalide : {etat}")
        if sauvegarde.get("requested_months") == demandes:
            effectifs = sauvegarde.get("months")
            _valider_mois(effectifs)
            if not set(demandes).issubset(effectifs):
                raise ValueError(f"mois initiaux absents de l'etat : {etat}")
            args.months = sorted(set(effectifs), reverse=True)

    while True:
        _ecrire_atomique(etat, json.dumps({
            "requested_months": demandes, "months": args.months,
        }, sort_keys=True, indent=2).encode("utf-8"))
        _cmd_extract(args)
        _cmd_screen(args)
        try:
            args.stage = "reserve"
            _cmd_finalize(args)
            _cmd_annotate(args)
            args.stage = "final"
            _cmd_finalize(args)
            return
        except SelectionIncomplete as erreur:
            plus_ancien = datetime.strptime(min(args.months), "%Y-%m").date()
            precedent = (plus_ancien - timedelta(days=1)).strftime("%Y-%m")
            if precedent < DATE_MINIMUM.strftime("%Y-%m"):
                raise SelectionIncomplete(
                    f"{erreur} ; tous les mois admissibles jusqu'a "
                    f"{min(args.months)} sont utilises. Aucun mois anterieur "
                    "ne sera ajoute pour exclure le corpus de preentrainement. "
                    "Le cache est conserve.") from erreur
            print(f"\n  {erreur}\n  ajout automatique de {precedent} ; "
                  "archives, candidats et analyses precedents conserves",
                  flush=True)
            args.months.append(precedent)
            # Meme sans --resume initial, une extension doit reutiliser le
            # travail fait au tour precedent, et non retelecharger le corpus.
            args.resume = True


def _valider_mois(mois) -> None:
    """Refuse les noms d'archives invalides et le corpus de preentrainement."""
    if not isinstance(mois, list) or not mois:
        raise ValueError("une liste non vide de mois YYYY-MM est requise")
    minimum = DATE_MINIMUM.strftime("%Y-%m")
    for valeur in mois:
        if not isinstance(valeur, str):
            raise ValueError("les mois doivent etre des chaines YYYY-MM")
        date_mois = datetime.strptime(valeur, "%Y-%m")
        if date_mois.strftime("%Y-%m") != valeur:
            raise ValueError(f"mois invalide : {valeur}, format YYYY-MM attendu")
        if valeur < minimum:
            raise ValueError(f"mois {valeur} exclu : minimum {minimum}, "
                             "pour eviter le corpus de preentrainement")


def _cmd_extract(args) -> None:
    dossier = Path(args.work_dir)
    archives = dossier / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    sources = []
    candidats = []
    for mois in args.months:
        url = URL_BROADCASTS.format(mois=mois)
        destination = archives / f"lichess_db_broadcast_{mois}.pgn.zst"
        if args.resume and destination.is_file():
            source = _source_reprise(destination, url, mois)
            print(f"  archive {mois} deja presente, telechargement ignore")
        else:
            source = download_archive(
                url, destination, None,
                progress=barre_progression(f"{mois}"))
            print(f"\n  archive {mois} : "
                  f"{destination.stat().st_size / 1e6:.1f} Mo, sha256 "
                  f"{source['archive_sha256'][:16]}...")
        sources.append(dict(source))

        # Un fichier de candidats par mois : une interruption ne perd que le
        # mois en cours, et une reprise reutilise les mois deja extraits.
        chemin_candidats = dossier / f"candidates_{mois}.jsonl.zst"
        if args.resume and chemin_candidats.is_file():
            extraits = lire_jsonl_zst(chemin_candidats)
            non_terminaux = filtrer_candidats_non_terminaux(extraits)
            if len(non_terminaux) != len(extraits):
                print(f"  candidats {mois} : "
                      f"{len(extraits) - len(non_terminaux)} racines "
                      "terminales retirees du cache")
                extraits = non_terminaux
                ecrire_jsonl_zst(extraits, chemin_candidats)
            print(f"  candidats {mois} deja extraits : {len(extraits)}, "
                  "reutilises")
        else:
            compteurs: dict = {}
            extraits = extract_candidates(
                read_games(destination, source), source,
                compteurs=compteurs,
                progress=_barre_extraction(
                    mois, dossier / f"{mois}.extraction.progress"))
            ecrire_jsonl_zst(extraits, chemin_candidats)
            print(f"\n  {mois} : {len(extraits)} candidats, rejets "
                  f"{compteurs}")
        candidats.extend(extraits)

    candidats = dedupliquer_candidats(candidats)
    ecrire_jsonl_zst(candidats, dossier / "candidates.jsonl.zst")
    _ecrire_atomique(
        dossier / "sources.json",
        json.dumps(sources, sort_keys=True, indent=2).encode("utf-8"))
    print(f"{len(candidats)} candidats extraits de {len(sources)} archives")


def _barre_extraction(mois: str, fichier=None):
    """Etat d'extraction en continu, sur stderr et dans un fichier de suivi."""
    def suivi(etat: dict) -> None:
        ligne = (f"{mois} : {etat['parties_lues']} parties lues, "
                 f"{etat['candidats']} candidats, {etat['rejets']} rejets")
        print(f"\r  {ligne}", end="", file=sys.stderr, flush=True)
        if fichier is not None:
            Path(fichier).write_text(ligne + "\n", encoding="utf-8")

    return suivi


def _source_reprise(destination: Path, url: str, mois: str) -> SourceInfo:
    digest = hashlib.sha256()
    with open(destination, "rb") as flux:
        for bloc in iter(lambda: flux.read(1 << 20), b""):
            digest.update(bloc)
    return SourceInfo(
        source_id=f"lichess-broadcasts-{mois}",
        url=url,
        month=mois,
        archive_sha256=digest.hexdigest(),
        license=LICENCE_BROADCASTS,
        attribution=ATTRIBUTION_BROADCASTS,
    )


def _config_store(identite) -> dict:
    """Configuration figee du cache ; tous les moteurs doivent etre identiques."""
    return {
        "schema": SCHEMA_STORE,
        "engine": identite,
        "budgets": {
            "screen": NODES_CRIBLAGE,
            "annotation": NODES_ANNOTATION,
            "audit": NODES_AUDIT,
        },
    }


def _ouvrir_stockfish(args, identite=None):
    """Ouvre un moteur ; l'identification du reseau n'a lieu qu'une fois."""
    if not args.stockfish:
        raise ValueError("--stockfish est obligatoire pour cette commande")
    moteur = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        configurer_moteur(moteur)
        if identite is None:
            identite = identifier_stockfish(
                moteur, args.stockfish,
                dossier_export=Path(args.work_dir) / "nnue")
    except BaseException:
        _fermer_moteur(moteur)
        raise
    return moteur, identite


def _fermer_moteur(moteur) -> None:
    """Ferme aussi un moteur deja termine, sans masquer l'erreur initiale."""
    try:
        moteur.quit()
    except (chess.engine.EngineError, OSError, TimeoutError) as erreur:
        print(f"  fermeture Stockfish : {erreur}", file=sys.stderr)


def _fermer_moteurs(moteurs) -> None:
    for moteur in moteurs:
        _fermer_moteur(moteur)


def _ouvrir_moteurs(args):
    """Ouvre `--workers` moteurs du meme binaire et rend leur identite.

    L'identification (hash du binaire et export du reseau embarque) est faite
    une seule fois : tous les moteurs viennent du meme chemin, et reexporter
    98 Mo par moteur ne sert a rien.
    """
    moteurs = []
    identite = None
    try:
        for _ in range(max(1, args.workers)):
            moteur, identite = _ouvrir_stockfish(args, identite)
            moteurs.append(moteur)
    except BaseException:
        _fermer_moteurs(moteurs)
        raise
    assert identite is not None
    return moteurs, identite


def _en_parallele(moteurs, elements, traitement, *, a_reception=None,
                  progress=None, reouvrir_moteur=None):
    """Une tache active par moteur ; arret de la soumission au premier echec.

    Une annotation invalide ou un crash est retente une fois sur un moteur neuf
    si la fabrique est fournie. Un second echec est fatal. Seules les taches
    deja actives sont attendues ; leurs resultats reussis restent dans le cache.
    Le coordinateur conserve l'ordre des resultats et toutes les ecritures.
    """
    resultats = [None] * len(elements)
    if not elements:
        return resultats
    if not moteurs:
        raise ValueError("au moins un moteur est requis")

    def tache(index, element):
        try:
            return traitement(moteurs[index], element)
        except (RuntimeError, ValueError, chess.engine.EngineError) as erreur:
            if reouvrir_moteur is None:
                raise
            print(f"  Stockfish worker {index + 1} : {erreur} ; "
                  "une tentative sur un nouveau moteur", file=sys.stderr,
                  flush=True)
            _fermer_moteur(moteurs[index])
            moteurs[index] = reouvrir_moteur()
            return traitement(moteurs[index], element)

    futurs = {}
    a_soumettre = iter(enumerate(elements))
    faites = 0

    def soumettre(executeur, index):
        suivant = next(a_soumettre, None)
        if suivant is not None:
            ordre, element = suivant
            futur = executeur.submit(tache, index, element)
            futurs[futur] = (index, ordre, element)

    def enregistrer(futur, *, suivre=True):
        nonlocal faites
        _, ordre, element = futurs[futur]
        resultat = futur.result()
        if a_reception is not None:
            a_reception(element, resultat)
        resultats[ordre] = resultat
        faites += 1
        if suivre and progress is not None:
            progress({"faites": faites, "total": len(elements)})

    try:
        with ThreadPoolExecutor(max_workers=len(moteurs)) as executeur:
            try:
                for index in range(len(moteurs)):
                    soumettre(executeur, index)
                while futurs:
                    termines, _ = wait(futurs, return_when=FIRST_COMPLETED)
                    libres = []
                    # Traiter tous les achevements avant de soumettre la suite :
                    # une erreur deja disponible ne doit pas remplir la file.
                    for futur in sorted(termines, key=lambda f: futurs[f][1]):
                        enregistrer(futur)
                        index, _, _ = futurs.pop(futur)
                        libres.append(index)
                    for index in libres:
                        soumettre(executeur, index)
            except BaseException:
                for futur in futurs:
                    futur.cancel()
                raise
    except BaseException:
        # Le pool est ferme et les quelques taches actives ont termine. Garder
        # leurs succes, sans perdre l'exception qui a arrete la campagne.
        for futur in futurs:
            if not futur.cancelled() and futur.exception() is None:
                try:
                    enregistrer(futur, suivre=False)
                except Exception as erreur:
                    print(f"  resultat non sauvegarde : {erreur}",
                          file=sys.stderr)
        raise
    return resultats


def _cmd_screen(args) -> None:
    dossier = Path(args.work_dir)
    candidats = lire_jsonl_zst(dossier / "candidates.jsonl.zst")
    moteurs, identite = _ouvrir_moteurs(args)
    store = None
    try:
        store = AnnotationStore(dossier, _config_store(identite))
        a_cribler = [position for position in candidats
                     if store.get(cle_criblage(position)) is None]
        deja = len(candidats) - len(a_cribler)
        print(f"  criblage : {deja} positions deja en cache, "
              f"{len(a_cribler)} a analyser", flush=True)
        _en_parallele(
            moteurs, a_cribler,
            lambda moteur, position: screen_position(moteur, position),
            a_reception=lambda position, note: store.put(
                cle_criblage(position), note),
            reouvrir_moteur=lambda: _ouvrir_stockfish(args, identite)[0],
            progress=_barre_suivi("criblage", reprises=deja,
                                  fichier=dossier / "criblage.progress"))
        enrichis = [
            {**candidat,
             "screen_score": store.get(cle_criblage(candidat))}
            for candidat in candidats
        ]
    finally:
        if store is not None:
            store.close()
        _fermer_moteurs(moteurs)
    ecrire_jsonl_zst(enrichis, dossier / "screened.jsonl.zst")
    print(f"\n{len(enrichis)} positions criblees")


def _cmd_annotate(args) -> None:
    dossier = Path(args.work_dir)
    if args.pilot_count:
        candidats = lire_jsonl_zst(dossier / "candidates.jsonl.zst")
        positions = sorted(
            candidats,
            key=lambda c: _hash_cle(c["position_id"], SEL_PILOTE)
        )[:args.pilot_count]
        dossier = dossier / "pilot"
    else:
        reserve = json.loads(
            (dossier / "reserve.json").read_text(encoding="utf-8"))
        par_id = {c["position_id"]: c
                  for c in lire_jsonl_zst(dossier / "screened.jsonl.zst")}
        positions = [par_id[i] for i in reserve["ids"]]

    moteurs, identite = _ouvrir_moteurs(args)
    store = None
    try:
        store = AnnotationStore(dossier, _config_store(identite))
        deja = sum(
            1 for position in positions
            if store.get(cle_position(position, NODES_ANNOTATION)) is not None)
        print(f"  {deja} positions deja en cache, "
              f"{len(positions) - deja} a annoter")
        annoter_positions(
            moteurs, positions, NODES_ANNOTATION, store,
            dossier / "annotated.jsonl.zst",
            reouvrir_moteur=lambda: _ouvrir_stockfish(args, identite)[0],
            progress=_barre_suivi(
                "pilot" if args.pilot_count else "annotation",
                fichier=dossier / "annotation.progress"))
    finally:
        if store is not None:
            store.close()
        _fermer_moteurs(moteurs)
    print(f"\n{len(positions)} positions disponibles "
          f"({deja} du cache, {len(positions) - deja} annotees maintenant)")


def _barre_suivi(nom: str, *, reprises: int = 0, fichier=None,
                 intervalle_s: float = 2.0):
    """Etat de progression en continu, sur stderr et dans un fichier de suivi.

    L'affichage est limite a une mise a jour toutes les `intervalle_s` : sur
    une campagne de criblage de 100 000 positions, ecrire le fichier a chaque
    position ferait scanner autant de petits fichiers par l'antivirus Windows.
    La derniere position est toujours ecrite.
    """
    dernier = [0.0]

    def suivi(etat: dict) -> None:
        maintenant = time.monotonic()
        derniere_position = etat["faites"] == etat["total"]
        if maintenant - dernier[0] < intervalle_s and not derniere_position:
            return
        dernier[0] = maintenant
        complement = (f" ({reprises} reprises du cache)" if reprises else "")
        ligne = (f"{nom} : {etat['faites']}/{etat['total']} positions"
                 f"{complement}")
        print(f"\r  {ligne}", end="", file=sys.stderr, flush=True)
        if fichier is not None:
            Path(fichier).write_text(ligne + "\n", encoding="utf-8")

    return suivi


def annoter_positions(moteurs, positions, nodes: int, store, sortie, *,
                      progress=None, reouvrir_moteur=None) -> list:
    """Annote les positions en parallele et materialise le resultat.

    Chaque position est analysee seule, dans un moteur configure de facon
    identique : le resultat ne depend donc ni du nombre de moteurs ni de la
    repartition. Le cache SQLite est ecrit par le coordinateur, jamais par les
    travailleurs. `progress` est appele apres chaque position nouvellement
    annotee, pour suivre une campagne longue.
    """
    resultats: list = [None] * len(positions)
    for rang, position in enumerate(positions):
        enregistre = store.get(cle_position(position, nodes))
        if enregistre is not None:
            resultats[rang] = enregistre
    a_faire = [rang for rang, resultat in enumerate(resultats)
               if resultat is None]
    deja = len(positions) - len(a_faire)

    def reception(rang: int, annotee) -> None:
        store.put(cle_position(positions[rang], nodes), annotee)
        resultats[rang] = annotee

    def suivi(etat: dict) -> None:
        if progress is None:
            return
        progress({"faites": etat["faites"], "total": etat["total"],
                  "reprises": deja})

    _en_parallele(
        moteurs, a_faire,
        lambda moteur, rang: annotate_position(moteur, positions[rang], nodes),
        a_reception=reception,
        reouvrir_moteur=reouvrir_moteur,
        progress=suivi if progress is not None else None)
    ecrire_jsonl_zst(resultats, sortie)
    return resultats


def _cmd_finalize(args) -> None:
    dossier = Path(args.work_dir)
    if getattr(args, "repair_from_cached_audit", False):
        _finalize_cached_audit(args, dossier)
    elif args.stage == "reserve":
        _finalize_reserve(dossier)
    else:
        _finalize_final(args, dossier)


def _finalize_cached_audit(args, dossier: Path) -> None:
    """Lit uniquement les analyses existantes et refuse toute analyse manquante."""
    annotes = lire_jsonl_zst(dossier / "annotated.jsonl.zst")
    base = _selection_complete(annotes)
    ids_audit = select_audit_ids(base, AUDIT_COUNT)
    par_id = {position["position_id"]: position for position in base}
    if not (dossier / "annotations.sqlite").is_file():
        raise ValueError("cache d'audit absent ; aucune recherche lancee")
    # Un seul moteur pour son identification (binaire et NNUE), sans analyse.
    # Le fingerprint du cache doit toujours correspondre a ce moteur.
    moteur, identite = _ouvrir_stockfish(args)
    _fermer_moteur(moteur)
    profonds = []
    with AnnotationStore(dossier, _config_store(identite)) as store:
        for position_id in ids_audit:
            annotee = store.get(cle_position(par_id[position_id], NODES_AUDIT))
            if annotee is None:
                raise ValueError(
                    f"audit en cache incomplet : {position_id} ; "
                    "aucune recherche lancee")
            profonds.append(annotee)
    selection, search_ids, audit = finaliser_depuis_audit(annotes, profonds)
    sources = json.loads((dossier / "sources.json").read_text(encoding="utf-8"))
    manifeste = assembler_manifeste(
        dataset_version=args.version, sources=sources, selection=selection,
        search_ids=search_ids, stockfish=identite, audit=audit)
    chemin = write_dataset(selection, manifeste, args.output)
    print(json.dumps({"published": str(chemin), "positions": len(selection),
                      "search": len(search_ids),
                      "repaired_positions": audit["repaired_positions_in_dataset"],
                      "original_audit_passed": audit["passed"],
                      "accepted_with_warning": audit["accepted_with_warning"]},
                     sort_keys=True))


def _finalize_reserve(dossier: Path) -> None:
    enrichis = lire_jsonl_zst(dossier / "screened.jsonl.zst")
    for position in enrichis:
        position["wdl_bucket"] = wdl_category(float(position["screen_score"]))
    quotas = {cle: round(quota * FACTEUR_RESERVE)
              for cle, quota in QUOTAS.items()}
    rapport: dict = {}
    reserve = select_positions(enrichis, quotas, salt=SEL_RESERVE,
                               event_cap=EVENT_CAP, player_cap=PLAYER_CAP,
                               rapport=rapport)
    if not rapport["complet"]:
        raise SelectionIncomplete(
            "reserve incomplete, ajouter le mois precedent : "
            + json.dumps(rapport["manquants"], sort_keys=True))
    _ecrire_atomique(dossier / "reserve.json", json.dumps(
        {"ids": [position["position_id"] for position in reserve]},
        sort_keys=True, indent=2).encode("utf-8"))
    print(f"reserve de {len(reserve)} positions")


def _finalize_final(args, dossier: Path) -> None:
    annotes = lire_jsonl_zst(dossier / "annotated.jsonl.zst")
    rapport: dict = {}
    selection = select_positions(annotes, QUOTAS, salt=SEL_FINAL,
                                 event_cap=EVENT_CAP, player_cap=PLAYER_CAP,
                                 rapport=rapport)
    if not rapport["complet"]:
        raise SelectionIncomplete(
            "selection incomplete, ajouter le mois precedent : "
            + json.dumps(rapport["manquants"], sort_keys=True))

    ids_audit = select_audit_ids(selection)
    par_id = {position["position_id"]: position for position in annotes}
    moteurs, identite = _ouvrir_moteurs(args)
    store = None
    resultats: dict = {}
    a_faire = []
    try:
        store = AnnotationStore(dossier, _config_store(identite))
        for position_id in ids_audit:
            enregistre = store.get(cle_position(par_id[position_id],
                                                NODES_AUDIT))
            if enregistre is None:
                a_faire.append(position_id)
            else:
                resultats[position_id] = enregistre

        def reception(position_id: str, annotee) -> None:
            store.put(cle_position(par_id[position_id], NODES_AUDIT),
                      annotee)
            resultats[position_id] = annotee

        _en_parallele(
            moteurs, a_faire,
            lambda moteur, position_id: annotate_position(
                moteur, par_id[position_id], NODES_AUDIT),
            a_reception=reception,
            reouvrir_moteur=lambda: _ouvrir_stockfish(args, identite)[0],
            progress=_barre_suivi(
                "audit", reprises=len(ids_audit) - len(a_faire),
                fichier=dossier / "audit.progress"))
    finally:
        if store is not None:
            store.close()
        _fermer_moteurs(moteurs)
    profonds = [resultats[position_id] for position_id in ids_audit]

    search_ids, audit = finaliser(selection, profonds)
    sources = json.loads(
        (dossier / "sources.json").read_text(encoding="utf-8"))
    manifeste = assembler_manifeste(
        dataset_version=args.version, sources=sources, selection=selection,
        search_ids=search_ids, stockfish=identite, audit=audit)
    chemin = write_dataset(selection, manifeste, args.output)
    print(f"banc {args.version} publie : {chemin}")


if __name__ == "__main__":
    raise SystemExit(main())
