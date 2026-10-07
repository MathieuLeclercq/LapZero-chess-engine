"""Compare deux passages du banc de puzzles, apparies par numero de ligne.

Compare le meme puzzle joue par deux modeles et rapporte la resolution
appariee (McNemar exact), l'ecart median des priors, de la value et de la
part de visites. Les deux fichiers doivent venir de `puzzle_bench.py`, sur le
meme banc et au meme budget de recherche.

Outil de diagnostic, hors validation.

Usage :

    python dev_tools/compare_bench_models.py out/hist-500.csv out/puzzle506-500.csv
"""
import argparse
import csv
import statistics
from math import comb
from pathlib import Path

COLONNE_REUSSITE = "reussi_recherche"
COLONNES_CONTINUES = (
    ("p_correct_reseau", "prior du coup solution"),
    ("value_reseau", "value du reseau"),
    ("part_visites_correct", "part de visites du coup solution"),
)


def charger(chemin):
    """Rend les lignes du CSV indexees par leur numero de puzzle."""
    with open(chemin, newline="", encoding="utf-8") as fichier:
        return {int(ligne["ligne"]): ligne for ligne in csv.DictReader(fichier)}


def booleen(valeur):
    return valeur.strip().lower() == "true"


def mcnemar_exact(gauche_seul, droite_seul):
    """p bilaterale exacte de McNemar, sans correction de continuite."""
    total = gauche_seul + droite_seul
    if total == 0:
        return 1.0
    extrem = min(gauche_seul, droite_seul)
    p = 2 * sum(comb(total, i) for i in range(extrem + 1)) * 0.5 ** total
    return min(1.0, p)


def comparer(reference, candidat):
    communs = sorted(set(reference) & set(candidat))
    resultat = {
        "reference_lignes": len(reference),
        "candidat_lignes": len(candidat),
        "communs": len(communs),
        "erreurs": sum(
            1 for ligne in communs
            if reference[ligne]["erreur"].strip()
            or candidat[ligne]["erreur"].strip()),
        "medias": {},
    }

    reference_ok = [booleen(reference[ligne][COLONNE_REUSSITE])
                    for ligne in communs]
    candidat_ok = [booleen(candidat[ligne][COLONNE_REUSSITE])
                   for ligne in communs]
    reference_seul = sum(
        1 for a, b in zip(reference_ok, candidat_ok) if a and not b)
    candidat_seul = sum(
        1 for a, b in zip(reference_ok, candidat_ok) if b and not a)
    resultat.update({
        "reference_reussis": sum(reference_ok),
        "candidat_reussis": sum(candidat_ok),
        "reference_seul": reference_seul,
        "candidat_seul": candidat_seul,
        "p_mcnemar": mcnemar_exact(reference_seul, candidat_seul),
    })

    for colonne, _ in COLONNES_CONTINUES:
        valeurs_reference = [float(reference[ligne][colonne])
                             for ligne in communs]
        valeurs_candidat = [float(candidat[ligne][colonne])
                            for ligne in communs]
        ecarts = [y - x for x, y in zip(valeurs_reference, valeurs_candidat)]
        resultat["medias"][colonne] = (
            statistics.median(valeurs_reference),
            statistics.median(valeurs_candidat),
            statistics.median(ecarts) if ecarts else 0.0,
        )
    return resultat


def libelle(chemin):
    """Nom court d'un passage, dossier parent inclus pour lever l'ambiguite."""
    chemin = Path(chemin)
    return f"{chemin.parent.name}/{chemin.name}" if chemin.parent.name else chemin.name


def message_perimetre(resultat, gauche, droite):
    """Signale les lignes ecartees quand les deux passages ne couvrent pas le meme ensemble."""
    lignes_gauche = resultat["reference_lignes"]
    lignes_droite = resultat["candidat_lignes"]
    if resultat["communs"] >= max(lignes_gauche, lignes_droite):
        return None
    if lignes_gauche != lignes_droite:
        return (f"Tailles differentes : {gauche} {lignes_gauche} lignes, "
                f"{droite} {lignes_droite} lignes ; comparaison sur les "
                f"{resultat['communs']} puzzles communs.")
    return (f"Puzzles communs : {resultat['communs']} sur {lignes_gauche} lignes "
            "de chaque cote ; les lignes non communes sont ecartees.")


def formater(resultat, reference, candidat):
    gauche = libelle(reference)
    droite = libelle(candidat)
    lignes = [f"Puzzles apparies : {resultat['communs']}"]
    perimetre = message_perimetre(resultat, gauche, droite)
    if perimetre:
        lignes.append(perimetre)
    lignes += [
        f"Lignes en erreur : {resultat['erreurs']}",
        f"Resolution recherche : {gauche} {resultat['reference_reussis']}"
        f"/{resultat['communs']}"
        f" | {droite} {resultat['candidat_reussis']}/{resultat['communs']}",
        f"Discordants : {gauche} seul {resultat['reference_seul']}, "
        f"{droite} seul {resultat['candidat_seul']}",
        f"McNemar exact bilateral : p = {resultat['p_mcnemar']:.4f}",
    ]
    for colonne, texte in COLONNES_CONTINUES:
        valeur_gauche, valeur_droite, ecart = resultat["medias"][colonne]
        lignes.append(
            f"{texte} : mediane {gauche} {valeur_gauche:+.4f}, "
            f"{droite} {valeur_droite:+.4f}, ecart median {ecart:+.4f}")
    return "\n".join(lignes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="CSV du modele de reference")
    parser.add_argument("candidat", help="CSV du modele candidat")
    args = parser.parse_args()

    reference = charger(args.reference)
    candidat = charger(args.candidat)
    resultat = comparer(reference, candidat)
    print(formater(resultat, args.reference, args.candidat))


if __name__ == "__main__":
    main()
