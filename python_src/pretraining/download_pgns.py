
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
import requests
from bs4 import BeautifulSoup
import zipfile
import io
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
PAGE_URL = "https://www.pgnmentor.com/files.html"
BASE_URL = "https://www.pgnmentor.com/"
OUTPUT_DIR = "raw_pgns"
MAX_PLAYERS = 100

# Ajout d'un User-Agent standard pour se faire passer pour un navigateur
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/122.0.0.0 Safari/537.36'
}


def download_and_extract_pgns():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Analyse de la page {PAGE_URL}...")

    # Injection du paramètre headers
    response = requests.get(PAGE_URL, headers=HEADERS, verify=False)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', href=True)
    player_links = [a['href'] for a in links if
                    a['href'].startswith('players/') and a['href'].endswith('.zip')]

    player_links = list(dict.fromkeys(player_links))[:MAX_PLAYERS]

    print(f"{len(player_links)} archives zip trouvées. Début du téléchargement...\n")

    for i, link in enumerate(player_links, 1):
        file_url = BASE_URL + link
        filename = link.split('/')[-1]

        print(f"[{i}/{len(player_links)}] Téléchargement de {filename}...")
        try:
            # Injection du paramètre headers ici aussi
            r = requests.get(file_url, stream=True, headers=HEADERS, verify=False)
            r.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(OUTPUT_DIR)

            print(f" -> Fichier(s) extrait(s) dans {OUTPUT_DIR}/")
        except Exception as e:
            print(f" -> Erreur sur {filename} : {e}")


if __name__ == "__main__":
    download_and_extract_pgns()