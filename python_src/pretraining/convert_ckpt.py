
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import torch
from torch.amp import GradScaler
from core.model import ChessNet


def convert_lightning_to_pt(ckpt_path, output_pt_path, num_res_blocks=10, num_filters=128):
    print(f"Lecture du checkpoint Lightning : {ckpt_path}")

    # 1. Chargement du fichier généré par Lightning
    lightning_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lightning_state_dict = lightning_ckpt["state_dict"]

    # 2. Nettoyage des préfixes "model."
    clean_state_dict = {}
    for key, value in lightning_state_dict.items():
        if key.startswith("model."):
            clean_key = key[6:]  # On enlève les 6 premiers caractères ("model.")
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value

    # 3. Vérification de la compatibilité avec l'architecture
    print("Vérification de l'intégrité des poids...")
    dummy_model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    dummy_model.load_state_dict(clean_state_dict)

    # 4. Instanciation des états vierges pour AdamW et le Scaler
    dummy_optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)
    dummy_scaler = GradScaler("cuda", enabled=True)

    # 5. Création du dictionnaire final
    final_checkpoint = {
        "model_state_dict": clean_state_dict,
        "optimizer_state_dict": dummy_optimizer.state_dict(),
        "scaler_state_dict": dummy_scaler.state_dict(),
        "iteration": 0,
        "global_step": 0
    }

    torch.save(final_checkpoint, output_pt_path)
    print(f"Conversion réussie ! Nouveau fichier prêt pour le Self-Play : {output_pt_path}")


if __name__ == "__main__":

    from core.data_paths import CHECKPOINTS_DIR
    INPUT_CKPT = CHECKPOINTS_DIR / "2026_04_07_00h05_SUPERVISED_GM_NEW_MODEL.ckpt"
    OUTPUT_PT = CHECKPOINTS_DIR / "2026_04_07_00h10_UNSUPERVISED_NEW_MODEL.pt"

    convert_lightning_to_pt(INPUT_CKPT, OUTPUT_PT)
