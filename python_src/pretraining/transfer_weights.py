
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import torch
from core.model import ChessNet


def transfer_weights(old_ckpt_path, new_ckpt_path, num_res_blocks=10, num_filters=128):
    print(f"Chargement de l'ancien checkpoint : {old_ckpt_path}")
    # On charge sur CPU pour éviter de saturer la VRAM inutilement
    checkpoint = torch.load(old_ckpt_path, map_location='cpu', weights_only=True)
    old_state_dict = checkpoint['model_state_dict']

    print("Initialisation de la nouvelle architecture (SE-ResNet + Value 32)...")
    new_model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    new_state_dict = new_model.state_dict()

    transferred_count = 0
    skipped_layers = []

    # Filtrage et copie des poids
    for key, old_tensor in old_state_dict.items():
        if key in new_state_dict:
            new_tensor = new_state_dict[key]
            # Vérification stricte des dimensions
            if old_tensor.shape == new_tensor.shape:
                new_state_dict[key] = old_tensor
                transferred_count += 1
            else:
                skipped_layers.append(
                    f"{key} (Dimension différente: {old_tensor.shape} vs {new_tensor.shape})")
        else:
            skipped_layers.append(f"{key} (N'existe plus dans le nouveau modèle)")

    # Application du nouveau dictionnaire hybride au modèle
    new_model.load_state_dict(new_state_dict)

    # Création du nouveau checkpoint (On remet iteration et step à 0)
    # Note : On ne transfère PAS l'optimizer, Adam crasherait à cause des nouveaux poids.
    new_checkpoint = {
        'model_state_dict': new_model.state_dict(),
        'iteration': 0,
        'global_step': 0
    }

    torch.save(new_checkpoint, new_ckpt_path)

    print("\n" + "=" * 40)
    print("           BILAN DU TRANSFERT")
    print("=" * 40)
    print(f"Tenseurs copiés avec succès : {transferred_count}")
    print(f"Tenseurs laissés aléatoires : {len(new_state_dict) - transferred_count}")
    print("-" * 40)
    for skip in skipped_layers:
        print(f"Ignoré : {skip}")
    print("=" * 40)
    print(f"\nNouveau checkpoint prêt : {new_ckpt_path}")


if __name__ == "__main__":
    # Mets ici le chemin vers ton dernier bon checkpoint
    from core.data_paths import CHECKPOINTS_DIR
    OLD_CHECKPOINT = CHECKPOINTS_DIR / "2026_04_04_12h38_iter431_unsupervised.pt"

    # Le fichier qui servira de point de départ pour ton Apprentissage Supervisé
    NEW_CHECKPOINT = CHECKPOINTS_DIR / "modele_generation0_init.pt"

    transfer_weights(OLD_CHECKPOINT, NEW_CHECKPOINT)
