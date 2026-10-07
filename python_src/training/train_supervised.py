
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import argparse
from pathlib import Path
import torch
from core.data_paths import CHECKPOINTS_DIR
from torch.nn import functional as F

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from core.model import ChessNet
# from pretraining.sharded_dataset import ShardedChessDataset
from pretraining.dataset import ChessSupervisedDataset

torch.set_float32_matmul_precision('medium')


class AlphaZeroLightning(L.LightningModule):
    def __init__(self, learning_rate=1e-3, num_res_blocks=10, num_filters=128):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_policy, y_value = batch

        # Forward pass
        p_logits, v_pred = self(x)

        # 1. Perte de la Policy (Stratégie)
        policy_loss = F.cross_entropy(p_logits, y_policy)

        # 2. Perte de la Value (Évaluation)
        value_loss = F.mse_loss(v_pred, y_value)

        # 3. Précision Top-1
        # On extrait l'index du coup ayant la plus forte probabilité (dim=1)
        preds = torch.argmax(p_logits, dim=1)
        # On calcule la moyenne des prédictions correctes
        acc = (preds == y_policy).float().mean()

        # Perte totale
        loss = policy_loss + value_loss

        # Logging dynamique vers WandB
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/policy_loss", policy_loss)
        self.log("train/value_loss", value_loss)
        # Logging de la précision dans la barre de progression
        self.log("train/policy_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # L'optimiseur Adam est très robuste pour ce type de phase supervisée
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Entrainement supervise depuis des PGN.")
    parser.add_argument("--pgn-dir", type=Path,
                        default=_runtime.REPO_ROOT / "training_data" / "clean_pgns")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint Lightning a reprendre (sinon modele neuf).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not args.pgn_dir.is_dir():
        raise SystemExit(f"Dossier PGN absent : {args.pgn_dir}")
    # --- Configuration ---
    PGN_DIR = args.pgn_dir
    BATCH_SIZE = 2048

    # --- Initialisation de WandB ---
    wandb_logger = WandbLogger(
        project="alphazero-chess", name="supervised_phase_3_gm",
        log_model=False
    )

    # --- Préparation des Données ---
    # Instanciation de la nouvelle classe
    dataset = ChessSupervisedDataset(PGN_DIR)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINTS_DIR),
        filename="alphazero-supervised-{step}",
        every_n_train_steps=3000,
        save_top_k=-1,
        save_last=True
    )

    # --- Instanciation ---
    # 1. On instancie le wrapper Lightning avec l'architecture vierge
    model = AlphaZeroLightning(
        learning_rate=1e-3,
        num_res_blocks=10,
        num_filters=128
    )

    # --- Entraînement ---
    trainer = L.Trainer(
        max_epochs=4,
        # limit_train_batches=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        log_every_n_steps=50
    )

    trainer.fit(
        model,
        train_dataloaders=dataloader,
        ckpt_path=str(args.resume) if args.resume is not None else None
    )
