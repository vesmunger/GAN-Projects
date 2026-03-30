import pytorch_lightning as pl
from src.datamodule import MNISTDataModule
from src.lightning_module import GAN

def train():
    # 1. Initialize Data and Model
    dm = MNISTDataModule()
    model = GAN(latent_dim=100)

    # 2. Setup Trainer (Professional Auto-detection)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # 3. Fit
    trainer.fit(model, dm)

if __name__ == "__main__":
    import torch
    train()
