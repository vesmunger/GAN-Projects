import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from .models import Generator, Discriminator

class GAN(pl.LightningModule):
    def __init__(self, latent_dim: int = 100, lr: float = 0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False # Manual optimization for GANs

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()

        # Train Generator
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        g_loss = torch.nn.functional.binary_cross_entropy(y_hat, torch.ones_like(y_hat))

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Train Discriminator
        y_hat_real = self.discriminator(real_imgs)
        real_loss = torch.nn.functional.binary_cross_entropy(y_hat_real, torch.ones_like(y_hat_real))

        y_hat_fake = self.discriminator(fake_imgs.detach())
        fake_loss = torch.nn.functional.binary_cross_entropy(y_hat_fake, torch.zeros_like(y_hat_fake))
        
        d_loss = (real_loss + fake_loss) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]

    def on_train_epoch_end(self):
        # Professional way to visualize progress every epoch
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu().detach()
        grid = plt.figure(figsize=(8, 3))
        for i in range(sample_imgs.size(0)):
            plt.subplot(1, 6, i+1)
            plt.imshow(sample_imgs[i, 0, :, :], cmap='gray')
            plt.axis('off')
        plt.show()
        plt.close()
