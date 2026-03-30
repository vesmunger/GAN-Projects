# MNIST-GAN-Lightning

A professional, modular implementation of a Generative Adversarial Network (GAN) built with PyTorch Lightning. This project demonstrates stable GAN training using manual optimization, deep convolutional architectures, and automated data pipelines.

---

## 📌 Project Overview

This repository implements a DCGAN-inspired architecture to generate handwritten digits from the MNIST dataset.

* **Generator**: Transforms latent noise into (28 \times 28) grayscale images using transposed convolutions
* **Discriminator**: Acts as a binary classifier, distinguishing real images from generated samples

The project emphasizes training stability and modular design for experimentation and scalability.

---

## 🚀 Key Features

* **Manual Optimization**
  Full control over the adversarial training loop between Generator and Discriminator

* **Stable Architectures**
  Uses Batch Normalization and LeakyReLU to improve convergence and reduce mode collapse

* **Mixed Precision Training**
  Supports 16-bit mixed precision for faster and more efficient GPU training

* **Automated Logging**
  Integrated with TensorBoard for real-time tracking of:

  * Generator loss (`g_loss`)
  * Discriminator loss (`d_loss`)

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-gan-lightning.git
cd mnist-gan-lightning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Run the training script:

```bash
python train.py
```

---

## 🏗️ Modular Architecture & Scalability

This project follows the **Open-Closed Principle**, allowing easy extension without modifying core logic.

### 🔄 Swap Dataset

Create a new data module (e.g., `Cifar10DataModule`) in:

```
src/datamodule.py
```

Then pass it into the trainer in `train.py`.

---

### 🧠 Extend Models

You can introduce improved architectures such as:

* Self-Attention GAN
* Residual GAN

Add them in:

```
src/models.py
```

As long as input/output shapes remain consistent, no changes are required in the training loop.

---

### ⚙️ Hardware Agnostic

Using:

```python
accelerator="auto"
```

The project runs seamlessly on:

* CPU
* CUDA (NVIDIA GPUs)
* MPS (Apple Silicon)

---

## 📊 Results

After ~20 epochs, the Generator produces clear and diverse handwritten digits from random noise.

To visualize training progress:

```bash
tensorboard --logdir lightning_logs
```

This will display:

* Loss curves
* Training metrics
* Model performance over time

---

## 🎯 Use Cases

* Learning GAN architectures and adversarial training
* Experimenting with generative models
* Prototyping deep learning pipelines with PyTorch Lightning

---

## 🔮 Future Improvements

* Add support for multiple datasets (CIFAR-10, custom datasets)
* Implement advanced GAN variants (WGAN, StyleGAN)
* Integrate evaluation metrics (FID, IS score)

---

## 📄 License

MIT

