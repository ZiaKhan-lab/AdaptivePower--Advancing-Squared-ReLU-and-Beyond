# AdaptivePower: Advancing Squared‑ReLU and Beyond

This repository contains the full implementation and benchmarks for novel activation functions—including Parametric, Soft, and Exponential Squared‑ReLU variants, the Adaptive Power Unit (APU)—and standard baselines. It provides theoretical analyses (differentiability, Lipschitz estimates), empirical evaluations on MNIST, CIFAR‑10, and EMNIST, across multiple architectures (SimpleCNN, ResNet, Transformer, DCGAN), and generates plots and GAN samples.
 
## Setup & Requirements
### 1. Clone the repo:
  ```
  git clone https://github.com/ZiaKhan-lab/AdaptivePower--Advancing-Squared-ReLU-and-Beyond.git
  cd AdaptivePower--Advancing-Squared-ReLU-and-Beyond
  ```
### 2. Install dependencies (tested on Python 3.8+):
```
pip install torch torchvision matplotlib tqdm numpy
```
### 3. Launch the notebook:
```
jupyter notebook activation_benchmarks.ipynb
```
### 4. Enable GPU
* **Kaggle:** Settings → Accelerator → GPU
* **Colab:** Runtime → Change runtime type → GPU

## Quick Usage
Inside the notebook, you can:
* Define which activations to test:
```
act_names = ['ReLU','LeakyReLU','PReLU','SoftReLU','ESReLU','APU']
```
* Run classifier benchmarks:
```
run_activation_benchmark('SimpleCNN','MNIST', act_names, epochs=5)
run_activation_benchmark('ResNet','CIFAR10', act_names, epochs=5)
run_activation_benchmark('Transformer','EMNIST', act_names, epochs=5)
```
* Run GAN benchmarks:
```
run_gan_benchmark('MNIST', act_names, epochs=5)
```
* Perform theoretical analyses:
```
analyze_all_activations()
run_gradient_flow_analysis(['MNIST','CIFAR10'], ['SimpleCNN','ResNet'], act_names)
```
All plots are saved under results/, and example GAN outputs under gan_samples/.
## Key Plots & Outputs
### 1. Activation Functions & Derivatives
``
results/activation_function_comparison.png
``
### 2. Training Curves
* SimpleCNN on MNIST: ``results/SimpleCNN_MNIST_comparison.png``
* ResNet on CIFAR‑10: ``results/ResNet_CIFAR10_comparison.png``
* Transformer on EMNIST:``results/Transformer_EMNIST_comparison.png``
### 3. GAN Performance
``results/GAN_MNIST_comparison.png``
### 4. Gradient Flow Analysis
``results/gradient_flow/``




