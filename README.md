# Pix2Pix MAR (Metal Artifact Reduction)

This repo contains our implementation of Pix2PixMAR, a model for Metal Artifact Reduction in CT images.

![results](images/200_2.png)
![results](images/200_3.png)

Metal artifacts in computed tomography (CT) images, caused by metallic implants or objects, pose significant challenges in accurate diagnosis by introducing severe streaks and distortions. Conventional methods for metal artifact reduction (MAR), such as linear interpolation or iterative reconstruction, often fail to completely eliminate these artifacts without compromising image quality. We propose a **Pix2Pix Generative Adversarial Network (GAN)** with **U-Net** generator model. Our model achieves an **SSIM of 0.8514** and a **PSNR of 29.0140**.

![results](images/Architecture.png)

## Installation

1. Install [Python](https://www.python.org/downloads/). This repo is running on python **3.9.18**. It's better to install and use Anaconda to manage your environment.
2. Install PIP, make sure it's the latest pip (only using python3) **(if you are not going the anaconda route)**

   ```bash
   python3 --version
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3 get-pip.py
   python3 -m pip install --upgrade pip
   ```

3. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) for GPU accelaration. If you do not have a nvidia GPU you can skip this and run the project with your CPU.
4. Install pytorch from their [site](https://pytorch.org/) and select the os and cude version you're running on. Example:

   `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`

   If you are not running CUDA:

   `pip3 install torch torchvision torchaudio`
5. Set up a jupyter kernel to run the .ipynb notebooks.

   ```bash
   pip install jupyter
   python -m ipykernel install --user --name [kernel name]
   ```

6. Clone this repo, pip Install the requirements file

   `pip install -r requirements.txt`

## Repo

This repo contains the following files:

- `Images`: Image files for this README file

- `pix2pixGAN`: Source files for the pix2pix GAN implementation
  - `criterion.py`: Classes for the loss functions
  - `dataset.py`: Dataset classes
  - `discriminator.py`: Discriminator model class
  - `generator.py`: U-Net generator model class
  - `metrics.py`: SSIM and PSNR calculation functions
  - `util.py`: A helper class to save data through training

- `exploring.ipynb`: A complete walkthrough guide notebook on running inference of the model
- `restore.py`: Inference script
- `train.py`: Training script

## Getting Started

The best way to get started with our model is to go throgh the `exploring.ipynb` notebook. It walks through the setup, inference, and evaluation.

To run the notebook you will first need to download the dataset and an Experimnet folder with the pretrained weights. You can find them here:

- [Dataset Download](https://github.com/LangruiZhou/HISMAR)
- [Weights Download](https://drive.google.com/file/d/1sk44Os_6Uwk59GD1bdXvsy-Sc_WRHPK8/view?usp=sharing)

Place the Dataset folder from the github link under the Data folder. You should end up with this structure:

```
Data
   L Dataset
      L test
         L GT
         L LI
         L Metal
      L train
         L GT
         L LI
         L Metal
```

Place the weights folder under the Experiments folder and unzip it. You should end up with this structure:

```
Experiments
    L Experiment_002_augmentation_False_epochs_200
        L generator_002.pt
        L November_14_2024_04_32AM_Metal_Artifacts_New.json
```

## Training

To train the model you will first need to download the dataset as explaing in the previous [section](#getting-started). Then simply run `train.py`.

Here's the list of arguments:

1. `--path`: path to the training dataset
2. `--batch_size` *(Optional)*: Batch size for training and validation.
3. `--epochs` *(Optional)*: Number of training epochs.
4. `--lr` *(Optional)*: Learning rate for the optimizer.
5. `--checkpoints` *(Optional)*: Enable saving checkpoints during training. Every 10 epochs the model state is saved.

Here's a sample command:

`python train.py --path Data/Dataset/train --epochs 100`

## Inference

To run inference simply run:

`python restore.py --path [path] --weights [path]`

The script takes 2 arguments:

1. `--path`: path to a corrupted image or a directory of corrupted images (.png extension)
2. `--weights`: path to the generator weights

## Acknowledgments

- Thanks to [pix2pix](https://github.com/akanametov/pix2pix) for their code that contributed to this project.
