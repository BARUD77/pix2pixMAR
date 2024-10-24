# Pix2Pix MAR (Metal Artifact Reduction)
## Project Description
This project uses a Pix2Pix Generative Adversarial Network (GAN) for Metal Artifact Reduction (MAR) in CT images. The goal is to improve the quality of CT images affected by metal artifacts using deep learning techniques.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following installed on your system:
- Python 3.12
- PyTorch
- CUDA (if using GPU)
- Git

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/pix2pixMAR.git
    ```

2. Navigate to the project directory:
    ```bash
    cd pix2pixMAR
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. If using a GPU, ensure CUDA is installed properly:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
