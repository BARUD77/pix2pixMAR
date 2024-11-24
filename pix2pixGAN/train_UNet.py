# imports
import argparse
import os
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from progress.bar import IncrementalBar

from generator import UnetGenerator
from criterion import GeneratorLoss
from utils import Logger, initialize_weights
from dataset import MetalArtifactDataset
from metrics import calculate_psnr, calculate_ssim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser for capturing input arguments
parser = argparse.ArgumentParser(description="Pix2Pix GAN for Metal Artifact Reduction")

# # Adding arguments for dataset paths
parser.add_argument('--gt_dir', type=str, required=True, help="path to the ground truth images.")
parser.add_argument('--metal_dir', type=str, required=True, help="path to the metal artifact affected images.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation.")
parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")

args = parser.parse_args()

# Use the passed arguments
gt_dir = args.gt_dir
metal_dir = args.metal_dir
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr

dataset = MetalArtifactDataset(metal_dir=metal_dir, gt_dir=gt_dir, augment=False)

# Define the split sizes
total_images = len(dataset)
validation_split = 0.15
val_size = int(total_images * validation_split)
train_size = total_images - val_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Model
print('Defining model!')
generator = UnetGenerator().to(device)

# Optimizer for the generator
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
g_criterion = UNetLoss()
logger = Logger(filename='Metal_Artifacts_New')

# Training loop
for epoch in range(epochs):
    ge_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(train_loader))
    for batch_idx, (x, real) in enumerate(train_loader):
        # Each batch contains artifact-affected images (x) and their corresponding clean images (real)
        x = x.to(device)
        real = real.to(device)

        # Generator's loss
        fake = generator(x)
        g_loss = g_criterion(fake, real)

        # Update the generator's parameters
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Add batch loss
        ge_loss += g_loss.item()
        bar.next()
    bar.finish()

    # Obtain per-epoch generator loss
    g_loss = ge_loss / len(train_loader)

    # Validation phase
    generator.eval()  # Set the model to evaluation mode
    ssim_score = 0.0
    psnr_score = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (val_x, val_real) in enumerate(val_loader):
            val_x, val_real = val_x.to(device), val_real.to(device)
            val_fake = generator(val_x)
            
            # Calculate SSIM and PSNR for each image in the batch
            for i in range(val_x.size(0)):
                ssim_score += calculate_ssim(val_fake[i], val_real[i])
                psnr_score += calculate_psnr(val_fake[i], val_real[i])
            num_batches += 1

    avg_ssim = ssim_score / (num_batches * batch_size)
    avg_psnr = psnr_score / (num_batches * batch_size)
    
    # Log losses and metrics
    logger.add_scalar('generator_loss', g_loss, epoch + 1)
    logger.add_scalar('SSIM', avg_ssim, epoch + 1)
    logger.add_scalar('PSNR', avg_psnr, epoch + 1)
    logger.save_weights(generator.state_dict(), 'generator')
    
    print(f"[Epoch {epoch+1}/{epochs}] [G loss: {g_loss:.3f}] [SSIM: {avg_ssim:.3f}] [PSNR: {avg_psnr:.3f}] ETA: {time.time() - start:.3f}s")
    
    # Return to training mode
    generator.train()

logger.close()
print('End of training process!')
