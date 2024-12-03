#imports
import argparse
import os
import h5py
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from progress.bar import IncrementalBar

from pix2pixGAN.generator import UnetGenerator
from pix2pixGAN.discriminator import ConditionalDiscriminator
from pix2pixGAN.criterion import GeneratorLoss, DiscriminatorLoss
from pix2pixGAN.utils import Logger, initialize_weights
from pix2pixGAN.dataset import MetalArtifactDataset
from pix2pixGAN.metrics import calculate_psnr, calculate_ssim

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Argument parser for capturing input arguments
    parser = argparse.ArgumentParser(description="Pix2Pix GAN for Metal Artifact Reduction")

    # # Adding arguments for dataset paths
    parser.add_argument('--path', type=str, required=True, help="path to the training dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--checkpoints', action='store_true', help="Enable saving checkpoints during training.")

    args = parser.parse_args()

    # Use the passed arguments
    path=args.path
    gt_dir = os.path.join(path, "GT") #args.gt_dir
    metal_dir = os.path.join(path, "Metal") #args.metal_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    # Prepare directories for Experiment
    experiments_path = "Experiments"
    os.makedirs(experiments_path, exist_ok=True)
    experiment_id = sum(os.path.isdir(os.path.join(experiments_path, d)) for d in os.listdir(experiments_path)) + 1
    experiment_dir_name = f"Experiment_{experiment_id}_epochs_{epochs}"
    experiment_dir_path = os.path.join(experiments_path, experiment_dir_name)
    if args.checkpoints:
        os.makedirs(os.path.join(experiment_dir_path, "checkpoints"), exist_ok=True)
    else:
        os.makedirs(experiment_dir_path, exist_ok=True)

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

    # models
    print('Defining models!')
    generator = UnetGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)

    #optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # loss functions
    g_criterion = GeneratorLoss(alpha=100)
    d_criterion = DiscriminatorLoss()
    logger = Logger(exp_name=experiment_dir_path, filename='logs')

    best_ssim = float('-inf')

    # training loop
    for epoch in range(epochs):
        ge_loss = 0.
        de_loss = 0.
        start = time.time()
        bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(train_loader))
        for batch_idx, (x, real) in enumerate(train_loader):
            # Each batch contains artifact-affected images (x)  and their corresponding clean images (real)
            x=x.to(device)
            real = real.to(device)

            # generator's loss
            fake = generator(x)
            fake_pred = discriminator(fake, x)
            g_loss = g_criterion(fake, real, fake_pred)
            
            # discriminator's loss 
            fake = generator(x).detach()
            fake_pred = discriminator(fake, x)
            real_pred = discriminator(real, x)
            d_loss = d_criterion(fake_pred, real_pred)
            
            # Generator`s params update
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Discriminator`s params update
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # add batch losses
            ge_loss += g_loss.item()
            de_loss += d_loss.item()
            bar.next()
        bar.finish()
        # obtain per epoch losses
        g_loss = ge_loss/len(train_loader)
        d_loss = de_loss/len(train_loader)

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
        logger.add_scalar('discriminator_loss', d_loss, epoch + 1)
        logger.add_scalar('SSIM', avg_ssim, epoch + 1)
        logger.add_scalar('PSNR', avg_psnr, epoch + 1)

        print(f"[Epoch {epoch+1}/{epochs}] [G loss: {g_loss:.3f}] [D loss: {d_loss:.3f}] [SSIM: {avg_ssim:.3f}] [PSNR: {avg_psnr:.3f}] ETA: {time.time() - start:.3f}s")
        
        #update the weights if ssim improved
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            logger.save_weights(generator.state_dict(), 'generator')
            logger.save_weights(discriminator.state_dict(), 'discriminator')
            print(f"Best SSIM improved to {best_ssim:.3f}. Saved generator weights.")

        # save checkpoints to showcase inference progression
        if(args.checkpoints and epoch % 10 == 0):
            logger.save_weights(generator.state_dict(), f'checkpoints/generator_{epoch}')
            
        # Return to training mode
        generator.train()

    logger.close()
    print('End of training process!')
