#imports

import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from progress.bar import IncrementalBar

from generator import UnetGenerator
from discriminator import ConditionalDiscriminator
from criterion import GeneratorLoss, DiscriminatorLoss
from utils import Logger, initialize_weights
from dataset import ArtifactDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# define transform
transform = transforms.Compose([
    #transforms.Pad((48, 48, 48, 48)),  # Add 24 pixels of padding to match 512x512 size
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images (optional)
])

# Path to your .txt file and root directory
txt_file = '/home/maya/train_640geo_dir.txt'  # Replace with actual path to your .txt file 
root_dir = '/home/maya/train_640reduced/'  # Replace with actual root directory

# Create the dataset
dataset = ArtifactDataset(txt_file=txt_file, root_dir=root_dir, transform=transform)

# Create the DataLoader with shuffling
batch_size = 32  # Adjust based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

#optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()
logger = Logger(filename='Metal_Artifacts')
epochs=5


# training loop
for epoch in range(epochs):
    ge_loss = 0.
    de_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(dataloader))
    for batch_idx, (x, real) in enumerate(dataloader):
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
    g_loss = ge_loss/len(dataloader)
    d_loss = de_loss/len(dataloader)
    # count timeframe
    end = time.time()
    tm = (end - start)
    logger.add_scalar('generator_loss', g_loss, epoch+1)
    logger.add_scalar('discriminator_loss', d_loss, epoch+1)
    logger.save_weights(generator.state_dict(), 'generator')
    logger.save_weights(discriminator.state_dict(), 'discriminator')
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, epochs, g_loss, d_loss, tm))
logger.close()
print('End of training process!')