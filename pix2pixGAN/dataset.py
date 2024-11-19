import os
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # Import PIL to handle image conversion


class ArtifactDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load the shuffled directory names from the txt file
        with open(txt_file, 'r') as f:
            self.image_dirs = f.readlines()
        
        # Process the paths to remove 'gt.h5' and keep only the directory paths
        self.image_dirs = [os.path.dirname(path.strip()) for path in self.image_dirs]

        # Create a list to hold all .h5 file paths
        self.h5_files = []
        
        # Iterate through each directory listed in the txt file
        for dir_path in self.image_dirs:
            folder_path = os.path.join(self.root_dir, dir_path)

            # Add all .h5 files in the directory, ignoring 'gt.h5'
            h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5') and f != 'gt.h5']
            self.h5_files.extend(h5_files)

    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_file_path = self.h5_files[idx]
        
        # Load the artifact (ma_CT) and ground truth (image) from the same .h5 file
        with h5py.File(h5_file_path, 'r') as h5_file:
            artifact_image = h5_file['ma_CT'][:]  # NumPy array
            gt_image = h5_file['image'][:]        # NumPy array
        
        # Convert NumPy arrays to PIL Images before applying transformations
        artifact_image = Image.fromarray(artifact_image)
        gt_image = Image.fromarray(gt_image)
        
        # Apply any transformations (e.g., Resize, ToTensor, Normalize, etc.)
        if self.transform:
            artifact_image = self.transform(artifact_image)
            gt_image = self.transform(gt_image)
        
        return artifact_image, gt_image





class MetalArtifactDataset(Dataset):
    def __init__(self, metal_dir, gt_dir, mode="down", augment=False):
        """
        Args:
            metal_dir (str): Path to the directory with artifact-affected images (metal).
            gt_dir (str): Path to the directory with ground truth images (GT).
            mode (str): Mode of image preprocessing - 'up' for padding to 512x512, 'down' for resizing to 256x256.
            augment (bool): If True, apply data augmentation.
        """
        self.metal_dir = metal_dir
        self.gt_dir = gt_dir
        self.mode = mode.lower()  # Convert to lowercase for consistency
        self.augment = augment

        # Get list of file names in the metal directory
        self.image_filenames = [f for f in os.listdir(metal_dir) if f.endswith('.png')]

        # Define transformations
        if self.mode == "up":
            # Padding transformation to 512x512
            self.transform = transforms.Compose([
                transforms.Pad(padding=(74, 74)),  # Pad (512 - 364) // 2 = 74 on each side
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        elif self.mode == "down":
            # Resize transformation to 256x256
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            raise ValueError("Invalid mode. Use 'up' for padding or 'down' for resizing.")

        # Define augmentations if enabled
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),  # Random rotation within 15 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Adjust brightness and contrast
            ])
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]

        # Load the metal artifact image and ground truth image
        metal_image_path = os.path.join(self.metal_dir, img_name)
        gt_image_path = os.path.join(self.gt_dir, img_name)

        # Open images
        metal_image = Image.open(metal_image_path).convert("L")  # Assuming grayscale images
        gt_image = Image.open(gt_image_path).convert("L")

        # Apply augmentations if enabled
        if self.augmentation:
            # Ensure consistent augmentation for paired images
            seed = torch.Generator().manual_seed(idx)
            torch.manual_seed(seed.initial_seed())
            metal_image = self.augmentation(metal_image)
            gt_image = self.augmentation(gt_image)

        # Apply transformations (either padding or resizing based on mode)
        metal_image = self.transform(metal_image)
        gt_image = self.transform(gt_image)

        return metal_image, gt_image