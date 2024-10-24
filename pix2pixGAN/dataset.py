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