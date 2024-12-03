import os
import argparse
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from pix2pixGAN.generator import UnetGenerator 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def restoreImage(ma_image, filename):
    ma_tensor = transform(ma_image).unsqueeze(0).to(device)

    # Generate the cleaned image
    with torch.no_grad():
        generated_tensor = generator(ma_tensor)
    
    # Remove batch dimension and apply inverse transform for visualization
    generated_image = inverse_transform(generated_tensor.squeeze(0).cpu())
    generated_image = generated_image.convert("L")

    # Save the generated image
    generated_image.save(filename)
    print(f"Saved Image: {filename}")

    return generated_image


if __name__ == "__main__":
    # Argument parser for capturing input arguments
    parser = argparse.ArgumentParser(description="Pix2Pix GAN for Metal Artifact Reduction")

    # # Adding arguments for dataset paths
    parser.add_argument('--path', type=str, required=True, help="path to the corrupted image or directory of images")
    parser.add_argument('--weights', type=str, required=True, help="path to the generator weights")
    parser.add_argument('--output_dir', type=str, default='outputs', help="path to the output directory")

    args = parser.parse_args()

    # Check if the path exists
    if not os.path.exists(args.path):
        print(f"Error: The path '{args.path}' does not exist.")
        sys.exit(1)  # Exit with a non-zero status to indicate an error

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained generator model
    generator = UnetGenerator().to(device)
    generator.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    generator.eval()  # Set model to evaluation mode

    # Transformation to match training preprocessing (e.g., normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Inverse transform for visualization (to undo normalization)
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),  # Reverse normalization if mean=0.5, std=0.5
        transforms.ToPILImage()
    ])

    # Check if the path is a directory
    if os.path.isdir(args.path):
        print(f"Processing the directory '{args.path}'...")

        # If it's a directory, get a list of all image files
        image_files = [
            file for file in os.listdir(args.path)
            if os.path.isfile(os.path.join(args.path, file)) and file.lower().endswith(".png")
        ]
        
        metal_images = [Image.open(path).convert("L") for path in image_files]

        for i, ma_image in enumerate(metal_images):
            name = os.path.splitext(os.path.basename(image_files[i]))[0]
            restoreImage(ma_image, f"{args.output_dir}/{name}_restored.png")

        print(f"Completed restoration of '{len(metal_images)}' images...")

    else:
        # If it's not a directory, implement different logic
        print(f"Restoring the image '{args.path}'...")
        ma_image = Image.open(args.path).convert("L")
        name = os.path.splitext(os.path.basename(args.path))[0]
        generated_image = restoreImage(ma_image, f"{args.output_dir}/{name}_restored.png")
        
        # Display the original and restored images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ma_image, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(generated_image, cmap="gray")
        axes[1].set_title("Restored Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

