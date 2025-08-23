# ESRGAN Super-Resolution Project

This project implements a version of the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) model using PyTorch to perform image super-resolution. The model is trained on the DIV2K dataset to upscale low-resolution images by a factor of 4x, aiming to produce high-resolution images with enhanced details and perceptual quality.

## Project Structure

The notebook is organized into several sections:

1.  **Setup and Data Preparation**: Downloading and organizing the DIV2K dataset.
2.  **Model Architecture**: Defining the Generator and Discriminator networks based on the ESRGAN architecture.
3.  **Training**: Setting up the training loop, loss functions, and optimizers, and training the GAN.
4.  **Inference**: Implementing a function to use the trained model to perform super-resolution on new images.
5.  **Evaluation**: Calculating performance metrics like PSNR, SSIM, and FID to assess the model's quality.

## Implementation Details

### 1. Setup and Data Preparation

This section handles the initial setup, including mounting Google Drive (if using Colab) to access files like the Kaggle API key, installing necessary libraries, and downloading and organizing the dataset.

-   **Google Drive Mounting and Kaggle Setup**:
    -   Mounts your Google Drive to access files.
    -   Configures the Kaggle API by copying your `kaggle.json` file to the correct directory and setting permissions. This allows downloading datasets directly from Kaggle.
    -   Installs the `kaggle` library.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    # Setup Kaggle API
    !mkdir -p ~/.kaggle/
    !cp "/content/drive/MyDrive/kaggle/kaggle.json" ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    !python -m pip install -qq kaggle

    print("✅ Google Drive mounted and Kaggle API configured!")
    ```

-   **Dataset Download and Extraction**:
    -   Downloads the DIV2K dataset from Kaggle using the configured API.
    -   Extracts the downloaded zip file.

    ```python
    # Download DIV2K dataset from Kaggle
    !kaggle datasets download -d joe1995/div2k-dataset

    # Extract the dataset
    !unzip -qq "div2k-dataset.zip"

    # Check the extracted structure (optional)
    import os
    print("📁 Dataset structure:")
    for dirname, _, filenames in os.walk('/content'):
        if 'DIV2K' in dirname or 'div2k' in dirname:
            print(f"📂 {dirname}")
            if filenames:
                print(f"   Files: {len(filenames)} items")
    ```

-   **Organizing Dataset**:
    -   Creates a structured project directory (`/content/ESRGAN_Project`) with subdirectories for the dataset (train/hr, train/lr, valid/hr, valid/lr), models, and results.
    -   Searches the extracted files to find the actual High-Resolution (HR) and Low-Resolution (LR) image folders.
    -   Copies the HR images from the training set to the `dataset/train/hr` and `dataset/valid/hr` directories (first 700 for training, remaining for validation).
    -   Attempts to find and copy the corresponding LR images (specifically the x4 downsampled versions). If LR images are not found, the dataset class is designed to generate them by downsampling the HR images during training.

    ```python
    import os
    import shutil

    # Create organized dataset structure
    project_path = '/content/ESRGAN_Project'
    os.makedirs(f'{project_path}/dataset/train/hr', exist_ok=True)
    os.makedirs(f'{project_path}/dataset/train/lr', exist_ok=True)
    os.makedirs(f'{project_path}/dataset/valid/hr', exist_ok=True)
    os.makedirs(f'{project_path}/dataset/valid/lr', exist_ok=True)
    os.makedirs(f'{project_path}/models', exist_ok=True)
    os.makedirs(f'{project_path}/results', exist_ok=True)

    # ... (code for finding and copying images) ...

    print("\n📁 Final dataset structure:")
    print(f"Training HR: {len(os.listdir(f'{project_path}/dataset/train/hr'))} images")
    print(f"Training LR: {len(os.listdir(f'{project_path}/dataset/train/lr'))} images")
    print(f"Validation HR: {len(os.listdir(f'{project_path}/dataset/valid/hr'))} images")
    print(f"Validation LR: {len(os.listdir(f'{project_path}/dataset/valid/lr'))} images")
    ```

-   **Library Imports**:
    -   Installs core libraries like PyTorch, torchvision, OpenCV, Pillow, and Matplotlib.
    -   Imports necessary modules for building and training the neural networks, handling images, and plotting.

    ```python
    !pip install -q torch torchvision torchaudio
    !pip install -q opencv-python pillow matplotlib

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from torchvision.utils import save_image
    import torch.nn.functional as F
    import random
    from pathlib import Path

    print("✅ All libraries imported successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    ```

-   **Model Save/Load Functions**:
    -   Defines utility functions to save the complete training checkpoint (generator, discriminator, optimizers, epoch, loss) and to save just the generator model for inference.
    -   Includes functions to load a checkpoint to resume training and to load only the generator for inference.

    ```python
    # Model save/load configuration
    MODEL_SAVE_PATH = f"{project_path}/models/esrgan_generator.pth"
    CHECKPOINT_PATH = f"{project_path}/models/esrgan_checkpoint.pth"

    def save_model(generator, discriminator, optimizer_gen, optimizer_disc, epoch, loss):
        # ... saves model and checkpoint ...

    def load_model_for_inference(generator, device):
        # ... loads generator state dict ...

    def load_checkpoint(generator, discriminator, optimizer_gen, optimizer_disc, device):
        # ... loads checkpoint for resuming training ...

    def check_trained_model_exists():
        # ... checks if generator model exists ...

    print("✅ Save/Load functions defined!")
    ```

### 2. Model Architecture

This section defines the building blocks and the full Generator and Discriminator networks based on the ESRGAN architecture.

-   **Residual Dense Block (RDB)**:
    -   Implements the core building block of the ESRGAN generator.
    -   Consists of multiple convolutional layers within a dense connection structure, followed by a LeakyReLU activation.
    -   Uses residual connections to facilitate training of very deep networks.

    ```python
    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            # ... defines convolutional layers and LeakyReLU ...

        def forward(self, x):
            # ... implements forward pass with dense connections and residual addition ...
            return x5 * 0.2 + x # Residual scaling
    ```

-   **Residual-in-Residual Dense Block (RRDB)**:
    -   Combines multiple RDBs (typically 3) within another residual connection.
    -   This stacking of residual blocks is a key feature of ESRGAN, contributing to its performance.

    ```python
    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            # ... initializes RDBs ...

        def forward(self, x):
            # ... passes input through RDBs and adds residual connection ...
            return out * 0.2 + x # Residual scaling
    ```

-   **Generator**:
    -   Constructs the full generator network.
    -   Starts with an initial convolutional layer.
    -   Includes a "trunk" of multiple RRDB blocks.
    -   Followed by another convolutional layer.
    -   Uses upsampling layers (using `F.interpolate`) and convolutional layers to increase spatial resolution.
    -   Ends with a final convolutional layer and a `tanh` activation to output an image in the range [-1, 1].

    ```python
    class Generator(nn.Module):
        def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
            # ... defines layers: conv1, trunk (RRDBs), conv2, upconv1, upconv2, conv3, lrelu ...

        def forward(self, x):
            # ... implements forward pass with residual and upsampling connections ...
            return torch.tanh(out) # Output in [-1, 1]
    ```

-   **Discriminator**:
    -   Constructs the discriminator network, which is a standard convolutional network used in GANs to distinguish between real and fake images.
    -   Consists of a series of convolutional layers with increasing filter sizes and stride 2 for downsampling.
    -   Uses BatchNorm and LeakyReLU activations.
    -   The final layer outputs a single value representing the probability of the input image being real.

    ```python
    class Discriminator(nn.Module):
        def __init__(self, in_channels=3):
            # ... defines discriminator blocks with conv, BatchNorm, LeakyReLU ...
            # ... final conv layer outputs a single value ...

        def forward(self, img):
            return self.model(img)
    ```

### 3. Dataset Class

Defines a custom PyTorch Dataset class to handle loading and preprocessing the DIV2K images for training.

-   **`DIV2KDataset` Class**:
    -   Takes the paths to the HR and LR image folders as input.
    -   Handles loading HR images and finding/generating corresponding LR images. If an LR image doesn't exist (e.g., if you only downloaded HR), it automatically creates one by downsampling the HR image using bicubic interpolation.
    -   Implements random cropping for data augmentation during training, ensuring corresponding crops are taken from both HR and LR images.
    -   Applies a transformation (like converting to tensor and normalizing) to the images.

    ```python
    class DIV2KDataset(Dataset):
        def __init__(self, hr_folder, lr_folder, transform=None, crop_size=128):
            # ... initializes paths, lists images, sets transform and crop size ...

        def __len__(self):
            return len(self.hr_images)

        def __getitem__(self, idx):
            # ... loads HR and LR images (generating LR if needed) ...
            # ... performs random cropping ...
            # ... applies transform ...
            return lr_image, hr_image
    ```

### 4. Training Setup and Loop

This section sets up the training environment, initializes models, loss functions, and optimizers, and runs the training loop.

-   **Device Configuration**:
    -   Sets the device to GPU (`cuda`) if available, otherwise uses CPU.

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    ```

-   **Dataset Check and Skip Training Option**:
    -   Checks if training images are available.
    -   Checks if a trained model already exists and gives the user the option to skip training and use the existing model.

    ```python
    # Check dataset
    hr_count = len([f for f in os.listdir(train_hr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # ... checks hr_count ...

    # Check if model already exists and prompt user
    if check_trained_model_exists():
        choice = input("🔍 Trained model found! ... ")
        skip_training = True # or False based on user input
    else:
        skip_training = False
    ```

-   **Model Initialization**:
    -   Initializes the Generator and Discriminator models and moves them to the selected device.

    ```python
    generator = Generator(in_channels=3, num_channels=64, num_blocks=23).to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    ```

-   **Loss Functions and Optimizers**:
    -   Defines the loss functions: `BCEWithLogitsLoss` for the GAN adversarial loss and `L1Loss` (Mean Absolute Error) for the pixel-wise content loss.
    -   Initializes Adam optimizers for both the generator and discriminator with specified learning rates and beta values.

    ```python
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()

    optimizer_gen = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    ```

-   **Training Loop**:
    -   This is the main part of the training process, executed only if `skip_training` is False and training images are available.
    -   Sets training parameters like the number of epochs, batch size, and crop size.
    -   Defines image transformations (ToTensor and normalization).
    -   Creates the `DIV2KDataset` instance and a `DataLoader` to efficiently load data in batches.
    -   Iterates through epochs:
        -   Sets models to training mode (`.train()`).
        -   Iterates through batches from the DataLoader.
        -   **Trains the Generator**:
            -   Zeros generator gradients.
            -   Passes LR images through the generator to get fake HR images (`gen_hr`).
            -   Calculates the adversarial loss (`loss_GAN`) by comparing the discriminator's output on `gen_hr` to a tensor of ones (indicating real images, as the generator wants to fool the discriminator). It uses the detached real image predictions for relativistic GAN loss.
            -   Calculates the pixel-wise loss (`loss_pixel`) between `gen_hr` and the real HR images (`hr_imgs`).
            -   Combines the GAN and pixel-wise losses to get the total generator loss (`loss_gen`). A weight (e.g., 100) is typically applied to the pixel loss in ESRGAN variants.
            -   Performs backpropagation and updates generator weights.
        -   **Trains the Discriminator**:
            -   Zeros discriminator gradients.
            -   Passes real HR images and detached fake HR images (`gen_hr.detach()`) through the discriminator.
            -   Calculates the loss for real images (`loss_real`) and fake images (`loss_fake`) using `BCEWithLogitsLoss` and relativistic GAN principles.
            -   Combines `loss_real` and `loss_fake` to get the total discriminator loss (`loss_disc`).
            -   Performs backpropagation and updates discriminator weights.
        -   Prints batch and epoch progress, including generator and discriminator losses.
        -   Saves a training checkpoint every few epochs.
    -   Saves the final trained model after all epochs are completed.

    ```python
    if not skip_training and hr_count > 0:
        # ... sets parameters, defines transforms, creates dataset and dataloader ...

        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()
            # ... initialization of epoch losses and batch count ...

            for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
                # ... move data to device ...

                # Train Generator
                # ... generator training steps ...

                # Train Discriminator
                # ... discriminator training steps ...

                # ... update epoch losses and batch count ...
                # ... print batch progress ...

            # ... print epoch progress ...
            # ... save checkpoint ...

        # ... save final model ...
    # ... else branches for skipping or no dataset ...
    ```

### 5. Inference

Provides functions to use the trained generator model to perform super-resolution on individual images and visualize the results.

-   **`inference_single_image` Function**:
    -   Takes an image path and an optional output path as input.
    -   Initializes the generator model and loads the trained weights using `load_model_for_inference`.
    -   Loads and preprocesses the input image (converting to RGB, resizing to dimensions divisible by 4, applying the same normalization as training).
    -   Passes the preprocessed image tensor through the generator in evaluation mode (`torch.no_grad()`).
    -   Converts the output tensor back to a PIL Image (denormalizing and clamping values).
    -   Saves the resulting super-resolution image if an output path is provided.
    -   Returns the super-resolution PIL image and the input LR PIL image.

    ```python
    def inference_single_image(image_path, output_path=None):
        # ... loads generator, loads image, preprocesses ...
        # ... performs forward pass with torch.no_grad() ...
        # ... converts tensor to PIL Image, saves if needed ...
        return hr_img, img # Returns generated SR and input LR PIL images
    ```

-   **`display_comparison` Function**:
    -   Takes an input low-resolution image (PIL) and the generated super-resolution image (PIL) as input.
    -   Uses Matplotlib to display both images side-by-side for visual comparison.

    ```python
    def display_comparison(lr_image, hr_image):
        # ... uses Matplotlib to create subplots and display images ...
    ```

-   **Testing with Validation Image**:
    -   Loads a sample HR image from the validation set.
    -   Creates a low-resolution version of this image for testing.
    -   Calls `inference_single_image` to generate a super-resolution image from the test LR image.
    -   If successful, it calls `display_comparison` to show the input LR and generated SR.
    -   Also displays the original HR image alongside the input LR and generated SR for a three-way comparison.

    ```python
    # Test the trained model
    print("🧪 Testing the trained model...")
    # ... loads validation images, selects one, creates LR version ...
    # ... calls inference_single_image ...
    if result_hr_pil:
        print("✅ Super-resolution successful!")
        display_comparison(input_lr_pil, result_hr_pil)
        # ... displays three-way comparison (LR, SR, original HR) ...
    # ... else branches for failure or no model/images ...
    ```

-   **Upload and Test Function**:
    -   Provides a function (`upload_and_test`) that allows you to upload your own image (if running in Colab), perform super-resolution on it, display the comparison, and download the result.

    ```python
    from google.colab import files

    def upload_and_test():
        """Upload your own image and test super-resolution"""
        print("📤 Upload an image to test super-resolution:")
        uploaded = files.upload()
        # ... iterates through uploaded files ...
        # ... calls inference_single_image ...
        # ... displays comparison, downloads result ...
    ```

### 6. Evaluation Metrics

This section defines functions to calculate quantitative metrics (PSNR, SSIM, LPIPS, and FID) to evaluate the performance of the trained model.

-   **Metric Functions (PSNR, SSIM, LPIPS)**:
    -   Defines functions `calculate_psnr`, `calculate_ssim`, and `calculate_lpips` using libraries like `torch`, `numpy`, `skimage`, and `lpips`.
    -   These functions take image inputs (either PIL Images or Tensors depending on the function) and return the calculated metric score.
    -   Includes necessary preprocessing steps within the functions (like converting between image types, normalizing, handling device placement).

    ```python
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from torchvision.transforms import ToTensor
    import math
    from skimage.metrics import structural_similarity as ssim
    import lpips

    def calculate_psnr(img1, img2):
        # ... PSNR calculation ...
        pass

    def calculate_ssim(img1, img2):
        # ... SSIM calculation ...
        pass

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

    def calculate_lpips(img1, img2):
        # ... LPIPS calculation using the initialized model ...
        pass
    ```

-   **Metric Calculation and FID**:
    -   This block generates super-resolution images for the entire validation set and calculates PSNR, SSIM, and FID.
    -   It iterates through the validation HR images. For each, it gets/generates the LR version, generates the SR image using the trained model, and saves the generated SR image to a temporary directory (`fid_generated_sr`).
    -   While doing this, it calculates the PSNR and SSIM for each generated SR/original HR pair and stores them.
    -   After generating all SR images and calculating per-image metrics, it calculates the average PSNR and SSIM and prints them.
    -   Finally, it uses the `cleanfid` library to calculate the FID score between the directory of original validation HR images and the directory of generated SR images.
    -   The FID score is printed.

    ```python
    from cleanfid import fid
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    import math
    import torch.nn.functional as F

    # Define paths and create directory for generated images
    valid_hr_path = f'{project_path}/dataset/valid/hr'
    valid_lr_path = f'{project_path}/dataset/valid/lr'
    generated_sr_path = f'{project_path}/results/fid_generated_sr'
    os.makedirs(generated_sr_path, exist_ok=True)

    # ... Device configuration and model loading ...

    # Get list of validation HR images
    valid_hr_images = [f for f in os.listdir(valid_hr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize metric lists
    psnr_scores = []
    ssim_scores = []

    # Generate SR images and calculate per-image metrics
    with torch.no_grad():
        for img_name in valid_hr_images:
            # ... load/generate LR, load HR, generate SR, save SR ...
            # ... calculate PSNR and SSIM for the pair ...
            psnr_score = calculate_psnr(...)
            ssim_score = calculate_ssim(...)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)

    # Calculate and print average PSNR and SSIM
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    print("\n📊 Average Performance Metrics:")
    print(f"  Average PSNR: {avg_psnr:.4f}")
    print(f"  Average SSIM: {avg_ssim:.4f}")

    # Calculate and print FID
    print("\nCalculating FID...")
    fid_score = fid.compute_fid(valid_hr_path, generated_sr_path, mode="clean", device=str(device))
    print(f"  FID Score: {fid_score:.4f}")

    # Optional: Clean up generated images directory
    # import shutil
    # shutil.rmtree(generated_sr_path)
    ```

## How to Use

1.  **Setup**: Run the initial cells to mount Google Drive, configure Kaggle, download, and organize the dataset, and install libraries.
2.  **Model Definition**: Run the cells defining the `ResidualDenseBlock`, `RRDB`, `Generator`, and `Discriminator` classes.
3.  **Dataset Class**: Run the cell defining the `DIV2KDataset` class.
4.  **Training**: Run the cell to set up the training and the cell containing the training loop. You can choose to train from scratch or load an existing model if available.
5.  **Inference (Optional)**: Run the cells defining the inference and display functions. You can then use the testing code or the `upload_and_test()` function to apply the model to images.
6.  **Evaluation (Optional)**: Run the cells defining the metric calculation functions and the cell to calculate and display PSNR, SSIM, and FID on the validation set.
