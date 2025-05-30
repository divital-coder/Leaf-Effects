# main.py - RxRx1 Data Testing and Visualization
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from dataset import RxRx1Dataset
from data_utils import get_transforms, rxrx1_collate_fn
from config import (
    RXRX1_DATASET_ROOT, METADATA_CSV_PATH, IMAGE_SIZE_RGB, NUM_CHANNELS,
    BATCH_SIZE, NUM_WORKERS, VISUALIZATION_DIR, CHANNEL_NAMES
)

logger = logging.getLogger(__name__)

def plot_rxrx1_sample(image_tensor: torch.Tensor,
                      label: int,
                      plate_id: int,
                      metadata: Dict[str, Any],
                      sample_id: int,
                      class_names: List[str],
                      filename_prefix: str = "rxrx1_sample"):
    """Plot a 6-channel RxRx1 fluorescence image sample."""

    # Create subplot for 6 channels
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    class_name = class_names[label] if 0 <= label < len(class_names) else f"Unknown_{label}"

    fig.suptitle(f"RxRx1 Sample - ID: {sample_id}\n"
                f"siRNA: {class_name} | Plate: {metadata.get('plate', 'unknown')} | "
                f"Well: {metadata.get('well', 'unknown')} | Site: {metadata.get('site', 'unknown')}",
                fontsize=14)

    # Plot each channel
    for i, channel_name in enumerate(CHANNEL_NAMES):
        if i < image_tensor.shape[0]:
            channel_img = image_tensor[i].cpu().numpy()

            # Handle normalization - assume data is already normalized
            if channel_img.min() < 0:  # If normalized, denormalize for visualization
                channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min())

            im = axes[i].imshow(channel_img, cmap='gray')
            axes[i].set_title(f'Channel {channel_name}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()

    safe_filename = f"{filename_prefix}_id{sample_id}_sirna{label}_plate{plate_id}.png"
    save_path = os.path.join(VISUALIZATION_DIR, safe_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved RxRx1 sample visualization to {save_path}")

def test_rxrx1_dataset_loading(dataset_root: str, metadata_path: str, experiment: str = None):
    """Test RxRx1 dataset loading and visualization."""
    logger.info(f"Testing RxRx1 dataset loading from: {dataset_root}")
    logger.info(f"Metadata: {metadata_path}")

    # First, let's check what experiments are available
    metadata = pd.read_csv(metadata_path)
    available_experiments = metadata['experiment'].unique()
    logger.info(f"Available experiments ({len(available_experiments)}): {available_experiments[:10]}...")

    # Use a valid experiment if none specified or invalid one given
    if experiment is None or experiment not in available_experiments:
        # Use HUVEC-01 instead of HUVEC-1
        experiment = "HUVEC-01"
        logger.info(f"Using experiment: {experiment}")

    logger.info(f"Experiment: {experiment}")

    # Use validation transforms for consistent visualization
    transforms = get_transforms(train=False)

    dataset = RxRx1Dataset(
        root_dir=dataset_root,
        metadata_path=metadata_path,
        transform_rgb=transforms,
        split="train",
        experiment=experiment
    )

    if len(dataset) == 0:
        logger.warning(f"Dataset is empty. Check paths and metadata.")
        # Debug: Check what's available for this experiment
        exp_metadata = metadata[metadata['experiment'] == experiment]
        logger.info(f"Found {len(exp_metadata)} total entries for experiment {experiment}")
        train_entries = exp_metadata[exp_metadata['dataset'] == 'train']
        logger.info(f"Found {len(train_entries)} train entries for experiment {experiment}")
        if len(train_entries) > 0:
            logger.info(f"Sample train entry: {train_entries.iloc[0].to_dict()}")
        return

    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Genetic perturbations: {dataset.num_classes}")
    logger.info(f"  Experimental plates: {dataset.num_plates}")
    logger.info(f"  Classes: {dataset.classes[:10]}...")  # Show first 10

    # Create dataloader for batch testing
    dataloader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, 4, len(dataset)),
        shuffle=True,
        num_workers=0,  # Easier debugging
        collate_fn=rxrx1_collate_fn
    )

    try:
        batch_data = next(iter(dataloader))

        logger.info(f"Sample batch:")
        logger.info(f"  RGB tensor shape: {batch_data['rgb_image'].shape}")
        logger.info(f"  Labels shape: {batch_data['label'].shape}")
        logger.info(f"  Plate IDs shape: {batch_data['plate_id'].shape}")
        logger.info(f"  Labels (first few): {batch_data['label'][:4].tolist()}")
        logger.info(f"  Plate IDs (first few): {batch_data['plate_id'][:4].tolist()}")

        # Plot a few samples
        for i in range(min(3, batch_data['rgb_image'].shape[0])):
            # Get metadata for this sample
            sample_idx = batch_data['id'][i].item()
            sample_metadata = dataset.metadata.iloc[sample_idx].to_dict()

            plot_rxrx1_sample(
                image_tensor=batch_data['rgb_image'][i],
                label=batch_data['label'][i].item(),
                plate_id=batch_data['plate_id'][i].item(),
                metadata=sample_metadata,
                sample_id=sample_idx,
                class_names=dataset.classes,
                filename_prefix=f"rxrx1_{experiment}_sample_{i}"
            )

        logger.info(f"Successfully tested and visualized RxRx1 samples.")

    except StopIteration:
        logger.error(f"DataLoader is empty despite dataset having {len(dataset)} items.")
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)

def run_rxrx1_visualizations():
    """Run RxRx1 dataset testing and visualization."""
    # Ensure VISUALIZATION_DIR exists
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    logger.info("Starting RxRx1 dataset testing...")
    logger.info(f"Expected dataset paths:")
    logger.info(f"  Dataset root: {RXRX1_DATASET_ROOT}")
    logger.info(f"  Metadata path: {METADATA_CSV_PATH}")

    # Check if paths exist
    if not Path(RXRX1_DATASET_ROOT).exists():
        logger.error(f"RXRX1_DATASET_ROOT ('{RXRX1_DATASET_ROOT}') does not exist.")
        logger.info("Please verify your RxRx1 dataset is downloaded to the correct location.")
        logger.info(f"Expected structure:")
        logger.info(f"  {RXRX1_DATASET_ROOT}/")
        logger.info(f"    ├── HUVEC-01/")
        logger.info(f"    │   ├── Plate1/")
        logger.info(f"    │   │   ├── A01_s1_w1.png")
        logger.info(f"    │   │   ├── A01_s1_w2.png")
        logger.info(f"    │   │   └── ...")
        logger.info(f"    └── ...")
        return

    if not Path(METADATA_CSV_PATH).exists():
        logger.error(f"METADATA_CSV_PATH ('{METADATA_CSV_PATH}') does not exist.")
        logger.info("Please verify metadata.csv is in the correct location.")
        return

    # Test dataset loading with corrected experiment name
    test_rxrx1_dataset_loading(RXRX1_DATASET_ROOT, METADATA_CSV_PATH)

    logger.info("Finished RxRx1 dataset testing.")

if __name__ == "__main__":
    # Setup basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_rxrx1_visualizations()
