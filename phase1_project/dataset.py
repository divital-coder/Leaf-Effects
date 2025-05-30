# dataset.py - RxRx1 Cellular Image Dataset for Batch Effect Analysis
import os
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from typing import Tuple, Optional, Dict, List, Any, Callable  # Add Callable here
import logging

from config import (
    RXRX1_DATASET_ROOT, METADATA_CSV_PATH, NUM_CHANNELS, CHANNEL_NAMES,
    IMAGE_SIZE_RGB, ORIGINAL_IMAGE_SIZE, DEFAULT_STAGE_MAP, DEV_MODE
)

logger = logging.getLogger(__name__)

class RxRx1Dataset(Dataset):
    """
    Dataset class for RxRx1 cellular images focusing on batch effect correction.

    The RxRx1 dataset contains 6-channel fluorescence microscopy images of cells
    treated with various siRNA (genetic perturbations). Each image belongs to a
    specific experimental plate which represents a potential batch effect source.

    Structure expected:
    - images/experiment/plate/well_site_channel.png
    - metadata.csv with columns: experiment, plate, well, site, cell_type, sirna_id, etc.
    """

    def __init__(self,
                 root_dir: str = RXRX1_DATASET_ROOT,
                 metadata_path: str = METADATA_CSV_PATH,
                 transform_rgb: Optional[Callable] = None,
                 split: str = "train",
                 experiment: str = None,
                 class_to_idx: Optional[Dict[str, int]] = None,
                 classes: Optional[List[str]] = None):
        """
        Initialize RxRx1Dataset.

        Args:
            root_dir: Path to images directory
            metadata_path: Path to metadata.csv
            transform_rgb: Transform to apply to 6-channel images
            split: Dataset split ("train", "test")
            experiment: Experiment name (e.g., "HUVEC-01")
            class_to_idx: Optional pre-computed class mapping
            classes: Optional pre-computed class list
        """
        self.root_dir = Path(root_dir)
        self.metadata_path = Path(metadata_path)  # Ensure Path object
        self.transform_rgb = transform_rgb
        self.split = split
        self.experiment = experiment

        # Load and filter metadata
        self.metadata = self._load_metadata()

        if len(self.metadata) == 0:
            logger.warning("No data found for the specified criteria")
            # Initialize empty attributes to prevent AttributeError
            self.classes = []
            self.class_to_idx = {}
            self.sirna_id_to_class_idx = {}
            self.plates = []
            self.plate_to_idx = {}
            self.num_classes = 0
            self.num_plates = 0
            return

        # Set up class mappings
        if classes is not None and class_to_idx is not None:
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            self._setup_class_mappings()

        # Always create sirna_id_to_class_idx mapping
        self._setup_sirna_mappings()

        # Set up plate mappings
        self._setup_plate_mappings()

        # Final dataset info
        self.num_classes = len(self.classes)
        self.num_plates = len(self.plates)

        logger.info(f"RxRx1Dataset initialized:")
        logger.info(f"  Split: {self.split}, Experiment: {self.experiment}")
        logger.info(f"  Samples: {len(self.metadata)}")
        logger.info(f"  Genetic perturbations (classes): {self.num_classes}")
        logger.info(f"  Experimental plates: {self.num_plates}")

    def _setup_sirna_mappings(self):
        """Create mapping from sirna_id to class index."""
        self.sirna_id_to_class_idx = {}

        for idx, row in self.metadata.iterrows():
            sirna_id = row['sirna_id']
            class_name = f"siRNA_{sirna_id}"

            if class_name in self.class_to_idx:
                self.sirna_id_to_class_idx[sirna_id] = self.class_to_idx[class_name]
            else:
                # Fallback: assign to class 0 or create new class
                logger.warning(f"siRNA_id {sirna_id} not found in classes, assigning to class 0")
                self.sirna_id_to_class_idx[sirna_id] = 0

    def _setup_class_mappings(self):
        """Create class mappings from genetic perturbations."""
        if 'sirna_id' in self.metadata.columns:
            unique_sirnas = sorted(self.metadata['sirna_id'].unique())
            logger.info(f"Using sirna_id as classes: {len(unique_sirnas)} genetic perturbations")

            self.classes = [f"siRNA_{sid}" for sid in unique_sirnas]
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        else:
            logger.warning("No sirna_id column found. Using dummy classes.")
            self.classes = ["dummy_class"]
            self.class_to_idx = {"dummy_class": 0}

    def _setup_plate_mappings(self):
        """Create plate mappings for batch identification."""
        if 'plate' in self.metadata.columns:
            unique_plates = sorted(self.metadata['plate'].unique())
            self.plates = [f"plate_{plate}" for plate in unique_plates]
            self.plate_to_idx = {plate_name: idx for idx, plate_name in enumerate(self.plates)}
        else:
            logger.warning("No plate column found. Using dummy plates.")
            self.plates = ["dummy_plate"]
            self.plate_to_idx = {"dummy_plate": 0}

   # In the __getitem__ method, after loading the image but before transforms:

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        if idx >= len(self.metadata):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.metadata)}")

        try:
            row = self.metadata.iloc[idx]

            # Load 6-channel image
            rgb_image = self._load_image_channels(
                plate=row['plate'],
                well=row['well'],
                site=row['site']
            )

            # Apply transforms (these should work with 6-channel now)
            if self.transform_rgb:
                rgb_image = self.transform_rgb(rgb_image)

            # Manual normalization for 6 channels (simple approach)
            # Normalize each channel to have mean=0, std=1
            if rgb_image.dim() == 3:  # (C, H, W)
                for c in range(rgb_image.shape[0]):
                    channel = rgb_image[c]
                    if channel.std() > 1e-6:  # Avoid division by zero
                        rgb_image[c] = (channel - channel.mean()) / channel.std()

            # Get labels
            sirna_id = row['sirna_id']
            plate_id = row['plate']

            # Use the mapping we created in __init__
            class_label = self.sirna_id_to_class_idx.get(sirna_id, 0)
            plate_label = self.plate_to_idx.get(f"plate_{plate_id}", 0)

            return {
                'rgb_image': rgb_image,
                'label': torch.tensor(class_label, dtype=torch.long),
                'plate_id': torch.tensor(plate_label, dtype=torch.long),
                'id': torch.tensor(idx, dtype=torch.long),
                'metadata': {
                    'experiment': row.get('experiment', 'unknown'),
                    'plate': plate_id,
                    'well': row['well'],
                    'site': row['site'],
                    'sirna_id': sirna_id,
                    'cell_type': row.get('cell_type', 'unknown')
                }
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to prevent crashes
            dummy_image = torch.zeros((NUM_CHANNELS, IMAGE_SIZE_RGB[0], IMAGE_SIZE_RGB[1]))
            return {
                'rgb_image': dummy_image,
                'label': torch.tensor(0, dtype=torch.long),
                'plate_id': torch.tensor(0, dtype=torch.long),
                'id': torch.tensor(idx, dtype=torch.long),
                'metadata': {'error': str(e)}
            }

    def _load_metadata(self) -> pd.DataFrame:
        """Load and filter metadata for the specified split and experiment."""
        try:
            if not self.metadata_path.exists():
                logger.warning(f"Metadata file not found at {self.metadata_path}. Creating dummy data.")
                return self._create_dummy_metadata()

            df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded metadata with {len(df)} total entries")

            # Filter by experiment if specified and exists
            if self.experiment and 'experiment' in df.columns:
                exp_values = df['experiment'].unique()
                if self.experiment in exp_values:
                    df = df[df['experiment'] == self.experiment]
                    logger.info(f"Filtered to {len(df)} entries for experiment '{self.experiment}'")
                else:
                    logger.warning(f"Experiment '{self.experiment}' not found. Using first available: {exp_values[0]}")
                    self.experiment = exp_values[0]
                    df = df[df['experiment'] == self.experiment]

            # Filter by dataset split (train/test in RxRx1)
            if 'dataset' in df.columns:
                if self.split == "train":
                    df = df[df['dataset'] == 'train']
                elif self.split == "test":
                    df = df[df['dataset'] == 'test']
                logger.info(f"Filtered to {len(df)} entries for dataset split '{self.split}'")

            # Filter to only plates that exist in your images folder
            available_plates = self._get_available_plates()
            if available_plates:
                df = df[df['plate'].isin(available_plates)]
                logger.info(f"Filtered to {len(df)} entries for available plates: {available_plates}")

            # Development mode: limit data for faster testing
            if DEV_MODE and len(df) > 1000:
                logger.info(f"DEV_MODE: Limiting dataset to 1000 samples for faster testing")
                df = df.sample(n=1000, random_state=42).reset_index(drop=True)

            # Ensure required columns exist
            required_cols = ['plate', 'well', 'site', 'sirna_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                return self._create_dummy_metadata()

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return self._create_dummy_metadata()

    def _get_available_plates(self) -> List[int]:
        """Get list of plates that actually exist in the images folder."""
        available_plates = []
        try:
            # Check what's directly in images folder (Plate3, Plate4)
            if self.root_dir.exists():
                for item in self.root_dir.iterdir():
                    if item.is_dir() and item.name.startswith('Plate'):
                        try:
                            plate_num = int(item.name.replace('Plate', ''))
                            available_plates.append(plate_num)
                        except ValueError:
                            continue

            # Also check experiment subdirectories
            if self.experiment:
                exp_dir = self.root_dir / self.experiment
                if exp_dir.exists():
                    for item in exp_dir.iterdir():
                        if item.is_dir() and item.name.startswith('Plate'):
                            try:
                                plate_num = int(item.name.replace('Plate', ''))
                                available_plates.append(plate_num)
                            except ValueError:
                                continue

            available_plates = sorted(list(set(available_plates)))
            logger.info(f"Found plates in filesystem: {available_plates}")

        except Exception as e:
            logger.warning(f"Error scanning for available plates: {e}")

        return available_plates

    def _create_dummy_metadata(self) -> pd.DataFrame:
        """Create dummy metadata for testing when real metadata is unavailable."""
        logger.info("Creating dummy RxRx1 metadata for testing")

        dummy_data = []
        for plate in range(1, 6):  # 5 dummy plates
            for well_row in ['A', 'B']:
                for well_col in ['01', '02']:
                    well = f"{well_row}{well_col}"
                    for site in [1, 2]:
                        # Create dummy genetic perturbation classes
                        sirna_id = (plate + site) % 10  # Integer IDs as in real RxRx1
                        cell_type = f"CellType_{plate % 3}"

                        dummy_data.append({
                            'experiment': self.experiment,
                            'plate': plate,
                            'well': well,
                            'site': site,
                            'cell_type': cell_type,
                            'sirna_id': sirna_id,  # Integer, not string
                            'dataset': self.split
                        })

        df = pd.DataFrame(dummy_data)
        logger.info(f"Created dummy metadata with {len(df)} entries")
        return df

    def _load_image_channels(self, plate: int, well: str, site: int) -> torch.Tensor:
        """Load all 6 fluorescence channels for a given image location."""
        channels = []

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            # Updated paths based on your actual structure
            possible_paths = [
                # Structure 1: images/images/experiment/Plate*/well_ssite_channel.png
                self.root_dir / self.experiment / f"Plate{plate}" / f"{well}_s{site}_{channel_name}.png",
                # Structure 2: images/images/experiment/well_ssite_channel.png (flat structure)
                self.root_dir / self.experiment / f"{well}_s{site}_{channel_name}.png",
                # Structure 3: Different plate naming
                self.root_dir / self.experiment / f"plate{plate}" / f"{well}_s{site}_{channel_name}.png",
                # Structure 4: Different file extensions
                self.root_dir / self.experiment / f"Plate{plate}" / f"{well}_s{site}_{channel_name}.tiff",
                self.root_dir / self.experiment / f"Plate{plate}" / f"{well}_s{site}_{channel_name}.jpg",
                # Structure 5: Different site naming
                self.root_dir / self.experiment / f"Plate{plate}" / f"{well}_site{site}_{channel_name}.png",
            ]

            img_array = None
            found_path = None

            for img_path in possible_paths:
                try:
                    if img_path.exists():
                        # Load as grayscale (single channel)
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        found_path = img_path
                        break
                except Exception as e:
                    logger.debug(f"Failed to load {img_path}: {e}")
                    continue

            if img_array is None:
                # Log the first few attempted paths for debugging
                logger.warning(f"No image found for Plate{plate}/{well}_s{site}_{channel_name}")
                logger.debug(f"Tried paths: {[str(p) for p in possible_paths[:3]]}")
                # Create dummy channel data
                img_array = np.random.rand(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]).astype(np.float32)
            else:
                # Log successful path (only for first channel to avoid spam)
                if channel_idx == 0:
                    pass
                    # logger.info(f"✅ Successfully found images at: {found_path.parent}")

            channels.append(img_array)

        # Stack channels to create 6-channel image: (C, H, W)
        image_tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
        return image_tensor

    def __len__(self) -> int:
        return len(self.metadata)
