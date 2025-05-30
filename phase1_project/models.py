# models.py - Neural network models for RxRx1 batch effect correction
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import logging
from torch.autograd import Function
from typing import Dict, Any, Tuple

from config import MODEL_NAME, PRETRAINED, NUM_CHANNELS

logger = logging.getLogger(__name__)

class GradientReversalLayer(Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def gradient_reversal(x, alpha=1.0):
    """Functional interface for gradient reversal."""
    return GradientReversalLayer.apply(x, alpha)

class FeatureExtractor(nn.Module):
    """
    Feature extractor for 6-channel RxRx1 fluorescence images.
    Learns batch-invariant representations when used with adversarial training.
    """

    def __init__(self, backbone_name: str = MODEL_NAME, pretrained: bool = PRETRAINED):
        super().__init__()

        if backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            # Modify first conv layer for 6 channels
            self.backbone.conv1 = nn.Conv2d(
                NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            if pretrained:
                # Initialize new conv layer with pretrained weights (average across channels)
                with torch.no_grad():
                    pretrained_weight = models.resnet50(pretrained=True).conv1.weight
                    new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, NUM_CHANNELS, 1, 1)
                    self.backbone.conv1.weight = nn.Parameter(new_weight)
                    logger.info("Initialized 6-channel conv1 with averaged pretrained weights")

            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            # Modify first conv layer for 6 channels
            self.backbone.conv1 = nn.Conv2d(
                NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            if pretrained:
                # Initialize new conv layer with pretrained weights
                with torch.no_grad():
                    pretrained_weight = models.resnet18(pretrained=True).conv1.weight
                    new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, NUM_CHANNELS, 1, 1)
                    self.backbone.conv1.weight = nn.Parameter(new_weight)

            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            # Try loading from timm for other architectures
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    in_chans=NUM_CHANNELS,
                    num_classes=0  # Remove classification head
                )
                self.feature_dim = self.backbone.num_features
                logger.info(f"Loaded {backbone_name} from timm with {NUM_CHANNELS} input channels")
            except Exception as e:
                logger.error(f"Failed to load {backbone_name}: {e}")
                raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        """Extract features from 6-channel input."""
        return self.backbone(x)

class PerturbationClassifier(nn.Module):
    """
    Classifier for genetic perturbation prediction (biological signal).
    This represents the "main task" we want to preserve.
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)

class BatchDiscriminator(nn.Module):
    """
    Domain discriminator that predicts experimental plate (batch).
    Acts as adversary to create plate-invariant features.
    """

    def __init__(self, feature_dim: int, num_plates: int, dropout: float = 0.5):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_plates)
        )

    def forward(self, features, alpha=1.0):
        """Apply gradient reversal before classification."""
        reversed_features = gradient_reversal(features, alpha)
        return self.discriminator(reversed_features)

class DANNModel(nn.Module):
    """
    Domain Adversarial Neural Network for batch effect correction in RxRx1.

    Architecture components:
    1. Feature Extractor: Maps 6-channel images to batch-invariant features
    2. Perturbation Classifier: Predicts genetic perturbation (siRNA) classes
    3. Batch Discriminator: Predicts experimental plate with gradient reversal

    The adversarial training forces the feature extractor to learn representations
    that are useful for perturbation classification but uninformative about batch.
    """

    def __init__(self,
                 num_perturbations: int,
                 num_plates: int,
                 backbone_name: str = MODEL_NAME,
                 pretrained: bool = PRETRAINED):
        super().__init__()

        self.feature_extractor = FeatureExtractor(backbone_name, pretrained)
        self.perturbation_classifier = PerturbationClassifier(
            self.feature_extractor.feature_dim, num_perturbations
        )
        self.batch_discriminator = BatchDiscriminator(
            self.feature_extractor.feature_dim, num_plates
        )

        self.num_perturbations = num_perturbations
        self.num_plates = num_plates

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DANN.

        Args:
            x: Input 6-channel images (B, 6, H, W)
            alpha: Gradient reversal strength

        Returns:
            Dictionary with perturbation and batch predictions
        """
        # Extract features
        features = self.feature_extractor(x)

        # Predict genetic perturbations (main biological task)
        perturbation_predictions = self.perturbation_classifier(features)

        # Predict experimental plates (adversarial task)
        batch_predictions = self.batch_discriminator(features, alpha)

        return {
            'perturbation_predictions': perturbation_predictions,
            'batch_predictions': batch_predictions,
            'features': features
        }

class BaselineModel(nn.Module):
    """
    Baseline model without adversarial training.
    Used for comparison to show batch effects.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_name: str = MODEL_NAME,
                 pretrained: bool = PRETRAINED):
        super().__init__()

        self.feature_extractor = FeatureExtractor(backbone_name, pretrained)
        self.classifier = PerturbationClassifier(
            self.feature_extractor.feature_dim, num_classes
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through baseline model."""
        features = self.feature_extractor(x)
        predictions = self.classifier(features)
        return predictions

def get_model(model_type: str,
              num_classes: int,
              num_plates: int = None,
              backbone_name: str = MODEL_NAME,
              pretrained: bool = PRETRAINED) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "dann" or "baseline"
        num_classes: Number of genetic perturbation classes
        num_plates: Number of experimental plates (required for DANN)
        backbone_name: Feature extractor backbone
        pretrained: Use pretrained weights

    Returns:
        Initialized model
    """
    if model_type.lower() == "dann":
        if num_plates is None:
            raise ValueError("num_plates is required for DANN model")
        model = DANNModel(num_classes, num_plates, backbone_name, pretrained)
        logger.info(f"Created DANN model: {num_classes} classes, {num_plates} plates")
    elif model_type.lower() == "baseline":
        model = BaselineModel(num_classes, backbone_name, pretrained)
        logger.info(f"Created baseline model: {num_classes} classes")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
