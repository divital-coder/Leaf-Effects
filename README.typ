#import "@preview/frame-it:1.2.0": *

#let text-color = black
#let background-color = white
#if sys.inputs.at("theme", default: "light") == "dark" {
  text-color = rgb(240, 246, 252)
  background-color = rgb("#0d1117")
}

#set text(text-color)
#set page(fill: background-color, height: auto, margin: 4mm)

// Define frames
#let (feature, example, variant, syntax) = frames(
  feature: ("Feature",),
  example: ("Example", gray),
  syntax: ("Syntax",),
  variant: ("Variant", orange),
)

#show: frame-style(styles.boxy)

// Frame 1: Patch Embedding
#variant[Patch Embedding][Initial tokenization][
The `PatchEmbed` class is responsible for breaking down the input image into fixed-size patches. This is the first step in the Vision Transformer pipeline, converting raw pixels into a sequence of embeddings.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = img_size; self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # B, C_embed, H_grid, W_grid
        x = x.flatten(2) # B, C_embed, N_patches_flat
        x = x.transpose(1, 2) # B, N_patches_flat, C_embed
        return self.norm(x)
```
It uses a 2D convolution with kernel size and stride equal to the patch size to achieve this efficiently.
]

#pagebreak()

// Frame 2: Hierarchical Stages
#variant[Hierarchical Stages][Progressive processing][
The `HVTStage` manages a sequence of Transformer blocks. It supports progressive downsampling, allowing the model to build a hierarchical representation of features, similar to a CNN.

```python
class HVTStage(nn.Module):
    def __init__(self, dim: int, current_input_resolution_patches: Tuple[int, int], depth: int, ...):
        super().__init__()
        # ... (initialization)
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for i in range(depth)])
        self.downsample_layer = None
        if downsample_class is not None:
            self.downsample_layer = downsample_class(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x
```
Each stage consists of multiple transformer blocks followed by an optional downsampling layer.
]

#pagebreak()

// Frame 3: Feature Pyramid
#variant[Feature Pyramid][Multi-scale output][
The `forward_features_encoded` method extracts features from both the RGB and optional spectral streams. It returns the encoded features along with the original patch grid dimensions, facilitating multi-scale analysis.

```python
def forward_features_encoded(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None):
    # ... (setup)
    x_rgb_encoded, rgb_orig_patch_grid = self._forward_stream(
        rgb_img, self.rgb_patch_embed, self.rgb_pos_embed, self.pos_drop_rgb,
        self.rgb_stages, self.norm_rgb_final_encoder
    )
    # ... (spectral stream handling)
    return x_rgb_encoded, x_spectral_encoded, rgb_orig_patch_grid, spectral_orig_patch_grid
```
This dual-stream approach allows the model to leverage both visual and spectral information.
]

#pagebreak()

// Frame 4: Efficiency Optimizations
#variant[Efficiency Optimizations][Patch Merging][
`PatchMerging` reduces the spatial resolution of the feature map by a factor of 2 while increasing the channel dimension. This is crucial for creating the hierarchical structure and maintaining computational efficiency.

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution_patches: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (reshaping and selecting 2x2 patches)
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)
```
It concatenates features from 2x2 neighboring patches and projects them to a new dimension.
]

#pagebreak()

// Frame 5: Feature Analysis Result
#variant[Feature Analysis][t-SNE Visualization][
#image("assets/tsne_feature_space_comparison.png", width: 100%)
The t-SNE plot demonstrates the clear separation of semantic classes in the feature space, indicating that the model has learned discriminative representations.
]

#pagebreak()

// Frame 6: Training Convergence Result
#variant[Training Convergence][Loss & Accuracy][
#image("assets/convergence_plot.png", width: 100%)
The convergence plots show the training and validation loss/accuracy over epochs. The model demonstrates stable training and good generalization.
]

#pagebreak()

// Frame 7: Transfer Learning Result
#variant[Transfer Learning][Confusion Matrix][
#image("assets/confusion_matrix.png", width: 100%)
The confusion matrix highlights the model's performance on downstream classification tasks, showing high accuracy across most classes.
]

#pagebreak()

// Frame 8: Ablation Studies Result
#variant[Ablation Studies][Architectural Impact][
#image("assets/convergence_plot_detailed_ablations.png", width: 100%)
This plot compares the performance of different architectural configurations, justifying the design choices made in the final model.
]

#pagebreak()

// Frame 9: Attention Analysis Result
#variant[Attention Analysis][Rollout Visualization][
#image("assets/attention_rollout_visualization.png", width: 100%)
Attention rollout visualizations reveal which parts of the image the model focuses on, providing interpretability for its predictions.
]
