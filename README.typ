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
  variant: ("Variant",),
)

#show: frame-style(styles.boxy)

// Frame 1: Patch Embedding
#feature[Patch Embedding][Initial tokenization of input images][
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
]

#pagebreak()

// Frame 2: Hierarchical Stages
#feature[Hierarchical Stages][Stacked transformer blocks with progressive downsampling][
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
]

#pagebreak()

// Frame 3: Feature Pyramid
#feature[Feature Pyramid][Multi-scale output representations for downstream tasks][
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
]

#pagebreak()

// Frame 4: Efficiency Optimizations
#feature[Efficiency Optimizations][Includes optimized attention mechanisms and memory-efficient implementation][
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
]
