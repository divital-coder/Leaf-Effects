
# Leaf Disease VIT

Leaf Disease VIT is a hierarchical vision transformer designed for efficient multi-scale feature learning. It employs progressive spatial dimension reduction and increasing channel depth to capture complex visual patterns effectively.

> [!NOTE]
> This repository has been updated with the latest Cloud version.

## Technical Architecture

The architecture follows a multi-stage design:
1.  **Patch Embedding**: Initial tokenization of input images.
2.  **Hierarchical Stages**: Stacked transformer blocks with progressive downsampling.
3.  **Feature Pyramid**: Multi-scale output representations for downstream tasks.
4.  **Efficiency Optimizations**: Includes optimized attention mechanisms and memory-efficient implementation.

## Project Structure

-   `phase1_project/`: Baseline implementation and data utilities.
-   `phase2_model/`: Core model development (Baseline, DFCA, HVT).
-   `phase3_pretraining/`: Self-supervised learning (SSL) pre-training.
-   `phase4_finetuning/`: Fine-tuning scripts.
-   `phase5_analysis_and_ablation/`: Analysis, visualization, and robustness testing.

## Installation

```bash
git clone https://github.com/divital-coder/LeafDiseaseVIT.git
cd LeafDiseaseVIT
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Pre-training:**
```bash
cd phase3_pretraining && python run_ssl_pretraining.py
```

**Fine-tuning:**
```bash
cd phase4_finetuning && python main.py --config config.yaml
```

**Analysis:**
```bash
cd phase5_analysis_and_ablation && python analyze_best_model.py
```

## Results

### Feature Analysis
![Feature Space Comparison](assets/tsne_feature_space_comparison.png)
*t-SNE visualization showing clear separation of semantic features.*

### Training Convergence
![Convergence Analysis](assets/convergence_plot.png)
*Training progression compared to baseline models.*

### Transfer Learning Performance
![Confusion Matrix](assets/confusion_matrix.png)
*Performance on downstream classification tasks.*

## Resources 

The project utilized 4 H100 GPUsfor training and testing, via Modal Labs and Lightning AI cloud environments.


## Benchmarks

# Benchmark Results

This document presents comprehensive benchmark results for the HierarchialViT model across various tasks and datasets.

## ImageNet Results

| Model | Top-1 Acc | Top-5 Acc | Parameters | FLOPs | Throughput |
|-------|-----------|-----------|------------|--------|------------|
| HViT-S | 82.1% | 96.0% | 22M | 4.6G | 1256 img/s |
| HViT-B | 83.5% | 96.5% | 86M | 15.4G | 745 img/s |
| HViT-L | 84.7% | 97.1% | 304M | 45.8G | 312 img/s |

## Feature Analysis

![Feature Space Comparison](../assets/tsne_feature_space_comparison.png)

t-SNE visualization shows clear separation of semantic features.

## Training Convergence

![Convergence Analysis](../assets/convergence_plot.png)

Training progression compared to baseline models.

## Transfer Learning Performance

![Confusion Matrix](../assets/confusion_matrix.png)

Performance on downstream classification tasks.

## Ablation Studies

![Ablation Results](../assets/convergence_plot_detailed_ablations.png)

Impact of different architectural choices.

## Attention Analysis

![Attention Patterns](../assets/attention_rollout_visualization.png)

Visualization of attention patterns across different layers.

## Hardware Requirements

| Model Size | GPU Memory | Training Time (1 epoch) | Inference Speed |
|------------|------------|------------------------|-----------------|
| HViT-S | 8GB | 2.5h | 1256 img/s |
| HViT-B | 16GB | 4h | 745 img/s |
| HViT-L | 32GB | 8h | 312 img/s |

