# Speaker Recognition

## Objective
Train a speaker recognition system using speech embeddings extracted from log-Mel spectrograms.

## Dataset
VoxCeleb2
[Original paper] (https://arxiv.org/pdf/1806.05622)

## Suggested starting models
ResNet18/34 audio classification baselines
## Tasks
Extract log-Mel spectrograms
Fine-tune speaker classification model
Extract speaker embeddings and visualize (t-SNE/UMAP)
Analyze confusion between speakers
## Evaluation metrics
Speaker identification accuracy
ROC curves

## Done

- Filtered speakers with insufficient data
- Generated train/dev/test split metadata
- Extracted log-Mel spectrogram features
- Saved metadata files (CSV and JSON) in the data directory
