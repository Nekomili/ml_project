# Material Detection Project

A university group project for detecting and classifying materials from images using computer vision and deep learning.

## Project Overview

This project demonstrates a complete machine learning pipeline from raw image data to trained object detection models. The team explored multiple approaches including YOLO and DenseNet architectures for material classification.

## Team

- Hiedanpää Sami
- Ilomäki Reetta
- Imporanta Minna
- Kitunen Joonas
- Kylmäniemi Mika

## Technical Approach

### Data Pipeline

The project implements a medallion-style data architecture:
- **Bronze**: Raw annotation data
- **Silver**: Cleaned and validated annotations
- **Gold**: Training-ready dataset with train/validation split

### Models Explored

| Model | Description |
|-------|-------------|
| **YOLO** | Object detection for material localization |
| **DenseNet** | Classification of material types |
| **Hybrid** | Combined detection + classification pipeline |

### Key Components

- Automated annotation import pipeline
- Data validation and outlier filtering
- Configurable train/validation/test splits
- Hyperparameter experimentation across multiple training runs

## Project Structure

```
├── data/               # Annotation metadata (anonymized)
├── notebooks/          # Data exploration and analysis
├── YOLO/               # Model training notebooks and results
└── src/code/           # Python pipeline scripts
```

## Key Learnings

| Topic | What We Learned |
|-------|-----------------|
| **Annotation Workflow** | Collaborative image labeling process |
| **YOLO Training** | Object detection with custom datasets |
| **DenseNet** | Transfer learning for classification |
| **Data Pipelines** | Bronze/Silver/Gold data architecture |
| **Team Collaboration** | Git workflow with multiple contributors |

## Technologies

- Python, PyTorch, Ultralytics YOLO
- SQLite for data management
- Docker for development environment
- JupyterLab for experimentation

## Results

Training metrics and model evaluation results are preserved in the training output folders.

---

## Disclaimer

> **⚠️ Important Notice**
>
> **This repository is intended to demonstrate project principles and processes, not to be executed.**
>
> Due to anonymization and repository size:
> - The project is **not runnable** in its current form
> - All credentials, API keys, and paths have been replaced with placeholders
> - Original image data is **not included**
> - Annotation data files (`data/labels_*`) have been removed due to large file size
> - The purpose is to **review the process and approach**, not to use the code directly
>
> This is a portfolio piece showing:
> - How we structured the ML pipeline
> - Our approach to data processing
> - Team collaboration and experimentation
>
> All client-related information has been removed.
>
> **Note:** The working language of this project was Finnish, so Finnish text and references may appear throughout the codebase. These have not been translated as project ownership belongs to the team.

---

*University course project - Machine Learning / Computer Vision*