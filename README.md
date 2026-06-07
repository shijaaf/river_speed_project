# River Speed Estimation Project

## Overview

This project estimates river flow velocity from video recordings using computer vision, machine learning, and deep learning techniques.

The project was developed as a complete experimental pipeline that investigates multiple approaches for river surface velocity estimation and compares their performance.

The final solution combines:

- Optical Flow Motion Features
- Advanced Motion Descriptors
- Deep Features extracted using ResNet18
- Ensemble Machine Learning Models

The project is organized into multiple phases, starting from dataset analysis and video preprocessing, and ending with hybrid feature fusion and deep learning experiments.

---

# Dataset

The dataset contains:

- River flow videos (`.avi`, `.mp4`)
- Ground-truth flow velocity labels

Dataset characteristics:

- 24 river videos
- 22 valid labeled samples used for training and evaluation
- Different camera viewpoints
- Different lighting conditions
- Seeded and unseeded river surfaces
- Stabilized and non-stabilized videos

---

# Project Structure

```text
river_speed_project/

├── data/
│   ├── videos/
│   └── labels/
│
├── outputs/
│   ├── frames/
│   ├── logs/
│   ├── models/
│   └── results/
│
├── src/
│   ├── read_labels.py
│   ├── video_info.py
│   ├── preprocess.py
│   ├── optical_flow_features.py
│   ├── train_ml_model.py
│   ├── train_dl_model.py
│   ├── improved_features.py
│   ├── train_improved_model.py
│   ├── tracking_features.py
│   ├── train_klt_models.py
│   ├── advanced_motion_features.py
│   ├── train_advanced_motion_model.py
│   ├── train_hybrid_model.py
│   ├── deep_feature_extraction.py
│   ├── train_deep_feature_model.py
│   ├── train_hybrid_deep_model.py
│   ├── train_hybrid_deep_tinycnn.py
│   ├── train_hybrid_deep_cnnlstm.py
│   ├── train_tuned_ensemble.py
│   └── finalize_project.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Project Phases

## Phase 1 — Dataset Analysis

- Read label file
- Validate video files
- Extract video metadata
- FPS
- Duration
- Resolution

---

## Phase 2 — Video Preprocessing

- Frame extraction
- Resize
- Grayscale conversion
- Gaussian blur

---

## Phase 3 — Optical Flow Feature Extraction

Dense Farneback Optical Flow:

- Motion Magnitude
- Motion Direction
- Statistical Motion Features

---

## Phase 4 — Machine Learning Baseline

Models:

- Random Forest
- Extra Trees
- Gradient Boosting
- SVR
- Ridge Regression

Evaluation:

- Leave-One-Out Cross Validation (LOOCV)

---

## Phase 5 — Deep Learning Baseline

Model:

- MLP Regressor

Purpose:

- Baseline deep learning experiment

---

## Phase 6 — Improved Motion Features

Additional motion descriptors:

- Flow X statistics
- Flow Y statistics
- Motion density
- Motion per second
- Moving area ratio

---

## Phase 7 — KLT Tracking Features

Algorithms:

- Shi-Tomasi Corner Detection
- Lucas-Kanade Tracking

---

## Phase 8 — KLT Regression Models

Machine learning models trained using KLT tracking features.

---

## Phase 9 — Advanced Motion Features

Advanced motion descriptors:

- Motion Energy
- Motion Entropy
- Active Motion Ratio
- Direction Consistency

---

## Phase 10 — Advanced Motion Modeling

Model comparison using advanced motion features.

---

## Phase 11 — Hybrid Feature Fusion

Feature fusion:

- Improved Motion Features
- Advanced Motion Features

---

## Phase 12 — Feature Optimization

- Feature Selection
- Model Finalization

---

## Phase 13 — Deep Feature Extraction

Feature extraction using:

- ResNet18
- Transfer Learning

512-dimensional deep feature vectors are extracted from each video.

---

## Phase 14 — Hybrid Motion + Deep Features

Feature fusion:

- Optical Flow Features
- Advanced Motion Features
- ResNet18 Deep Features

Best project performance obtained in this phase.

---

## Phase 15 — Tiny CNN Regression

Deep learning regression using:

- 1D Tiny CNN

---

## Phase 16 — CNN-LSTM Regression

Sequence modeling using:

- CNN
- LSTM

---

## Phase 17 — Tuned Ensemble Models

Weighted ensemble of:

- Extra Trees
- Random Forest
- Gradient Boosting
- SVR
- Ridge

---

# Final Model

Final selected model:

```text
Hybrid Motion Features
+
Advanced Motion Features
+
ResNet18 Deep Features
+
Extra Trees Regressor
```

Best achieved performance:

```text
MAE ≈ 0.356 m/s
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

⚠️ **Important**

Do **not** run individual phase scripts directly unless you are performing development or experimentation.

The project entry point is:

```bash
python main.py
```

`main.py` orchestrates the complete pipeline, including:

- Dataset analysis
- Video preprocessing
- Feature extraction
- Model training
- Evaluation
- Result generation
- Final report generation

---

# Outputs

Generated files are stored inside:

```text
outputs/
```

### Results

```text
outputs/results/
```

Contains:

- Evaluation metrics
- Prediction tables
- Feature importance reports
- Generated datasets
- Final project reports

### Models

```text
outputs/models/
```

Contains:

- Trained machine learning models
- Deep learning checkpoints
- Final hybrid models

### Logs

```text
outputs/logs/
```

Contains:

- Prediction logs
- Processing time logs
- Speed estimation logs

---

# Evaluation Metrics

The following metrics are reported:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

Primary evaluation metric:

```text
MAE (Mean Absolute Error)
```

---

# Final Deliverables

The project produces:

- Trained machine learning models
- Trained deep learning models
- Prediction tables
- Evaluation metrics
- Feature importance analysis
- Processing logs
- River flow speed estimation logs
- Final Persian technical report

---

# Author

River Speed Estimation Project

Computer Vision • Machine Learning • Deep Learning 
