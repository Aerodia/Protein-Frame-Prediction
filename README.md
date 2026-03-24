# Protein Frame Prediction (PFP)

A scalable, physics-aware deep learning system for predicting future atomic configurations of proteins using a hybrid temporal and spatial modeling approach.

## Overview

Protein dynamics are fundamental to understanding biological processes such as ligand binding, protein folding, and allosteric regulation. Traditional Molecular Dynamics (MD) simulations provide detailed insights but are computationally expensive and time-consuming. This project proposes a machine learning-based alternative to predict the next atomic frame in a protein trajectory with high physical accuracy, enabling faster and more scalable simulation of protein dynamics.

## Key Idea

The system separates the learning task into two complementary components:

| Component | Role |
|---|---|
| **LSTM (Temporal Expert)** | Captures time-dependent motion patterns from sequential frames |
| **E3NN (Spatial Expert)** | Enforces geometric and physical constraints in 3D space |

This design ensures that predictions are both **dynamically consistent** and **physically valid**.

##  Features

- Sliding window-based temporal modeling (15–25 frames)
- Bidirectional LSTM with multiple stacked layers
- E(3)-Equivariant Graph Neural Network for spatial reasoning
- Physics-aware loss functions
- Autoregressive multi-step prediction capability
- Efficient handling of large-scale datasets (up to terabytes)

## Data Pipeline

### Input
- Molecular Dynamics trajectories in **PDB format**

### Preprocessing Steps
1. Centering to remove global translation
2. Scaling using radius of gyration
3. Temporal window construction
4. Graph construction for GNN input
5. Data sharding for efficient I/O

### Output Format
```
(Frames, Atoms, Coordinates)
Example: (10000, 3441, 3)
```

## Model Architecture

The system is built on two tightly integrated components that jointly model the **temporal** and **spatial** dynamics of protein motion.

### Temporal Model — LSTM

The LSTM-based temporal model captures **sequential dependencies across frames**, learning motion and velocity patterns over time. Bidirectional processing allows the model to leverage both past and future context, yielding a richer understanding of temporal dynamics.

### Spatial Model — E3NN

The E3NN-based spatial model represents the protein as a **molecular graph**, where nodes correspond to atoms and edges encode chemical bonds or spatial proximity. The model is **rotationally and translationally equivariant**, ensuring consistent predictions regardless of protein orientation, and refines predicted coordinates to enforce physical realism.

## Loss Function

Training uses a **composite loss** balancing accuracy with physical plausibility:

| Component | Purpose |
|---|---|
| **MSE Loss** | Positional accuracy of predicted coordinates |
| **Bond Length Penalty** | Preserves structural integrity |
| **Collision Penalty** | Prevents atomic overlap |
| **Contact Map Consistency** | Maintains correct spatial relationships between atoms |

## Evaluation Metrics

Model performance is assessed across multiple dimensions:

- **RMSD** — Root Mean Square Deviation for positional accuracy
- **Bond Length & Angle Deviation** — Structural correctness
- **Contact Map Similarity** — Verification of spatial interactions
- **Multi-step Rollout Stability** — Accuracy retention over successive prediction steps

##  Training Strategy

| Technique | Description |
|---|---|
| **Optimizer** | Adam with learning rate scheduling |
| **Batch Strategy** | Mini-batch gradient descent |
| **Teacher Forcing** | Applied in early training stages to stabilize sequence prediction |
| **Curriculum Learning** | Progressively increases prediction horizon from single-step to multi-step forecasting |

## Challenges & Mitigation

| Challenge | Mitigation Strategy |
|---|---|
| Overfitting on repetitive MD data | Predict displacement instead of absolute coordinates; apply noise injection and data augmentation; use temporal striding to reduce redundancy; incorporate physics-based regularization |
| Large-scale data handling | Data sharding and streaming; efficient batching and preprocessing pipelines |

##  Applications

-  Drug discovery and virtual screening
-  Protein folding analysis
-  Accelerated molecular simulations
-  Computational biology research

##  Tech Stack

| Library | Purpose |
|---|---|
| **Python** | Core language |
| **PyTorch** | Deep learning framework |
| **PyTorch Geometric** | Graph neural network support |
| **E3NN** | Equivariant neural networks |
| **NumPy / SciPy** | Numerical computation |
| **HDF5** | Large-scale data handling |
