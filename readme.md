# Graph Neural Networks for Molecule Property Prediction (QM9 Dataset)

This repository contains code and notebooks for a project focused on predicting molecular properties, specifically the dipole moment, using Graph Neural Networks (GNNs) on the QM9 dataset.


## Project Overview

This project investigates the application of Graph Neural Networks (GNNs) for predicting the dipole moment (μ) of small organic molecules from the QM9 dataset.  We explore and compare the performance of several popular GNN architectures, including:

*   **Graph Convolutional Network (GCN)**
*   **Graph Isomorphism Network (GIN)**
*   **Graph Attention Network (GAT)**
*   **Graph Transformer Network (GraphTransformer)**

The goal is to demonstrate the effectiveness of GNNs for learning molecular representations and predicting quantum chemical properties, and to analyze the strengths and weaknesses of different GNN architectures for this task.

## Notebook Descriptions

*   **`dataset_EDA.ipynb`**: This notebook performs Exploratory Data Analysis (EDA) on the QM9 dataset. It includes:
    *   Loading and inspecting the dataset.
    *   Visualizing molecular structures (2D and 3D).
    *   Analyzing dataset statistics (molecular size, weight, bond lengths, etc.).
    *   Exploring the chemical space using dimensionality reduction techniques (t-SNE, PCA).
    *   Performing graph theoretical analysis and intermolecular similarity analysis.


*   **`training.ipynb`**: This is the main notebook for the project, containing the implementation of the GNN models, the training and evaluation pipeline, and the code for generating results and plots.  It includes:
    *   Definitions for GCN, GIN, GAT, and GraphTransformer models (and potentially GTN, MPNN, EdgeConv).
    *   Data loading and preprocessing for the QM9 dataset.
    *   Training and evaluation functions.
    *   Code to train and compare the performance of different GNN architectures.
    *   Plotting of validation loss curves and ground truth vs. prediction scatter plots.



## Model Files

The repository includes saved model weights files (``.pt`` files) for each of the trained GNN architectures:

*   `GCN_model.pt`
*   `GIN_model.pt`
*   `GAT_model.pt`
*   `GraphTransformer_model.pt`
*   `GTN_model.pt` (Potentially Graph Transformer Network or another GNN variant)
*   `MPNN_model.pt` (Potentially a Message Passing Neural Network model)

These files can be loaded to re-evaluate the trained models or for further experimentation.

## Data Directory (`data/`)

The `data/` directory is intended to store the QM9 dataset. The notebook code assumes the QM9 dataset is downloaded and placed in this directory.  The dataset is automatically downloaded and processed by PyTorch Geometric when you run the notebooks for the first time.

## Findings

*   **GraphTransformer Outperforms Others:** The GraphTransformer Network achieved the lowest test Mean Squared Error (MSE) loss for dipole moment prediction, demonstrating the **importance of edge attributes as features** for superior performance compared to GCN, GIN, and GAT in this experiment.
*   **Attention Mechanisms are Beneficial:** GAT also showed strong performance, highlighting the effectiveness of attention mechanisms in capturing relevant molecular features for property prediction.
*   **GIN Instability:** The GIN model exhibited training instability and poorer generalization performance compared to other architectures, suggesting it might not be as well-suited for this specific task or require further hyperparameter tuning.
*   **Computational Trade-off:**  More complex, attention-based models (GAT and GraphTransformer) generally required longer training times than simpler models (GCN, GIN), indicating a trade-off between model complexity/performance and computational cost.
*   **Data Normalization Improves Training:** Data normalization of the target dipole moment property was implemented, contributing to more stable and potentially faster training.


## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AyoubFrihaoui/Graph-Neural-Networks-GNNs---Molecular-Property-Prediction
    cd Graph-Neural-Networks-GNNs---Molecular-Property-Prediction
    ```
2.  **Install Dependencies:** Ensure you have the required libraries installed. You can use `mamba` or `pip` as indicated in the notebooks.  A typical installation command might look like:
    ```bash
    %mamba install -q -y -c pyg pyg
    %pip install numpy pandas matplotlib torch_geometric tqdm scikit-learn py3Dmol rdkit
    ```
    Run this command within a notebook cell.
3.  **Run the Notebooks:** Open and run the notebooks sequentially:
    *   `dataset_EDA.ipynb` (to explore the dataset)
    *   `training.ipynb` (optional, if you want to explore specific training code)

## Contact

For questions or inquiries, please contact a.frihaoui@esi-sba.dz

---

