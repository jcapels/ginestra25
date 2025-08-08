# **GINESTRA**: **G**raph-based **embeddIN**gs for **E**xploring **S**econdary-metabolite **T**axonomy, **R**ecognition, and **A**nnotation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)

This repository contains the official PyTorch implementation for the research project **GINESTRA**.

> **GINESTRA**: **G**raph-based **embeddIN**gs for **E**xploring **S**econdary-metabolite **T**axonomy, **R**ecognition, and **A**nnotation.

The GINESTRA project explores the application of various Graph Neural Network (GNN) architectures to the challenge of classifying natural products and secondary metabolites directly from their molecular graphs.

## Citation

This repository is inspired to our paper _Leveraging Molecular Graphs for Natural Product Classification_. 
If you use this code or the ideas presented in our work, please cite our paper:

```bibtex
@inproceedings{moleculargnp2025,
  author    = {Prete, Alessia Lucia and Corradini, Barbara Toniella and Costanti, Filippo and Scarselli, Franco and Bianchini, Monica},
  title     = {Leveraging Molecular Graphs for Natural Product Classification},
  booktitle   = {Computers in Biology and Medicine},
  year      = {2025},
  note      = {Under review}
}
```

---

## Overview

Natural Products (NPs) represent a rich source of bioactive compounds with high structural diversity and therapeutic potential.
Automatic classification of NPs is critical to ensure safety, support regulatory compliance, inform product usage, and enable the discovery of new pharmacologically relevant molecules. 
However, traditional rule-based approaches and hand-crafted molecular fingerprints often fall short in capturing the structural and biosynthetic complexity of NPs.
GNNs are well-suited for this task, as they can model both the topology and local chemical environments of molecules. 
We evaluate multiple GNN architectures on curated NP dataset and assess their ability to generalize across hierarchical classification targets.
These findings highlight the potential of GNNs as effective tools for NP classification. 
By leveraging graph-based representations, GNNs offer a scalable, data-driven approach that better reflects the structural and functional complexity of natural products. 
This work provides methodological guidance and encourages broader adoption of deep learning in natural product research and drug discovery.

## Models Implemented

This repository includes clean, modular, and reproducible implementations for the following models:

-   **GCN** (Graph Convolutional Network)
-   **GAT** (Graph Attention Network)
-   **GIN** (Graph Isomorphism Network)
-   **GINE** (Graph Isomorphism Network with Edge Features)
-   **GATE** (Graph Attention Network with Edge Features)
-   **MLP** (Multi-Layer Perceptron, as a baseline)

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/GINESTRA.git
    cd GINESTRA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    Run the setup/packages_installer.sh file to install the required packages
    ```bash 
    bash ./setup/packages_installer.sh
    ```
---

## Usage

### 1. Configuration

All hyperparameters, dataset paths, and experiment settings can be configured in the `config.py` file. The `PARAM_GRID` dictionary is used to define the search space for the manual grid search.

### 2. Running an Experiment

To run an experiment for a specific model, execute its corresponding script. For example, to run the GIN model:

```bash
python GIN_main.py
```

The script will automatically:
-   Load the dataset specified in `config.py`.
-   Iterate through all hyperparameter combinations defined in `PARAM_GRID`.
-   Train and evaluate the model for `N_RUNS` with different random seeds.
-   Apply early stopping to prevent overfitting.
-   Save the best model weights, logs, and final statistics in a timestamped folder inside the `experiments/` directory.

---

## Repository Structure

```
GINESTRA/
│
├── experiments/              # Output directory for models, logs, and reports
├── data/                     # Directory for datasets
├── models/                   # GNN model definitions (.py files)
│   ├── GIN.py
│   └── ...
├── utils/                    # Utility functions (early stopping, seeding, etc.)
│   ├── earlystop.py
│   └── ...
│
├── config.py                 # Main configuration file
├── GIN_main.py  # Experiment script for GIN
├── GCN_main.py  # Experiment script for GCN
└── ...                       # Other experiment scripts
│
└── README.md                 # This file
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions regarding the code or the paper, please contact [alessia.prete@example.com](mailto:alessia.prete@example.com) or open an issue in this repository.
