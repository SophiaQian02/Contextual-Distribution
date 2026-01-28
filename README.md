# Code for ICML submission

This repository contains the implementation of **EntSE** (Entropy-based Structure Encoding) and **KerSE** (Kernel-based Structure Encoding) integrated into three different Graph Neural Network frameworks: **GraphGPS**, **KP-GNN**, and **PPGN**.

## Project Structure

The codebase is organized into three main folders, each corresponding to a framework integration:

*   **GraphGPS/**: Integration with the GraphGPS framework (Recipe for a General, Powerful, Scalable Graph Transformer).
*   **KP-GNN/**: Integration with Kernel Propagation Graph Neural Networks.
*   **PPGN/**: Integration with Provably Powerful Graph Networks.

## Methods

-   **EntSE**: Uses Von Neumann Entropy of the graph (or subgraphs) as a structural encoding.
-   **KerSE**: Uses Kernel-based features (random walk or diffusion kernels) as a structural encoding.

## Usage

### 1. GraphGPS

The GraphGPS implementation incorporates EntSE and KerSE as Positional/Structural Encodings (`posenc`).

**Configuration:**
-   **EntSE**: Configured via `posenc_VonNeumann`.
    -   Enable with `posenc_VonNeumann.enable: True`
    -   Key parameter: `k_order`
-   **KerSE**: Configured via `posenc_VonNeumannKernel`.
    -   Enable with `posenc_VonNeumannKernel.enable: True`
    -   Key parameters: `k_order`, `n_neighbors`

**Running Experiments:**
You can run experiments using the provided config files or by overriding parameters.

Example (ZINC dataset):
```bash
cd GraphGPS

# Run with EntSE
python main.py --cfg configs/GPS/zinc-GPS+Von.yaml

# Run with KerSE
python main.py --cfg configs/GPS/zinc-GPS+VonKernel.yaml
```

### 2. KP-GNN

The KP-GNN implementation adds specific encoders for EntSE and KerSE.

**Implementation Details:**
-   **EntSE**: Implemented via `LinearEntEncoder`.
-   **KerSE**: Implemented via `LinearKerEncoder`.

**Running Experiments:**
The method selection logic is handled in the training scripts (e.g., `train_CSL.py`, `train_TU.py`), attempting to load EntSE/KerSE based on model arguments.

Example:
```bash
cd KP-GNN
python train_CSL.py --model_name KerSE ...
```

### 3. PPGN

The PPGN implementation modifies the dataset and utility functions to compute and append EntSE and KerSE features to the input.

**Configuration:**
-   **EntSE**: Controlled by `use_subgraph_entropy`.
    -   Parameters: `subgraph_entropy_k`.
-   **KerSE**: Controlled by `use_kernel_features`.
    -   Parameters: `kernel_k`, `kernel_m`.

**Running Experiments (ZINC):**
Navigate to the ZINC directory and run the main script. You may need to modify the configuration `utils.py` or arguments to enable the specific flags.

```bash
cd PPGN/ZINC
python main.py
```

## Requirements

Please refer to the individual folders for specific installation instructions and requirements, as each framework may have different dependencies.
