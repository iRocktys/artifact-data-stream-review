# Denial-of-Service Attack Variants: Benchmarking Streaming Anomaly Detection and Classification Methods

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![CapyMOA](https://img.shields.io/badge/Framework-CapyMOA-orange)](https://capymoa.org/) [![Optuna](https://img.shields.io/badge/Optimization-Optuna-green)](https://optuna.org/) [![Dataset](https://img.shields.io/badge/Dataset-CICDDoS2019-purple)](https://www.unb.ca/cic/datasets/ddos-2019.html) [![Git LFS](https://img.shields.io/badge/Large_Files-Git_LFS-informational)](https://git-lfs.com/)

This repository contains the public artifact associated with the paper **"Denial-of-Service Attack Variants: Benchmarking Streaming Anomaly Detection and Classification Methods"**, submitted to the **2026 Brazilian Symposium on Computing Systems Engineering (SBESC 2026)**.

The paper presents a benchmark of supervised classifiers and anomaly detection algorithms for **DoS/DDoS attack detection in data streams**. The experimental design uses dynamic scenarios derived from the CICDDoS2019 dataset to evaluate how different DoS/DDoS attack variants affect stream learning models over time. The benchmark compares supervised classifiers and anomaly detectors under four scenarios: **Consistency**, **Generalization**, **Adaptation**, and **Recurrence**.

This artifact includes the source code, processed CSV scenarios, experimental outputs, and execution notebooks required to inspect and reproduce the main results reported in the paper. The raw CICDDoS2019 files are not redistributed due to their size and external distribution conditions, but the processed CSV scenarios used in the experiments are included in the `data/` directory.

The repository is not anonymized and retains its public GitHub ownership and development history, as permitted for the SBESC 2026 submission.

---

# README.md Structure

This README is organized as follows:

1. **Project Title and Summary**: presents the artifact and its relationship to the submitted paper.
2. **README.md Structure**: describes the organization of this document.
3. **Artifact Availability**: summarizes the public availability and reproducibility support provided by the artifact.
4. **Basic Information**: describes the artifact components and execution environment.
5. **Dependencies**: lists software dependencies, benchmark data, and external resources.
6. **Security Concerns**: describes potential risks and safe execution recommendations.
7. **Installation**: explains how to clone, install, and prepare the artifact.
8. **Minimum Test**: provides a lightweight execution path to verify installation.
9. **Experiments**: describes how to reproduce the main claims of the paper.
10. **LICENSE**: presents the current licensing status of the repository.

---

# Artifact Availability

The artifact is publicly available in the project repository and includes code, processed CSV scenarios, notebooks, dependencies, and instructions for executing the main evaluation workflow.

Experimental outputs, plots, and tables are also provided to support inspection and reproduction of the reported results. Full regeneration of the processed CSV scenarios from the raw CICDDoS2019 dataset is optional and depends on manually obtaining the original dataset.

---

# Basic Information

## Artifact Components

The repository is organized as follows:

```text
.
├── data/
│   └── processed CSV scenarios used in the experiments
├── output/
│   └── experimental outputs, generated tables, plots, and PDF files
├── src/
│   ├── Anomaly/
│   │   ├── Models.py
│   │   ├── Optimizer.py
│   │   └── Pipeline.py
│   ├── Classification/
│   │   ├── Models.py
│   │   ├── Optimizer.py
│   │   └── Pipeline.py
│   ├── Data/
│   │   ├── Handler.py
│   │   ├── Processor.py
│   │   └── ScenarioGenerator.py
│   └── Results/
│       ├── BestResultsTable.py
│       ├── Metrics.py
│       ├── PlotPaper.py
│       ├── Plots.py
│       ├── PlotsBestModels.py
│       └── TableResults.py
├── AnomalyDetection.ipynb
├── Classification.ipynb
├── Database.ipynb
├── Results.ipynb
├── requirements.txt
└── README.md
```

Some file and directory names may use UTF-8 characters. The repository structure should be preserved when running the notebooks, since several paths are referenced directly by the experimental pipeline.

## Execution Environment

The artifact was designed to run in a local Python environment using Jupyter notebooks.

Recommended software environment:

- Operating system: Linux, macOS, or Windows.
- Python: 3.9 or higher.
- Java: JRE/JDK installed and available in the system path, required by MOA/CapyMOA.
- Git LFS: required to download large CSV files, plots, and PDF outputs.
- Jupyter Notebook or JupyterLab.
- Internet access during installation to download Python packages.

The artifact was successfully tested with Python 3.12 in the authors' local environment. However, Python 3.9 or higher is expected to be supported, depending on the compatibility of the installed dependencies.

## Hardware Requirements

A standard desktop or laptop machine is sufficient for inspecting the artifact and running the minimum test.

Recommended resources:

- CPU: 4 cores or higher.
- RAM: at least 8 GB recommended.
- Disk: at least 2 GB free after cloning the repository and downloading Git LFS files.
- GPU: not required.

Running all optimization experiments may require substantially more time than the minimum test, especially because Optuna optimization and repeated executions are used in the full experimental workflow.

---

# Dependencies

## Python Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The main libraries used by the artifact include:

- CapyMOA: stream learning algorithms and data stream evaluation.
- Optuna: hyperparameter optimization.
- scikit-learn: preprocessing, metrics, and auxiliary machine learning routines.
- pandas: CSV reading and data manipulation.
- numpy: numerical operations.
- matplotlib: plots and visual outputs.
- Jupyter: notebook execution.

The exact dependency list should be obtained from the `requirements.txt` file included in the repository.

## External Benchmark Dataset

The original **CICDDoS2019** dataset is not included in this repository due to its size and external distribution requirements.

The processed CSV scenarios used in the experiments are included in the `data/` directory. Therefore, the main experiments can be reproduced without downloading the raw CICDDoS2019 files.

The `Database.ipynb` notebook documents the preprocessing and scenario generation procedure from the original CICDDoS2019 files. This notebook is optional for reproducing the main experiments and only runs if the raw dataset is manually downloaded and placed in the expected structure.

To regenerate the processed CSV scenarios from the original CICDDoS2019 dataset, place the raw files under:

```text
datasets/
└── CICDDoS2019/
    └── 01-12/
        └── <raw CICDDoS2019 files>
```

Alternatively, if the raw dataset is stored outside this repository, set the `CICDDOS2019_RAW_DIR` environment variable before running `Database.ipynb`.

Linux/macOS:

```bash
export CICDDOS2019_RAW_DIR="/path/to/CICDDoS2019/01-12"
```

Windows PowerShell:

```powershell
$env:CICDDOS2019_RAW_DIR="C:\path\to\CICDDoS2019\01-12"
```

## Git LFS Dependencies

This repository contains processed CSV scenarios, plots, and PDF outputs. Some of these files are tracked using Git LFS due to their size.

After cloning the repository, install Git LFS and download the tracked files:

```bash
git lfs install
git lfs pull
```

To verify that the LFS files were correctly downloaded, run:

```bash
git lfs ls-files
```

If CSV, image, or PDF files appear as small pointer files instead of complete files, run:

```bash
git lfs pull
```

---

# Security Concerns

The artifact does not execute network attacks and does not generate malicious traffic. The experiments operate only on CSV files containing preprocessed flow records and produce metrics, tables, plots, and PDF outputs.

Recommended safety practices for users:

1. Run the artifact in a local virtual environment.
2. Do not execute notebooks with administrative privileges.
3. Do not download or execute unknown external files beyond the documented dependencies.
4. Treat the raw CICDDoS2019 regeneration step as optional.
5. Use the processed CSV files already available in `data/` for the main artifact evaluation.

The artifact is intended only for offline experimental evaluation of DoS/DDoS detection models using existing data.

---

# Installation

## 1. Clone the Repository

Clone the public project repository:

```bash
git clone https://github.com/iRocktys/artifact-data-stream-review.git
cd artifact-data-stream-review
```

## 2. Download Git LFS Files

```bash
git lfs install
git lfs pull
```

Check whether LFS files are available:

```bash
git lfs ls-files
```

## 3. Create a Virtual Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 5. Check Java Availability

CapyMOA/MOA requires Java. Verify that Java is available:

```bash
java -version
```

If Java is not available, install a JRE/JDK compatible with CapyMOA/MOA before running the notebooks.

---

# Minimum Test

This section provides a lightweight test to verify whether the artifact environment is correctly installed.

## Step 1: Start Jupyter

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

## Step 2: Open the Results Notebook

Open:

```text
Results.ipynb
```

This notebook reads already generated experimental outputs and organizes the results into tables and plots. It is the recommended first test because it does not require regenerating the original CSV scenarios from the raw CICDDoS2019 dataset.

## Step 3: Execute the Initial Cells

Run the first cells that import dependencies and load result files from `output/`.

Expected behavior:

- Python imports should complete without errors.
- Files from `output/` should be found.
- Tables and/or plots should be generated from existing outputs.

Expected resource usage:

- RAM: approximately 1 GB to 4 GB, depending on the loaded outputs.
- Disk: no significant additional disk usage.
- Runtime: a few minutes for the initial cells.

If this test succeeds, the installation and data access through Git LFS are likely correct.

---

# Experiments

The full experimental workflow is organized around the processed CSV scenarios available in `data/`.

Recommended execution order:

1. `Results.ipynb`: inspect generated tables and plots from existing outputs.
2. `AnomalyDetection.ipynb`: execute anomaly detection experiments.
3. `Classification.ipynb`: execute supervised classification experiments.
4. `Database.ipynb`: optional notebook for regenerating scenarios from the original CICDDoS2019 dataset.

The raw CICDDoS2019 dataset is not required for reproducing the main results from the processed CSV files. It is only required if users want to regenerate the processed scenarios from scratch.

## Claim #1: Supervised classifiers achieve the highest overall performance in most scenarios

The paper reports that supervised classifiers, especially ensemble-based methods such as Adaptive Random Forest and Hoeffding Adaptive Tree, achieved the highest overall F1-scores across several DoS/DDoS stream scenarios.

### How to Reproduce

1. Ensure that Git LFS files were downloaded:

```bash
git lfs pull
```

2. Open:

```text
Classification.ipynb
```

3. Execute the cells that load processed CSV scenarios from `data/`.

4. Run the supervised classifier experiments for the four scenarios:

- Consistency;
- Generalization;
- Adaptation;
- Recurrence.

5. Compare the generated metrics with the tables and outputs available in `output/`.

No additional configuration file needs to be changed for this reproduction path. Parameters such as scenario name, poisoning level, model name, number of Optuna trials, and warm-up size can be adjusted directly in the corresponding notebook cells.

### Expected Output

The notebook should generate classification metrics such as:

- F1-score;
- Precision;
- Recall;
- False Positives;
- False Negatives.

Expected behavior:

- Ensemble-based supervised classifiers should appear among the best-performing models in several scenario and poisoning-level combinations.
- Results may show minor numerical differences due to stochastic components and optimization procedures.

### Expected Resources

- RAM: 4 GB to 8 GB recommended.
- Disk: no major additional disk usage beyond outputs.
- Runtime: depends on the number of scenarios and whether optimization is enabled; full execution may take several hours.

## Claim #2: Anomaly detectors are competitive in specific settings but depend strongly on thresholding and decision strategies

The paper reports that anomaly detectors such as Adaptive Isolation Forest and Half-Space Trees can achieve competitive results in specific scenarios, especially when combined with feature selection, threshold calibration, and window-based decision strategies.

### How to Reproduce

1. Ensure that Git LFS files were downloaded:

```bash
git lfs pull
```

2. Open:

```text
AnomalyDetection.ipynb
```

3. Execute the cells that load processed CSV scenarios from `data/`.

4. Run the anomaly detection experiments using the available models and threshold strategies.

5. Compare the generated metrics, plots, and optimization outputs with the files available in `output/`.

No additional configuration file needs to be changed for this reproduction path. Parameters such as detector name, threshold strategy, window-based decision strategy, number of Optuna trials, and warm-up size can be adjusted directly in the corresponding notebook cells.

### Expected Output

The notebook should produce:

- Anomaly scores;
- Binary predictions after thresholding;
- F1-score, Precision, Recall, FP, and FN;
- Plots showing model behavior over the stream.

Expected behavior:

- Anomaly detectors should show stronger dependence on threshold calibration and decision strategies than supervised classifiers.
- Window-based strategies such as moving average or temporal persistence may reduce unstable decisions caused by isolated anomaly score peaks.

### Expected Resources

- RAM: 4 GB to 8 GB recommended.
- Disk: additional output files may be generated.
- Runtime: may take several hours if all optimization procedures are enabled.

## Claim #3: Attack similarity affects model behavior over time

The paper reports that model behavior changes across the four scenarios: Consistency, Generalization, Adaptation, and Recurrence. Similar attack variants tend to be less challenging, while more distinct or alternating attack patterns may increase the occurrence of errors.

### How to Reproduce

1. Open:

```text
Results.ipynb
```

2. Load the existing outputs from the `output/` directory.

3. Generate the tables and plots comparing the four scenarios.

4. Inspect the plots of False Positives and False Negatives over the stream.

No additional configuration file needs to be changed for this reproduction path. Existing outputs in `output/` are sufficient to inspect the claim.

### Generate the Paper Figure

The consolidated 2-by-4 figure used in the paper can be generated from a Jupyter notebook cell:

```python
from src.Results.PlotPaper import PlotPaper

result = PlotPaper(
    projectRoot=".",
    showFigure=True,
).generate()
```

The figure is saved in both PDF and PNG formats:

```text
output/Results/plots_paper/BestModels_FP_FN_AllScenarios_2x4.pdf
output/Results/plots_paper/BestModels_FP_FN_AllScenarios_2x4.png
```

If the notebook is located in a subdirectory, set `projectRoot` to the project root, for example `projectRoot=".."`.

### Expected Output

The notebook should reproduce plots and tables showing differences in model behavior across:

- Consistency;
- Generalization;
- Adaptation;
- Recurrence.

Expected behavior:

- More stable behavior is expected in scenarios with repeated or similar attacks.
- More dynamic scenarios may show increased errors, especially for anomaly detectors.

### Expected Resources

- RAM: 1 GB to 4 GB.
- Disk: no major additional disk usage.
- Runtime: a few minutes to inspect existing outputs.

## Optional Experiment: Regenerating Processed CSV Scenarios

The `Database.ipynb` notebook can be used to regenerate the processed CSV scenarios from the raw CICDDoS2019 files.

This step is optional because the raw CICDDoS2019 dataset is not included in this repository.

### Required Setup

Place the original raw dataset files under:

```text
datasets/
└── CICDDoS2019/
    └── 01-12/
        └── <raw CICDDoS2019 files>
```

or set:

```bash
export CICDDOS2019_RAW_DIR="/path/to/CICDDoS2019/01-12"
```

Windows PowerShell:

```powershell
$env:CICDDOS2019_RAW_DIR="C:\path\to\CICDDoS2019\01-12"
```

### Expected Output

The notebook should generate processed CSV scenarios in `data/`.

### Expected Resources

- RAM: 8 GB recommended.
- Disk: depends on the size of the raw CICDDoS2019 files.
- Runtime: depends on the storage device and dataset location.

---

# LICENSE

No explicit open-source license is currently included in this repository. The artifact is publicly available for inspection and reproduction of the experiments; reuse, modification, or redistribution remains subject to permission from the authors until a formal license is added.
