# Benchmarking Anomaly Detection Algorithms and Supervised Classifiers for Data Streams in the Presence of Denial-of-Service Attack Variants

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![CapyMOA](https://img.shields.io/badge/Framework-CapyMOA-orange)](https://capymoa.org/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-green)](https://optuna.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CICDDoS2019-purple)](https://www.unb.ca/cic/datasets/ddos-2019.html)
[![Git LFS](https://img.shields.io/badge/Large_Files-Git_LFS-informational)](https://git-lfs.com/)

This repository contains the anonymous artifact associated with the submitted paper **"Adaptive Cybersecurity: Benchmarking Anomaly Detection and Classification in Dynamic Data Streams"**.

It includes the source code, processed CSV scenarios, experimental outputs, and execution notebooks used in the evaluation of anomaly detection and supervised classification models under dynamic data stream scenarios.

The repository was prepared for a double-blind review process. Author names, institutional affiliations, personal identifiers, acknowledgments, and repository history were omitted to preserve anonymity during the review stage.

## Overview

This project provides the anonymous artifact associated with a benchmark of supervised classifiers and anomaly detection algorithms for DoS/DDoS attack detection in data streams.

The study focuses on dynamic scenarios built from the CICDDoS2019 dataset, with emphasis on how different DoS/DDoS attack variants affect stream learning models. The experiments are centered on denial-of-service traffic patterns, including DNS-, LDAP-, and SYN-based attack variants.

The benchmark compares supervised classifiers and anomaly detectors under four experimental scenarios: Consistency, Generalization, Adaptation, and Recurrence. These scenarios were designed from an attack similarity analysis to evaluate how models behave when exposed over time to recurring, similar, distinct, and alternating attack patterns.

The implementation includes:

1. Supervised stream classifiers evaluated under the prequential Test-then-Train protocol;
2. Anomaly detection algorithms that continuously produce anomaly scores;
3. Feature selection based on attack behavior analysis;
4. Threshold calibration and window-based decision strategies for anomaly detection;
5. Hyperparameter optimization routines based on Optuna;
6. Processed CSV scenarios and experimental outputs used to reproduce the benchmark results.

## Repository Structure

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

## Evaluated Models

### Anomaly Detection

Evaluated methods:

- Autoencoder;
- Adaptive Isolation Forest;
- Half-Space Trees.

### Supervised Classification

Evaluated methods:

- Leveraging Bagging;
- Hoeffding Adaptive Tree;
- Adaptive Random Forest;
- Hoeffding Tree.

## Experimental Protocol

The experiments are executed in a data stream setting. Each instance is processed sequentially, and the models are evaluated according to the corresponding learning strategy.

For supervised classifiers, the experiments follow a prequential evaluation procedure, in which each instance is first tested and then used to update the model.

For anomaly detectors, models produce anomaly scores that are later converted into predictions using threshold-based decision strategies.

### Warm-up

All models use an initial warm-up period of **2,000 samples**. These samples are used only for model initialization and adaptation. Final global metrics reported at the end of the stream do not include the warm-up samples.

## Metrics

The implemented metrics include:

- F1-score;
- Precision;
- Recall;
- MCC;
- FPR;
- TPR;
- False Positives;
- False Negatives.

The project also includes behavioral analyses associated with attack regions in the stream, allowing the evaluation of model behavior during and after attack intervals.

## Dataset Availability

The original CICDDoS2019 dataset is not included in this repository due to its size and external distribution requirements.

However, the processed CSV scenarios used in the experiments are provided in the `data/` directory. These files correspond to the generated data stream scenarios evaluated in the submitted paper and can be used directly to reproduce the anomaly detection and classification experiments.

The `Database.ipynb` notebook documents the preprocessing and scenario generation procedure from the original CICDDoS2019 files. Since the raw CICDDoS2019 dataset is not included, this notebook is provided for methodological transparency and will only run if the original dataset is manually downloaded and placed in the expected local directory structure.

For artifact review, the main reproducibility path starts from the processed CSV files already available in `data/`.

### Optional raw dataset path

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

The raw CICDDoS2019 files are not required to run the main experiments, because the processed CSV scenarios are already available in `data/`.

## Installation

### Requirements

- Python 3.9 or higher;
- Java JRE/JDK, required by MOA/CapyMOA;
- Git LFS, required if downloading the repository with processed CSV files, plots, and PDF outputs.

### Clone the repository

During the review process, this repository should be accessed through the anonymized link provided in the submitted paper.

```bash
git clone <anonymous-repository-url>
cd <repository-folder>
```

If Git LFS files are not downloaded automatically, run:

```bash
git lfs install
git lfs pull
```

### Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Recommended Execution Path

For reviewers, the recommended execution path is:

1. Use the processed CSV scenarios available in `data/`;
2. Run `AnomalyDetection.ipynb` to evaluate anomaly detection models;
3. Run `Classification.ipynb` to evaluate supervised classifiers;
4. Run `Results.ipynb` to reproduce tables and plots.

The `Database.ipynb` notebook is optional for artifact review. It is only required if the reviewer wants to regenerate the CSV scenarios from the original CICDDoS2019 dataset.

## Notebook Description

### `AnomalyDetection.ipynb`

Loads the processed CSV scenarios, builds the data stream, applies preprocessing, executes anomaly detection models, performs optimization, and generates the corresponding metrics and plots.

### `Classification.ipynb`

Loads the processed CSV scenarios and evaluates supervised classifiers in a prequential setting.

### `Database.ipynb`

Documents the preprocessing and scenario generation procedure from the original CICDDoS2019 dataset. This notebook is not part of the default execution path for reviewers because the raw CICDDoS2019 files are not included in this repository.

### `Results.ipynb`

Reads experimental outputs, organizes metrics, and generates plots used in the experimental analysis.

## Example: Anomaly Detection Optimization

```python
from src.Anomaly.Optimizer import AnomalyOptunaOptimizer

optimizer = AnomalyOptunaOptimizer(
    stream=stream,
    n_trials=10,
    discretization_threshold="params",
    target_class="macro",
    target_class_pass=0,
    target_names=targets
)

best_model = optimizer.optimize(
    model_name="HST",
    warmup_instances=2000
)
```

## Outputs

The repository includes experimental outputs used in the analysis, such as:

- Processed CSV files;
- Prequential metric outputs;
- Optimization results;
- Graphical outputs;
- PDF files;
- Model behavior plots over attack regions.

These outputs are included to allow reviewers to inspect the experimental artifacts without necessarily rerunning all experiments from scratch.

## Git LFS Notes

This repository contains processed CSV scenarios, plots, and PDF outputs. Some of these files are tracked using Git LFS due to their size.

After cloning the repository, install Git LFS and download the tracked files:

```bash
git lfs install
git lfs pull
```

To verify that the LFS files were correctly tracked, run:

```bash
git lfs ls-files
```

If CSV, image, or PDF files appear as small pointer files instead of full files, run:

```bash
git lfs pull
```

## Reproducibility Notes

To reproduce the experiments:

1. Install the dependencies listed in `requirements.txt`;
2. Ensure that Git LFS files were downloaded;
3. Use the processed CSV scenarios already available in `data/`;
4. Execute the anomaly detection and classification notebooks;
5. Generate the final plots and tables using `Results.ipynb`.

Due to optimization procedures and possible implementation-level randomness, minor numerical variations may occur across executions. When applicable, random seeds should be fixed in the corresponding notebook or script before execution.

## Anonymity Notice

This repository was prepared for double-blind review. Author names, affiliations, acknowledgments, repository ownership information, and personal identifiers were removed or omitted.
