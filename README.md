# Anonymous Artifact: Intrusion Detection in Data Streams

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![CapyMOA](https://img.shields.io/badge/Framework-CapyMOA-orange)](https://capymoa.org/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-green)](https://optuna.org/)

This repository contains the source code, generated datasets, experimental outputs, and execution notebooks used to evaluate intrusion detection models in data stream scenarios.

The repository was prepared for a double-blind review process. Author names, institutional affiliations, personal identifiers, acknowledgments, and repository history were omitted to preserve anonymity during the review stage.

## Overview

The project evaluates machine learning models for intrusion detection in continuous data streams. The experimental protocol considers two predictive approaches: supervised stream classifiers and anomaly detection methods.

The implementation includes:

1. Supervised stream classifiers evaluated in a prequential setting;
2. Anomaly detection methods that produce continuous anomaly scores;
3. Thresholding and decision strategies for converting anomaly scores into binary predictions;
4. Warm-up handling before computing final global metrics;
5. Optimization routines based on Optuna;
6. Experimental CSV outputs and graphical outputs used in the analysis.

## Repository Structure

```text
.
├── data/
│   └── ...
│
├── output/
│   └── ...
│
├── src/
│   ├── Anomaly/
│   │   ├── Models.py
│   │   ├── Optimizer.py
│   │   └── Pipeline.py
│   │
│   ├── Classification/
│   │   ├── Models.py
│   │   ├── Optimizer.py
│   │   └── Pipeline.py
│   │
│   ├── Data/
│   │   ├── Handler.py
│   │   ├── Processor.py
│   │   └── ScenarioGenerator.py
│   │
│   └── Results/
│       ├── BestResultsTable.py
│       ├── Metrics.py
│       ├── Plots.py
│       ├── PlotsBestModels.py
│       └── TableResults.py
│
├── .gitattributes
├── .gitignore
├── AnomalyDetection.ipynb
├── Classification.ipynb
├── Database.ipynb
├── Results.ipynb
├── requirements.txt
└── README.md
```

## Source Code Modules

### `src/Anomaly/`

This module contains the anomaly detection pipeline.

- `Models.py`: definition and configuration of anomaly detection models;
- `Optimizer.py`: hyperparameter and decision-strategy optimization routines;
- `Pipeline.py`: execution flow for anomaly detection experiments.

### `src/Classification/`

This module contains the supervised stream classification pipeline.

- `Models.py`: definition and configuration of supervised stream classifiers;
- `Optimizer.py`: hyperparameter optimization routines;
- `Pipeline.py`: execution flow for prequential classification experiments.

### `src/Data/`

This module contains routines for dataset loading, preprocessing, and scenario generation.

- `Handler.py`: data loading and handling utilities;
- `Processor.py`: preprocessing routines used before stream construction;
- `ScenarioGenerator.py`: generation of experimental scenarios used in the evaluation.

### `src/Results/`

This module contains routines for metric computation, result organization, and plotting.

- `BestResultsTable.py`: generation of tables with the best experimental configurations;
- `Metrics.py`: computation of global and stream-based metrics;
- `Plots.py`: general plotting routines;
- `PlotsBestModels.py`: plotting routines for selected best-performing models;
- `TableResults.py`: organization and export of result tables.

## Evaluated Models

### Anomaly Detection

The anomaly detection module includes methods that estimate the degree of deviation of each instance from the behavior learned from the stream.

Evaluated methods include:

- Autoencoder;
- Adaptive Isolation Forest;
- Half-Space Trees.

### Supervised Classification

The classification module includes supervised algorithms updated incrementally during stream processing.

Evaluated methods include:

- Leveraging Bagging;
- Hoeffding Adaptive Tree;
- Adaptive Random Forest;
- Hoeffding Tree.

## Experimental Protocol

The experiments are executed in a data stream setting. Each instance is processed sequentially according to the evaluation strategy adopted for each predictive approach.

For supervised classifiers, the experiments follow a prequential evaluation procedure, in which each instance is first tested and then used to update the model.

For anomaly detectors, models produce anomaly scores that are converted into predictions using threshold-based decision strategies.

### Warm-up

All models use an initial warm-up period of **2,000 samples**. These samples are used only for model initialization and adaptation. Final global metrics reported at the end of the stream do not include the warm-up samples.

## Metrics

The framework computes traditional classification metrics and stream-oriented outputs used in the experimental analysis.

The implemented metrics include:

- F1-score;
- Precision;
- Recall;
- MCC;
- FPR;
- TPR;
- False Positives;
- False Negatives.

The project also includes graphical analyses associated with attack regions in the stream, allowing the inspection of model behavior during and after attack intervals.

## Data and Outputs

The `data/` directory contains the generated CSV scenarios used in the experiments. These files are provided to support reproducibility of the submitted work.

The `output/` directory contains experimental outputs used in the analysis, such as generated tables, intermediate CSV files, and plots.

Raw traffic captures and original external datasets are not included. The CSV files included in this repository correspond to processed scenarios used by the experimental pipeline.

If Git LFS is enabled for CSV files or images, make sure these files were correctly downloaded before running the notebooks:

```bash
git lfs install
git lfs pull
git lfs ls-files
```

## Installation

### Requirements

- Python 3.9 or higher;
- Java JRE/JDK, required by MOA/CapyMOA;
- Git LFS, required if the repository stores generated CSV files and graphical outputs through LFS.

### Clone the Repository

During the review process, this repository should be accessed through the anonymized link provided in the submitted paper.

```bash
git clone <anonymous-repository-url>
cd <repository-folder>
```

### Create a Virtual Environment

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

The experiments can be executed through the notebooks available in the repository root.

### Anomaly Detection

Use:

```text
AnomalyDetection.ipynb
```

This notebook loads the dataset, builds the data stream, applies preprocessing, executes anomaly detection models, performs optimization, and generates the corresponding metrics and plots.

### Supervised Classification

Use:

```text
Classification.ipynb
```

This notebook evaluates supervised classifiers in a prequential setting.

### Dataset Processing

Use:

```text
Database.ipynb
```

This notebook contains routines for dataset preparation and scenario generation.

### Results and Plots

Use:

```text
Results.ipynb
```

This notebook contains routines for reading experimental outputs, organizing metrics, and generating plots used in the experimental analysis.

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

## Reproducibility Notes

To reproduce the experiments:

1. Install the dependencies listed in `requirements.txt`;
2. Ensure that Git LFS files were downloaded, when applicable;
3. Run `Database.ipynb` if scenario generation is required;
4. Execute `AnomalyDetection.ipynb` and `Classification.ipynb`;
5. Generate the final plots and tables using `Results.ipynb`.

Due to optimization procedures and possible implementation-level randomness, minor numerical variations may occur across executions. When applicable, random seeds should be fixed in the corresponding notebook or script before execution.

## Anonymity Notice

This repository was prepared for double-blind review. Author names, affiliations, acknowledgments, repository ownership information, personal identifiers, and historical commit metadata were removed or omitted.

The non-anonymized repository information should be restored only in the final version, if the paper is accepted.
