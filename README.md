# Symbolic Neurons for Robust Scientific Equation Discovery

Official implementation and experimental code accompanying the paper:

> **Symbolic Neurons for Robust Scientific Equation Discovery**

This repository provides the Python implementation, experimental protocols, benchmark configurations, and analysis scripts used to investigate **symbolic neurons for interpretable and robust scientific equation discovery**.

The project explores a symbolic learning paradigm in which both the **mathematical structure** of a model and its **continuous parameters** are learned from data. The resulting symbolic expressions are designed to remain interpretable while providing robustness to noise, structural uncertainty, and physical constraints.

---

## Overview

Modern machine learning models can achieve excellent predictive performance, but their learned representations are often difficult to interpret. In scientific applications, however, prediction alone is frequently insufficient: researchers may require an explicit mathematical relationship that can be inspected, validated, and potentially interpreted as a scientific law.

This project investigates **symbolic neurons** as interpretable computational units for scientific equation discovery.

A symbolic neuron can be viewed as a parameterised symbolic function

$$
y = f_S(x;\theta),
$$

where:

* \(x\) denotes the input variables;
* \(S\) is a symbolic structure selected from a discrete hypothesis space;
* \(\theta\) contains continuous parameters estimated from data;
* \(f_S\) is constructed from a predefined or learnable library of mathematical primitives.

The framework therefore separates two complementary learning problems:

1. **Structure learning** — discovering the mathematical form \(S\);
2. **Parameter learning** — estimating the numerical parameters \(\theta\).

The current experimental setting focuses on the single-neuron case, while the framework is designed to support future extensions toward compositional symbolic neural architectures.

---

## Key Features

The repository supports experiments involving:

* Symbolic regression
* Symbolic structure search
* Continuous parameter optimisation
* Interpretable equation discovery
* Robustness to noisy observations
* Structural uncertainty
* Complexity-aware model selection
* Physical constraints
* Explicit and implicit equations
* Symbolic primitive libraries
* Comparison with existing symbolic regression approaches
* Reproducible benchmark experiments
* Automatic generation of discovered mathematical expressions
* Quantitative and qualitative evaluation

---

## Conceptual Framework

The symbolic neuron is defined by the combination of learned structure and learned parameters:

$$
f_{S,\theta}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}.
$$

The general optimisation problem can be written as

$$
(S^\ast,\theta^\ast)
=
\arg\min_{S,\theta}
\mathcal{L}
\left(
y,
f_S(x;\theta)
\right)
+
\lambda\,\Omega(S),
$$

where:

* \(\mathcal{L}\) is the data-fitting loss;
* \(\Omega(S)\) is a complexity or parsimony penalty;
* \(\lambda\) controls the trade-off between accuracy and symbolic complexity.

This formulation allows the framework to explicitly balance:

> **accuracy ↔ interpretability ↔ complexity ↔ robustness**

---
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ALKhater-20/Symbolic-Neurons-for-Robust-Scientific-Equation-Discovery.git
cd Symbolic-Neurons-for-Robust-Scientific-Equation-Discovery

```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, if the project provides a Conda environment:

```bash
conda env create -f environment.yml
conda activate symbolic-neurons
```

---

## Quick Start

A minimal example can be run with:

```bash
python scripts/run_baseline.py
```

To reproduce the main symbolic equation discovery experiment:

```bash
python scripts/run_experiment.py
```

The discovered expression, prediction metrics, complexity measures, and optimisation statistics are saved in the `results/` directory.

---

## Experimental Protocol

The experiments are organised around several research questions.

### E1 — Baseline Equation Discovery

The first experiment evaluates whether the proposed symbolic neuron can recover known mathematical relationships from data.

The evaluation considers:

* predictive error;
* symbolic complexity;
* equation recovery;
* parameter estimation accuracy;
* computational cost.

---

### E2 — Robustness to Noise

The robustness experiments evaluate the behaviour of symbolic equation discovery under increasing levels of observational noise.

For a clean target

$$
y=f(x),
$$

the observed response is generated as

$$
y_{\sigma}=f(x)+\epsilon,
$$

where \(\epsilon\) represents controlled observational noise.

Multiple noise levels are evaluated to determine whether the discovered symbolic structure remains stable.

---

### E3 — Structural Uncertainty

This experiment investigates whether multiple symbolic structures can provide similar predictive performance.

The analysis therefore considers not only the best-fitting equation but also:

* alternative structures;
* complexity;
* stability across runs;
* parameter variability;
* structural agreement.

This is important because scientific equation discovery should not equate a small prediction error with certainty about the underlying mathematical structure.

---

### E4 — Physical Constraints

Where applicable, physical or domain-specific constraints are incorporated into the discovery process.

Examples include:

* dimensional consistency;
* parameter bounds;
* positivity constraints;
* conservation relationships;
* known functional dependencies.

The objective can therefore be extended to

$$
\mathcal{J}(S,\theta)
=
\mathcal{L}_{data}
+
\lambda\Omega(S)
+
\gamma\mathcal{C}(S,\theta),
$$

where \(\mathcal{C}\) penalises violations of the specified constraints.

---

### E5 — Shared Symbolic Library

The framework supports a common library of symbolic primitives:

$$
\mathcal{L}
=
\{
+,-,\times,\div,
\sin,\cos,\exp,\log,\sqrt{\cdot},\ldots
\}.
$$

The library can be fixed or extended toward a learnable primitive library.

This provides a basis for future multi-neuron symbolic architectures in which different symbolic neurons share the same primitive vocabulary.

---

### E6 — Implicit Equation Discovery

In addition to explicit relationships of the form

$$
y=f(x),
$$

the framework can investigate implicit relationships

$$
F(x_1,\ldots,x_n)=0.
$$

This is particularly relevant for scientific systems where isolating a single dependent variable may be undesirable or impossible.

---

## Baselines

The experimental evaluation compares the proposed approach with representative symbolic and neural-symbolic methods, including, where applicable:

* **PySR**
* **Equation Learner (EQL)**
* **Deep Symbolic Regression (DSR)**
* **Kolmogorov-Arnold Networks (KAN)**
* Other symbolic regression baselines

The baseline implementations and versions should be recorded explicitly to ensure reproducibility.

For each method, the repository records, where available:

* prediction error;
* equation recovery;
* symbolic complexity;
* robustness to noise;
* computational cost;
* structural stability.

---

## Evaluation Metrics

The experiments use complementary predictive, structural, and interpretability metrics.

### Predictive Accuracy

Typical metrics include:

$$
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

and

$$
\mathrm{RMSE}
=
\sqrt{\mathrm{MSE}}.
$$

Additional metrics may include MAE and \(R^2\).

### Symbolic Complexity

The complexity of a discovered expression can be measured using:

* number of operators;
* expression-tree size;
* number of constants;
* expression depth;
* weighted symbolic complexity.

### Equation Recovery

When the ground-truth equation is known, structural recovery is evaluated by comparing the discovered expression with the target expression after appropriate symbolic simplification or equivalence checking.

### Robustness

Robustness is evaluated by measuring the degradation of predictive and structural performance as the noise level increases.

### Stability

Multiple independent runs are used, where appropriate, to evaluate the stability of:

* discovered structures;
* parameters;
* prediction performance.

---

## Reproducibility

All experiments should specify:

* random seed;
* dataset;
* train/validation/test split;
* noise level;
* primitive library;
* search-space configuration;
* optimisation method;
* stopping criteria;
* complexity penalty;
* number of independent runs.

For example:

```python
SEED = 42
```

Results should be saved together with the corresponding experimental configuration.

---

## Generating the Paper Figures

After running the experiments, figures can be generated using:

```bash
python scripts/generate_figures.py
```

Generated figures are stored in:

```text
results/figures/
```

These figures correspond to the plots and tables reported in the manuscript.

---

## Reproducing the Main Results

A complete reproduction can be performed using:

```bash
python scripts/run_all_experiments.py
```

The pipeline performs the following steps:

```text
Dataset generation
        ↓
Data preprocessing
        ↓
Symbolic structure search
        ↓
Parameter optimisation
        ↓
Model selection
        ↓
Robustness evaluation
        ↓
Baseline comparison
        ↓
Statistical analysis
        ↓
Figures and tables
```

If computational resources are limited, individual experiments can be executed separately.

---

## Random Seeds

To improve reproducibility, experiments use fixed random seeds whenever stochastic procedures are involved.

The seed is explicitly stored in the experimental configuration and results metadata.

---

## Hardware and Software

The experiments were designed to run on standard research workstations.

Recommended environment:

```text
Python >= 3.10
NumPy
SciPy
SymPy
Pandas
Scikit-learn
Matplotlib
```

Additional dependencies required by individual baselines are documented separately.

GPU acceleration is optional unless explicitly required by a particular baseline.

---

## Results

The repository contains the experimental outputs used to generate the results reported in the paper.

Results are organised into:

```text
results/
├── raw/
├── processed/
└── figures/
```

Raw experimental outputs should not be manually modified. Processed results and publication figures are generated by the analysis scripts.

---

## Citation

If you use this code or build upon this work, please cite the associated paper:

```bibtex
@article{symbolic_neurons_2026,
  title   = {Symbolic Neurons for Robust Scientific Equation Discovery},
  author  = {AUTHOR NAME},
  journal = {JOURNAL NAME},
  year    = {2026},
  doi     = {DOI}
}
```

The BibTeX entry will be updated after publication.

---

## Related Work

The project builds upon research in:

* Symbolic Regression
* Scientific Machine Learning
* Automated Scientific Discovery
* Interpretable Machine Learning
* Neural-Symbolic Learning
* Equation Learning
* Physics-Informed Machine Learning
* Evolutionary Computation

Representative methods include PySR, EQL, Deep Symbolic Regression, and related symbolic discovery approaches.

---

## License

This repository is released under the license specified in `LICENSE`. 

Unless otherwise stated, the code is intended for research and academic use.

---

## Contact

For questions, suggestions, or collaboration inquiries, please open a GitHub Issue or contact:

**Author:** AUTHOR NAME
**Institution:** INSTITUTION NAME
**Email:** EMAIL ADDRESS

---

## Acknowledgements

This work was developed as part of research on interpretable artificial intelligence, symbolic learning, and scientific equation discovery.

We thank the authors and developers of the open-source projects used as dependencies and baselines.

---

## Status

🚧 **Research code — under active development**

The repository is being prepared for the reproducibility of the experiments reported in the associated manuscript. Code, datasets, configurations, and experimental results may be updated as the manuscript evolves.
