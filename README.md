# Generative Neural Network for Geant4 Simulations (Experimental Prototype)

> ⚠️ **Project status: Early prototype / Work in progress.**
>
> The current model is **not yet able to accurately reproduce** the data used during training. This repository documents an early-stage experiment, not a tool ready for scientific use.

## Idea and Objectives

Monte Carlo simulations with [Geant4](https://geant4.web.cern.ch/) are highly accurate but computationally expensive. This project explores the following idea:

> **Train a generative neural network on existing simulation data to rapidly produce new synthetic particle tracks that are *statistically similar* to those observed during training.**

The project has two main objectives:

1. **Speed up simulations**: generating an event with the neural network should require significantly less computational time than simulating it with Geant4.
2. **Increase statistics**: starting from a limited simulated dataset, generate many additional synthetic events with compatible distributions.

A generative model **learns the distribution of its training data and samples new examples from it**. Consequently:

- ❌ **It cannot generate data from completely unseen physical scenarios**, such as different energies, geometries, particles, or physical processes not represented in the training set.
- ❌ **It cannot totally replace Geant4**.
- ✅ It can generate **new synthetic samples** consistent with the learned distribution. These are plausible recombinations of patterns already observed by the model, not new physical information.
- ⚠️ The quality of the generated samples entirely depends on how accurately the model has learned the underlying distribution. At present, this quality **is not yet sufficient**.

## How It Works

Each Geant4 event is treated as a **sequence of steps**. The network is autoregressive: given step *t*, it predicts step *t+1*. During generation, it starts from an initial step sampled from the empirical distributions of the original data and generates the track one step at a time.

**Current modeling choices:**

- Only **primary particles** (`ParentID == 0`) are considered, as a simplification for the first model.
- **Input**: initial energy of the primary particle and the sequence of steps.
- **Step features**: `Edep`, `X`, `Y`, `Z`, `Step` (continuous) and `VolumeName`, `MotherVolumeName` (categorical).
- **Continuous variables**: the network predicts the mean and log-variance of a Gaussian distribution, trained using **Gaussian Negative Log-Likelihood (NLL)** instead of MSE, in order to capture variability rather than simply averaging predictions.
- **Categorical variables**: volume names are predicted as classes using **Cross-Entropy** and sampled stochastically during generation (with temperature scaling), avoiding arbitrary rounding operations.
- **Physical constraints during generation**: constraints are applied to quantities such as the step length (`Step`) and energy conservation (remaining energy), using thresholds sampled from the original data.

### Architecture

A hybrid **GRU + Transformer** model with adaptive chunking (`model_architecture.py`).

| Component | Role |
|---|---|
| **GRU** | Maintains the hidden state across steps, efficiently handling long sequences |
| **Transformer encoder** | Uses self-attention to capture long-range dependencies within the sequence |
| **Chunking** | Splits each event into blocks of steps (`max_chunk_len`). This is analogous to minibatching, but along the steps of an event rather than across events |
| **Integrated generation** | Generation is implemented within the model, ensuring that the GRU hidden state remains consistent across generated steps |

## Repository Structure

The repository contains **two variants** of the same experiment, differing only in their data normalization approach:

```text
.
├── file_norm/        # Variant with MinMax normalization [0, 1] of continuous features
└── file_not_norm/    # Variant without normalization (physical units)
```

Each folder contains:

| File | Description |
|---|---|
| `TrainingNN.ipynb` | Data loading, encoding, normalization, padding/masking, model training, and weight saving |
| `GenerativeNN.ipynb` | Loading the trained model, generating synthetic events, inverse normalization, and saving the output to `.txt` files |
| `analysis.ipynb` | Comparison between simulated and generated data (total deposited energy, number of steps, etc.) |
| `model_architecture.py` | Definition of `GRUTransformerGenerator` (model, supporting loss functions, and generation routines) |
| `first_step_distributions.py` | `FirstStepSampler`: samples the initial step and remaining-energy thresholds from the original data distributions |
| `best_model_weights.pt` | Model weights corresponding to the best validation loss |
| `encoders_scalers.pkl` | `LabelEncoder` and scalers used during training (required to decode the model output) |

## Data

The training data are **not included** in this repository. They consist of tab-separated text files containing Geant4 simulation output, with the following columns:

```text
EventNumber, ParentID, TrackID, ParticleName, ParticleEn(MeV), Edep(MeV),
X(mm), Y(mm), Z(mm), Step(mm), Track(mm), VolumeName, MotherVolumeName,
CenterX(mm), CenterY(mm), CenterZ(mm)
```

The notebooks currently use the file:

`DatiPerNN_BigMixture_1500keV_PulseTube_1e4EInt.txt`

This dataset contains primary electrons with an initial energy of 1.5 MeV and 10⁴ simulated events.

To reproduce the experiment, provide a dataset with the same format and update the corresponding file path in the notebooks.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
pip install torch numpy pandas scikit-learn plotly tqdm jupyter
```

The code automatically uses the GPU if one is available; otherwise, it runs on the CPU.

## Usage

1. **Training**: Open `TrainingNN.ipynb` in the selected folder (`file_norm` or `file_not_norm`) and run all cells. The training process saves `best_model_weights.pt` and `encoders_scalers.pkl`.
2. **Generation**: Open `GenerativeNN.ipynb` and run the cells. Synthetic events are saved to a `*_generated.txt` file using the same format as the original Geant4 data. Main parameters include `n_particles`, `E_init`, `temperature`, and `noise_scale`.
3. **Analysis**: Open `analysis.ipynb` to compare histograms of simulated and generated data.

## Known Limitations

As this is an experimental prototype, the project currently has several limitations:

- **Insufficient accuracy**: generated distributions (e.g. total deposited energy and number of steps) do not yet match the simulated ones.
- **Primary tracks only**: secondary particles and particle showers are not modeled.
- **Single energy and particle type**: the model is trained on fixed-energy electrons and does not generalize to other regimes.
- **Some parameters are manually assigned** during generation (`ParentID`, `TrackID`, `ParticleName`, and initial energy) rather than learned by the model.
- **Small, non-optimized model**: hyperparameters have not yet been systematically explored.
- **Validation strategy**: the 80/20 train/test split uses the test set for early stopping as well, without an independent validation set.
- **No quantitative evaluation metrics**: comparisons are currently based only on visual inspection of histograms.

## Possible Future Developments

- Quantitative metrics for comparing distributions (KS test, Wasserstein distance, etc.).
- Independent train/validation/test splits.
- Extension to multiple energies and particle types (energy-conditioned generation).
- Modeling of secondary particles.
- Hyperparameter tuning and exploration of larger architectures.
- Measurement of the speedup achieved compared to Geant4 simulations.

## Notes

- `encoders_scalers.pkl` is a *pickle* file. Only load it if you trust its source, as pickle files can execute arbitrary code.
- The notebooks contain saved outputs and file paths specific to the system on which they were originally executed.
