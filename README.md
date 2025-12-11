# Marine Mammal Bioacoustic Classification

This repository contains a deep learning pipeline for **classifying marine mammals from underwater audio recordings**.  
The project explores multiple audio representations (raw waveform, Mel spectrograms, and wavelet scattering coefficients) and combines them using **ResNet backbones** and a **fusion MLP**.

The code is written in **PyTorch** and **torchaudio**, with additional feature extraction using **Kymatio’s 1D Scattering Transform**.

---

## Project Overview

- Input: `.wav` recordings of marine mammals, organised by species in subfolders.
- Preprocessing:
  - Resampling all audio to a common sampling rate (`target_sr = 47600`).
  - Centered padding / cutting to a fixed length (`cut_point = 8000` samples).
  - Class balancing (only classes with at least 50 samples are kept).
  - Duplicate recordings removed.
- Feature Extraction:
  - On-the-fly **Mel spectrograms** (`n_mels = 64`).
  - **Wavelet scattering coefficients** (orders 1 and 2) using `Scattering1D`.
- Models:
  - Three small **ResNet** models:
    - `model_mel`: operates on Mel spectrograms.
    - `model_wst_1`: operates on 1st-order scattering coefficients.
    - `model_wst_2`: operates on 2nd-order scattering coefficients.
  - A final **MLP fusion model** that combines logits from the scattering-based ResNets (and optionally from the Mel model).
- Training:
  - The dataset is repeated 4×, creating **pseudo-classes** (e.g. from 8 species → 32 “classes”) for a harder supervision signal.
  - During evaluation, predictions are also reduced back to the original number of species via modulo arithmetic.
- Logging:
  - Training and validation losses/accuracies are stored as `.npy` arrays.
  - Training curves are saved as `.png` plots.

---

## Dataset Structure

The dataset is expected under a root folder, for example:

```text
/home/username/data/Cetacean_Classification/_common-frecuent/
├── SpeciesA
│   ├── file001.wav
│   ├── file002.wav
│   └── ...
├── SpeciesB
│   ├── file010.wav
│   └── ...
└── ...
```

Each first-level folder name is treated as the class label.
In the script, set the file variable to your dataset root:

`file = "/path/to/Cetacean_Classification/_common-frecuent"`

Only .wav files are processed.

## Dependencies

Main Python dependencies:

- `python >= 3.9`
- `torch`
- `torchaudio`
- `numpy`
- `pandas`
- `scipy`
- `soundfile`
- `scikit-learn`
- `matplotlib`
- `kymatio`

You can install them with:

`pip install torch torchaudio numpy pandas scipy soundfile scikit-learn matplotlib kymatio`

(Adjust the torch/torchaudio install command according to your CUDA setup.)

How to Run

1. Clone the repository and move into it:
   
`git clone https://github.com/<your-username>/<your-repo>.git`

`cd <your-repo>`


2. Prepare the dataset
- Place your audio data according to the directory structure above.
- Update the file path in the script.


3. Run training
Assuming the main script is named `marine_mammal_classification.py`:

`python marine_mammal_classification.py`


The script will:
- Load and preprocess the audio.
- Build train/validation splits with stratification.
- Train:
  - A ResNet on Mel spectrograms.
  - Two ResNets on scattering coefficients (orders 1 and 2).
  - An MLP to fuse their logits.
- Print training and validation accuracies both for:
  - The expanded pseudo-class problem (e.g. 32 classes),
  - And the original species-level problem (e.g. 8 classes, via modulo mapping).
- Save training curves to *.npy and *.png files.

## Outputs
- `modelmel_...npy, modelws1_...npy, modelws2_...npy`: training logs for each ResNet.
- `S1+S2_...npy, MLP_S+Mel...npy`: training logs for the MLP fusion models.
- `resnet_mel_training.png`: example training curve visualization.
- Console logs with:
  - Epoch-wise training/validation accuracies.
  - Final species-level accuracy for different fusion strategies.
  - Optional lambda-search results for linear combination of probabilities.

## Data Augmentation (Optional)

The script includes optional augmentation utilities:

- Waveform-level augmentations (augment_waveform):
  - Additive Gaussian noise.
  - Random gain jitter.
  - Small circular time-shifts.
- Spectrogram-level augmentations (augment_spectrogram):
  - Frequency masking.
  - Time masking (SpecAugment-style).

These are currently commented out in the dataset classes (AugmentedTensorDataset, MelSpecDataset).
You can enable them by uncommenting the relevant lines to regularise the models and improve generalisation.



