# RealSeeker: Fake vs Real Photo Detection (32x32)

A PyTorch pipeline that classifies small 32x32 RGB images as **REAL** or **FAKE** (AI-generated). It compares two architectures on the same data:

1. **Baseline CNN**: a compact 3-block convolutional network.
2. **Dual-branch model**: the same CNN plus a frequency-domain branch (2D FFT) whose features are fused with the CNN features.

Both models are trained, evaluated and compared on a 20,000-image held-out test set.

## Live demo

- Web demo: https://fakedetection-psi.vercel.app
- Model host (Hugging Face Space): https://huggingface.co/spaces/thanhmausac/RealSeeker

The web page is a static `index.html` on Vercel that sends the uploaded image to the Gradio app (`app.py`) running on the Hugging Face Space, which returns the REAL/FAKE probabilities. The model only sees a 32x32 version of the upload, so treat the output as a demo (see Limitations).

## Results

Evaluated on the test set (20,000 images, 10,000 per class).

| Metric | Baseline CNN | CNN + FFT branch |
|---|---|---|
| Parameters | 288,162 | 295,586 |
| Accuracy | 97.61% | **97.76%** |
| ROC-AUC | 0.9973 | **0.9975** |
| Precision (FAKE) | 97.34% | **97.72%** |
| Recall (FAKE) | **97.91%** | 97.80% |
| FAKE missed (predicted REAL) | **209** | 220 |
| REAL wrongly flagged as FAKE | 268 | **228** |

**Takeaway:** the two models are practically tied. The FFT branch gives slightly fewer false alarms on real photos, while the baseline misses slightly fewer fakes. These differences are small enough to fall within run-to-run noise (single training run each), so the frequency branch does not clearly justify its extra complexity on this dataset. Discriminative signal appears to be already strong in pixel space.

## Dataset

The data is the CIFAKE dataset (real CIFAR-10 images vs AI-generated counterparts), arranged for `torchvision.datasets.ImageFolder`:

```
Fake_vs_Real_Photo/
├── train/
│   ├── FAKE/   (50,000 images)
│   └── REAL/   (50,000 images)
└── test/
    ├── FAKE/   (10,000 images)
    └── REAL/   (10,000 images)
```

- Image size: 32x32 RGB
- Classes are perfectly balanced, so no resampling is needed (the loader supports `WeightedRandomSampler` for imbalanced data).
- Normalization uses mean/std computed from the training set (see `compute.py`):
  - mean = `[0.4720, 0.4629, 0.4178]`
  - std = `[0.2376, 0.2374, 0.2660]`

## Model architectures

### Baseline CNN (`FakeDetectorCNN`)

```
Input (3, 32, 32)
  → ConvBlock(3→32)    [Conv3x3-BN-ReLU ×2, MaxPool]  → 16x16
  → ConvBlock(32→64)   [Conv3x3-BN-ReLU ×2, MaxPool]  → 8x8
  → ConvBlock(64→128)  [Conv3x3-BN-ReLU ×2, MaxPool]  → 4x4
  → Global Average Pooling → 128
  → Dropout(0.3) → Linear(128→2)
```

### Dual-branch (`FakeDetectorDualBranch`)

- **Spatial branch:** identical to the baseline feature extractor (128-d vector).
- **Frequency branch:** `fft2` → log-magnitude → `fftshift` → two small conv layers (3→16→32) → linear projection (64-d vector). Aims to capture periodic upsampling artifacts left by generative models.
- **Fusion:** concatenate (128 + 64 = 192-d) → Dropout(0.3) → Linear(192→2).

## Project structure

```
.
├── transforms.py   # Shared preprocessing (train/eval); MEAN/STD live here
├── compute.py      # Computes dataset mean/std once
├── dataset.py      # DataLoaders (ImageFolder, optional class balancing)
├── model.py        # Baseline CNN and dual-branch model
├── train.py        # Training loop, cosine LR schedule, early stopping, checkpointing
├── evaluate.py     # Report, confusion matrix, ROC-AUC, recall-FAKE threshold table
├── tune.py         # Optuna hyperparameter search (optional)
├── analysis.py     # Worst-error gallery and Grad-CAM
├── app.py          # Gradio app served on the Hugging Face Space
├── index.html      # Static web UI (Vercel) that calls the Space
├── requirements.txt  # Only what the Space needs (training deps are in Setup below)
└── README.md
```

`transforms.py` is imported by both training and evaluation code so preprocessing can never drift between them.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn matplotlib tqdm optuna
```

Verify the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

```powershell
# 1. Compute normalization stats (paste the output into transforms.py)
python compute.py

# 2. Sanity checks
python dataset.py
python model.py

# 3. Train
python train.py --data-root G:/ML/files/Fake_vs_Real_Photo --model baseline --epochs 30 --out best_baseline.pt
python train.py --data-root G:/ML/files/Fake_vs_Real_Photo --model dual_branch --epochs 30 --out best_dual.pt

# 4. Evaluate
python evaluate.py --checkpoint best_baseline.pt --data-root G:/ML/files/Fake_vs_Real_Photo
python evaluate.py --checkpoint best_dual.pt --data-root G:/ML/files/Fake_vs_Real_Photo

# 5. Error analysis + Grad-CAM
python analysis.py --checkpoint best_baseline.pt --data-root G:/ML/files/Fake_vs_Real_Photo

# 6. (Optional) Hyperparameter search
python tune.py --data-root G:/ML/files/Fake_vs_Real_Photo --n-trials 20
```

## Run the web demo locally

```powershell
pip install gradio torch pillow numpy
python app.py
```

Open the printed local URL, upload an image and pick a model. `app.py` loads `best_baseline.pt` and `best_dual.pt` from the repo root and applies the same preprocessing as `transforms.py` (resize to 32x32, normalize with the training mean/std).

## Training setup

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr 1e-3, weight decay 1e-4) |
| Scheduler | Cosine annealing over 30 epochs |
| Loss | Cross-entropy |
| Batch size | 128 |
| Augmentation | Horizontal flip, small random crop (reflect padding), light color jitter |
| Regularization | BatchNorm, Dropout 0.3, early stopping (patience 7) |
| Hardware | NVIDIA RTX 2060 Super, ~60 s/epoch |

Augmentation is deliberately mild: strong blur or heavy JPEG-style corruption could erase the high-frequency artifacts the models need to detect.

## Threshold selection

`evaluate.py` prints, besides the default 0.5 threshold and the Youden's J optimum, a **recall-FAKE trade-off table**: for each threshold on P(FAKE) it shows recall, precision and how many real images get flagged. Lowering the threshold catches more fakes at the cost of more false alarms, so the operating point should be chosen from the application's tolerance for each error type.

## Limitations

- **Checkpoint selection uses the test set.** `train.py` keeps the epoch with the lowest test loss, so the reported test numbers are slightly optimistic. For a stricter evaluation, split a separate validation set from the training data and keep the test set untouched until the end.
- **Single run per model.** No seeds sweep or confidence intervals, so the baseline vs dual-branch gap should not be over-interpreted.
- **In-distribution only.** Results are on the same generator distribution as the training data. Performance on fakes from unseen generators is untested and likely lower.
- **Tiny inputs.** At 32x32, much of the high-frequency information that detectors of larger images rely on is already lost.

## Possible next steps

- Hold out a proper validation split and re-report test metrics.
- Repeat runs with multiple seeds and report mean ± std.
- Run `analysis.py` (worst errors, Grad-CAM) to check whether the models rely on meaningful cues or shortcuts.
- Test cross-generator generalization with fakes from a different source.
- Run the Optuna search and compare tuned configurations.
