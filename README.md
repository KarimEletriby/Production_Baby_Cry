# Baby Cry Classifier — Backend Package v1.0

## What's in this package

- `models/` — 3 trained HuBERT checkpoints
  - `stage1_binary.ckpt` (Stage 1: cry detection)
  - `stage2_4class_epoch6.ckpt` (Stage 2: ensemble member 1)
  - `stage2_4class_epoch7.ckpt` (Stage 2: ensemble member 2)
- `config/` — Label encoders and training config
- `inference/inference.py` — Production inference module
- `docs/` — Test results and visualizations
- `requirements.txt` — Python dependencies

## Performance (on held-out test set)

| Model | Accuracy | F1-macro | Notes |
|---|---|---|---|
| Stage 1 (binary) | 99.42% | 0.994 | Excellent generalization |
| Stage 2 (4-class ensemble) | 84.88% | 0.774 | scared=0.98, needs=0.89, physical_pain=0.70, burping=0.53 |

## Quickstart

```bash
# 1. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 2. Test on a single audio file
python inference/inference.py path/to/audio.wav
```

## Programmatic usage

```python
from inference.inference import BabyCryClassifier

clf = BabyCryClassifier(
    stage1_ckpt='models/stage1_binary.ckpt',
    stage2_ckpts=['models/stage2_4class_epoch6.ckpt',
                  'models/stage2_4class_epoch7.ckpt'],
    config_dir='config/',
    device='cuda',  # or 'cpu'
)

result = clf.predict('test.wav')
# Or pass a (sample_rate, np.array) tuple, or a numpy array
print(result)
```

## Output format

```json
{
  "audio_duration_s": 3.2,
  "stage1_prediction": "baby_cry",
  "stage1_confidence": 0.94,
  "stage1_all_probs": {"baby_cry": 0.94, "not_baby_cry": 0.06},
  "stage2_prediction": "needs",
  "stage2_confidence": 0.71,
  "stage2_all_probs": {
    "burping": 0.05,
    "needs": 0.71,
    "physical_pain": 0.18,
    "scared": 0.06
  }
}
```

If `stage1_prediction == "not_baby_cry"`, `stage2_*` fields will be `null`.

## Class definitions

### Stage 1 (binary)
- `baby_cry` — A baby is crying
- `not_baby_cry` — Anything else (laugh, noise, silence, environmental sounds)

### Stage 2 (4-class)
- `scared` — Sudden startle / fear cry (sharp, high-pitched)
- `needs` — General fussing (hungry, tired, or uncomfortable)
- `physical_pain` — Pain-related cry (e.g., colic — rhythmic, intense)
- `burping` — Needs to expel air after feeding (short bursts)

## Hardware requirements

| Setup | RAM | GPU | Inference time |
|---|---|---|---|
| GPU (T4 / RTX 3060+) | 4 GB | 4 GB VRAM | ~150 ms |
| CPU only | 4 GB | - | ~2 s |

## Known limitations

1. **Source bias**: training data is heavily weighted toward one source (`baby_crying` dataset).
   Real-world recordings from different mics/environments may see ~10-20pp accuracy drop.
2. **`burping` class**: only 240 training samples — model recall is moderate.
3. **Model is for AI assistance only** — not a medical diagnostic tool.

## License

Each underlying training dataset retains its original license:
- ESC-50: CC BY-NC 4.0
- infant_cry_corpus (Donate-a-Cry): CC BY-NC 4.0
- baby_crying / baby_sounds (Kaggle): see source

The trained model is your own derivative work.
