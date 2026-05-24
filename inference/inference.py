"""
Baby Cry Classifier — Inference Module
======================================
Two-stage cascade:
  Stage 1: Binary detection (baby_cry vs not_baby_cry)
  Stage 2: 4-class classification (scared/needs/physical_pain/burping)

Usage:
    from inference import BabyCryClassifier

    clf = BabyCryClassifier(
        stage1_ckpt='models/stage1_binary.ckpt',
        stage2_ckpts=['models/stage2_4class_epoch6.ckpt',
                      'models/stage2_4class_epoch7.ckpt'],
        config_dir='config/',
        device='cuda',  # or 'cpu'
    )
    result = clf.predict('audio.wav')
    print(result)
    # {
    #   "audio_duration_s": 3.2,
    #   "stage1_prediction": "baby_cry",
    #   "stage1_confidence": 0.94,
    #   "stage1_all_probs": {"baby_cry": 0.94, "not_baby_cry": 0.06},
    #   "stage2_prediction": "needs",
    #   "stage2_confidence": 0.71,
    #   "stage2_all_probs": {"burping": 0.05, "needs": 0.71, ...}
    # }
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
import pytorch_lightning as pl
from transformers import HubertModel
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score,
)

# ======= Constants =======
HUBERT_MODEL  = "facebook/hubert-base-ls960"
SAMPLE_RATE   = 16000
CHUNK_SAMPLES = 80000   # 5 seconds


# ======= Model class (exact match to training-time signature) =======
class HubertClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_weights=None,
                 hubert_model=HUBERT_MODEL, head_hidden_dim=256, dropout=0.1,
                 lr_head=1e-3, lr_backbone=1e-5, weight_decay=1e-4,
                 freeze_feature_extractor=True, freeze_n_transformer_layers=0,
                 warmup_steps=500, total_steps=None, label_names=None,
                 label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.label_names = label_names
        self._label_smoothing = label_smoothing
        self.hubert = HubertModel.from_pretrained(hubert_model)
        h = self.hubert.config.hidden_size
        if freeze_feature_extractor:
            self.hubert.feature_extractor._freeze_parameters()
        self.head = nn.Sequential(
            nn.Linear(h, head_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden_dim, num_classes),
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        # Dummy metrics so checkpoint loads cleanly
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.train_f1  = MulticlassF1Score (num_classes=num_classes, average="macro")
        self.val_f1    = MulticlassF1Score (num_classes=num_classes, average="macro")
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)

    def _masked_mean_pool(self, h, m):
        mm = m.unsqueeze(-1).float()
        return (h * mm).sum(1) / mm.sum(1).clamp(min=1.0)

    def _downsample_mask(self, am, target_len):
        m = F.adaptive_max_pool1d(am.float().unsqueeze(1), target_len).squeeze(1)
        return (m > 0.5).long()

    def forward(self, input_values, attention_mask):
        out = self.hubert(input_values=input_values, attention_mask=attention_mask, return_dict=True)
        h = out.last_hidden_state
        tm = self._downsample_mask(attention_mask, h.shape[1])
        return self.head(self._masked_mean_pool(h, tm))


# ======= Audio preprocessing =======
def preprocess_audio(audio_input, sample_rate=SAMPLE_RATE,
                     chunk_samples=CHUNK_SAMPLES):
    """
    Accepts:
      - file path (str)
      - tuple (sample_rate, np.array)
      - np.array (assumes already at target SR)

    Returns: input_values [1, chunk_samples], attention_mask [1, chunk_samples], duration_seconds
    """
    if isinstance(audio_input, str) or isinstance(audio_input, Path):
        wav, sr = sf.read(str(audio_input), dtype="float32")
    elif isinstance(audio_input, tuple):
        sr, wav = audio_input
        wav = np.asarray(wav, dtype=np.float32)
        if wav.max() > 1.5 or wav.min() < -1.5:  # int16 input
            wav = wav / 32768.0
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input.astype(np.float32)
        sr = sample_rate
    else:
        raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)

    n = len(wav)
    if n >= chunk_samples:
        start = (n - chunk_samples) // 2
        wav = wav[start:start + chunk_samples]
        mask = np.ones(chunk_samples, dtype=np.int64)
    else:
        pad = chunk_samples - n
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        mask = np.concatenate([np.ones(n, dtype=np.int64),
                               np.zeros(pad, dtype=np.int64)])

    # Zero-mean / unit-variance (HuBERT normalization)
    wav = wav - wav.mean()
    std = wav.std()
    if std > 1e-6:
        wav = wav / std

    return wav.astype(np.float32), mask, n / sample_rate


# ======= Main classifier class =======
class BabyCryClassifier:
    def __init__(self, stage1_ckpt, stage2_ckpts, config_dir, device="cuda"):
        """
        Args:
            stage1_ckpt: path to Stage 1 (.ckpt)
            stage2_ckpts: list of paths to Stage 2 ensemble (.ckpt)
            config_dir: folder containing label encoder JSONs
            device: 'cuda' or 'cpu'
        """
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        # Load configs
        config_dir = Path(config_dir)
        with open(config_dir / "stage1_label_encoder.json") as f:
            self.s1_enc = json.load(f)
        with open(config_dir / "stage2_label_encoder.json") as f:
            self.s2_enc = json.load(f)

        self.s1_labels = self.s1_enc["labels"]
        self.s2_labels = self.s2_enc["labels"]

        # Class weights are needed for ckpt load (any tensor of right shape)
        s1_w = torch.ones(len(self.s1_labels))
        s2_w = torch.ones(len(self.s2_labels))

        # Load Stage 1
        print(f"[Loading] Stage 1: {stage1_ckpt}")
        self.stage1 = HubertClassifier.load_from_checkpoint(
            str(stage1_ckpt), class_weights=s1_w, label_names=self.s1_labels,
            map_location=self.device,
        )
        self.stage1.eval().to(self.device)

        # Load Stage 2 ensemble
        self.stage2_models = []
        for ckpt in stage2_ckpts:
            print(f"[Loading] Stage 2: {ckpt}")
            m = HubertClassifier.load_from_checkpoint(
                str(ckpt), class_weights=s2_w, label_names=self.s2_labels,
                map_location=self.device,
            )
            m.eval().to(self.device)
            self.stage2_models.append(m)

        print(f"[Ready] device={self.device}, "
              f"stage1=1 model, stage2={len(self.stage2_models)} models")

    @torch.no_grad()
    def predict(self, audio_input):
        """Run the full two-stage cascade. Returns dict (see module docstring)."""
        wav, mask, duration = preprocess_audio(audio_input)
        iv = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        am = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        # Stage 1
        logits1 = self.stage1(iv, am)
        probs1 = F.softmax(logits1, dim=-1).cpu().numpy()[0]
        s1_idx = int(np.argmax(probs1))
        s1_label = self.s1_labels[s1_idx]
        s1_conf = float(probs1[s1_idx])

        result = {
            "audio_duration_s" : round(duration, 2),
            "stage1_prediction": s1_label,
            "stage1_confidence": round(s1_conf, 4),
            "stage1_all_probs" : {self.s1_labels[i]: round(float(probs1[i]), 4)
                                   for i in range(len(self.s1_labels))},
            "stage2_prediction": None,
            "stage2_confidence": None,
            "stage2_all_probs" : None,
        }

        # Stage 2 (only if cry detected)
        if s1_label == "baby_cry":
            all_probs = []
            for m in self.stage2_models:
                logits2 = m(iv, am)
                all_probs.append(F.softmax(logits2, dim=-1).cpu().numpy()[0])
            avg = np.mean(all_probs, axis=0)
            s2_idx = int(np.argmax(avg))
            result["stage2_prediction"] = self.s2_labels[s2_idx]
            result["stage2_confidence"] = round(float(avg[s2_idx]), 4)
            result["stage2_all_probs"]  = {self.s2_labels[i]: round(float(avg[i]), 4)
                                            for i in range(len(self.s2_labels))}

        return result


# ======= Quick test if run directly =======
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        sys.exit(1)

    clf = BabyCryClassifier(
        stage1_ckpt="models/stage1_binary.ckpt",
        stage2_ckpts=["models/stage2_4class_epoch6.ckpt",
                      "models/stage2_4class_epoch7.ckpt"],
        config_dir="config/",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    result = clf.predict(sys.argv[1])
    print(json.dumps(result, indent=2))
