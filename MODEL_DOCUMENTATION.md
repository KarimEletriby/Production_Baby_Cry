# BabyCry AI: Technical Architecture & Model Documentation

This document provides a comprehensive overview of the BabyCry AI classification system, including how audio is processed, validated, and classified, along with the model's architecture, classes, and a roadmap for future improvements.

---

## 1. Audio Processing & Validation Pipeline
Before the AI model processes any audio, the system runs a rigorous, multi-step validation pipeline to ensure the input is valid, safe, and actually contains meaningful acoustic data. This prevents the model from wasting compute resources on garbage data (like silence or pure noise).

### A. System & Format Validation
- **Size Constraints:** Rejects files larger than 25 MB or smaller than 10 KB.
- **Duration Constraints:** Audio must be between **0.5 seconds** and **30 seconds**.
- **Format Support:** Accepts standard audio formats (WAV, MP3, OGG, FLAC, M4A).
- **Corrupt File Check:** Uses `soundfile` to ensure the audio header and data are intact and readable.

### B. Acoustic Sanity Checks (Pre-Inference)
The system analyzes the raw audio waveform mathematically before sending it to the neural network. It rejects audio that falls into the following categories:
1. **Too Quiet (`TOO_QUIET`):** The RMS (Root Mean Square) energy is below the minimum threshold (e.g., pure silence or a muted microphone).
2. **Mostly Silence (`MOSTLY_SILENCE`):** The audio has too many silent frames compared to active acoustic frames.
3. **Pure Noise (`TOO_NOISY`):** High Zero-Crossing Rate (ZCR) indicating white noise, static, or heavy wind, with no harmonic content.
4. **Constant Tones (`CONSTANT_TONE` / `TOO_SMOOTH`):** Very low spectral flux indicating a synthetic beep, dial tone, or continuous alarm rather than natural human vocalization.
5. **Wrong Frequency Profile (`WRONG_FREQUENCY_PROFILE`):** The energy is heavily concentrated outside the typical human vocal/cry band (250 Hz - 3000 Hz).

---

## 2. Model Architecture
The AI engine uses a **Two-Stage Cascade Architecture** based on the `HubertClassifier` (a transformer-based acoustic model). This ensures high precision; the system first verifies if the sound is a baby cry before attempting to diagnose the reason.

### Stage 1: Primary Detection (Binary Classification)
- **Goal:** Determine if the audio contains a baby crying.
- **Model:** A single `HubertClassifier` checkpoint (`stage1_binary.ckpt`).
- **Accuracy:** ~99.4% F1-score.
- **Classes:**
  1. `baby_cry`
  2. `not_baby_cry` (If detected, the pipeline stops here).

### Stage 2: Diagnostic Classification (Multi-class)
- **Goal:** If Stage 1 detects a `baby_cry`, Stage 2 analyzes the emotional/physical state of the infant.
- **Model:** An ensemble of two `HubertClassifier` checkpoints (`epoch6` and `epoch7`). The predictions are averaged to improve stability and reduce variance.
- **Classes:**
  1. **`needs`** (Hunger / Discomfort / General Needs) - High accuracy (~88%)
  2. **`scared`** (Fear / Startled) - Very high accuracy (~97%)
  3. **`physical_pain`** (Colic / Pain) - Moderate accuracy (~69%)
  4. **`burping`** (Needs to burp) - Low accuracy (~54%)

*Note: Due to the lower accuracy of the `burping` class, the API automatically appends a specific warning to the user, advising them to rely on contextual clues (e.g., did the baby just feed?).*

---

## 3. Production Hardening & Safety Features
- **Inference Timeout:** Model execution is strictly bound by a 30-second timeout (`asyncio.wait_for`) to prevent server deadlocks.
- **Dynamic Confidence Warnings:** The API returns human-readable advice based on the model's confidence scores. Predictions with <55% confidence trigger a "Not entirely sure" warning.
- **Medical Disclaimer:** Every API response explicitly includes a medical disclaimer in Arabic, emphasizing that the AI is an assistive tool, not a medical diagnostic device.

---

## 4. Roadmap & Future Improvements

To transition this MVP from a Beta pilot to a robust commercial product, the following technical improvements are recommended:

### A. Data & Bias Mitigation
- **Source/Microphone Bias Correction:** The current model was trained heavily on specific datasets (`baby_crying`). Performance drops when exposed to different microphone qualities (e.g., cheap Android mics vs. iPhones) or noisy environments (echo, background TV). We need to augment the training data with simulated background noise (Noise Injection) and Room Impulse Responses (RIR).
- **Burping Class Enhancement:** Collect more targeted, high-quality samples of "burping" cries to improve the current 0.54 F1-score.

### B. System Optimization
- **Model Quantization (ONNX / TensorRT):** The current models are roughly ~3.3 GB in size and run on PyTorch. Converting the models to `ONNX` and applying `INT8` or `FP16` quantization will drastically reduce memory usage (RAM/VRAM) and speed up inference times, especially on CPU-only servers.
- **Streaming Audio Support:** Currently, the API waits for the entire file to upload before processing. Implementing WebSockets for streaming audio chunks could provide real-time, instantaneous feedback to the user.

### C. Continuous Learning
- **User Feedback Loop:** The mobile application should include a feedback mechanism (e.g., "Was this correct? Yes/No"). This telemetry data should be sent back to our servers, annotated, and used to continuously retrain and finetune the models in future epochs.
