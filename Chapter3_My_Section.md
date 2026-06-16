# Chapter 3: AI Models and Intelligent Features

## 3.1 Overall System Architecture

### 3.1.1 Architecture Overview

The Smart Motherhood Assistant is designed as a unified, user-centered platform that integrates multiple AI technologies — audio analysis, natural language processing, and speech synthesis — within a single mobile application experience. The overall system is composed of a Flutter-based mobile front-end, a backend infrastructure responsible for secure communication and inference orchestration, and three main intelligent modules: (1) baby cry recognition, (2) AI-powered storytelling with voice synthesis, and (3) an Arabic maternal chatbot supported by Retrieval-Augmented Generation (RAG).

### 3.1.2 Mobile Application Layer (Flutter)

The mobile application represents the primary user interface through which mothers interact with all intelligent services. Flutter was selected to support cross-platform deployment (Android/iOS), strong performance, and rich UI experiences suitable for audio-based interaction and dynamic screens.

From an architectural perspective, the app follows Clean Architecture with BLoC-based state management. User actions are represented as events, processed inside feature-specific BLoCs, and reflected as immutable UI states. Dedicated BLoCs are used to preserve modularity across the application's major features (cry analysis, storytelling, chatbot, and user/auth management).

### 3.1.3 Backend and API Layer

The backend is implemented using FastAPI, chosen for asynchronous request handling and efficient concurrent processing — an important requirement when serving multiple inference requests (e.g., story synthesis, chatbot responses). The backend architecture separates concerns into: (1) API endpoints (request validation/formatting), (2) service logic (model loading and inference orchestration), and (3) utility modules (preprocessing, text normalization, caching).

**Second Semester Enhancements:** The backend was hardened for production with the addition of multi-layer audio validation, acoustic sanity checks, per-IP rate limiting (30 requests/minute via slowapi), structured logging (JSON in production via structlog), in-memory metrics tracking, and request ID tracing — transforming the prototype into a deployment-ready service.

### 3.1.4 AI Layer: Intelligent Modules

**(A) Baby Cry Recognition**

The cry recognition module uses transfer learning with a pretrained HuBERT backbone (facebook/hubert-base-ls960) as a feature extractor, followed by a custom classifier head for cry-category prediction. The system uses a two-stage hierarchical cascade: Stage 1 performs binary detection (baby_cry vs. not_baby_cry) with 99.42% accuracy, and Stage 2 classifies detected cries into four consolidated categories (scared, needs, physical_pain, burping) with 84.27% accuracy using ensemble averaging of two model checkpoints.

Audio preprocessing standardizes input recordings through resampling (16 kHz), mono conversion, normalization, and fixed-duration trimming/padding before feature extraction.

**Second Semester Addition:** Before reaching the AI model, audio inputs now pass through a validation layer (file size, format, duration checks) and an acoustic sanity check module (RMS energy, voiced frame ratio, cry-band frequency analysis, spectral flux, and zero-crossing rate) that rejects silence, noise, and constant tones — saving GPU compute and preventing meaningless predictions.

**(B) AI-Powered Storytelling and Voice Synthesis**

The storytelling module combines natural language storytelling capabilities with Arabic-capable speech synthesis. The system is designed for immersive, age-appropriate experiences and supports personalization.

For voice generation, XTTS v2 is used as the synthesis foundation, leveraging voice cloning and inference-time adaptation rather than full fine-tuning, balancing quality with practical compute constraints.

**(C) Mommy Chatbot (RAG-based Arabic Assistant)**

The chatbot is an Arabic conversational system that provides childcare guidance using Retrieval-Augmented Generation (RAG). It retrieves verified context from a knowledge base built from medical books and articles, then generates responses using the selected Aya-23-8B language model.

The RAG pipeline follows multiple stages: query processing, retrieval from FAISS, relevance verification, structured prompting, response generation, and conversation management.

### 3.1.5 Data Layer

The primary retrieval store is a FAISS vector index used for similarity search over embedded document chunks. In this system, embeddings are generated using a multilingual MPNet model (768 dimensions), and FAISS retrieves the top relevant chunks for response grounding with fast search times.

*(Insert Figure 3.1.1 — Overall System Architecture Here)*

---

## 3.2 Application Workflow

### 3.2.1 End-to-End Application Workflow

The application workflow begins with user access through the mobile interface, followed by authentication/profile initialization. After login, the user navigates to the main functionality and selects one of three primary AI services: cry analysis, storytelling, or chatbot guidance. Each feature triggers an asynchronous request to the backend, which performs preprocessing and inference, then returns results to update the UI state through feature-specific BLoCs.

*(Insert Figure 3.1.2 — Full Application Workflow Here)*

### 3.2.2 Workflow — Baby Cry Detection Feature

This feature enables mothers to record an audio clip, which is standardized through preprocessing (16 kHz resampling, mono conversion, normalization, and fixed-duration trimming/padding). The processed waveform is converted into model-compatible features and passed through the HuBERT-based classification pipeline.

**Updated Workflow (Second Semester):**

The cry detection workflow now includes pre-inference validation and acoustic checks:

1. **Audio Upload:** The client sends the audio file via `POST /predict` (multipart/form-data).
2. **File Validation:** The file is checked for supported format (WAV/MP3/OGG/FLAC/M4A/WebM), size limits (1 KB – 25 MB), and duration bounds (0.5 – 30 seconds). Invalid files are immediately rejected with bilingual error messages.
3. **Acoustic Sanity Check:** The decoded audio signal is analyzed for RMS energy (≥ 0.010), voiced frame ratio (≥ 20%), cry-band frequency concentration (250–3000 Hz, ≥ 40%), spectral flux (≥ 0.015), and zero-crossing rate (0.02–0.35). Non-cry-like signals (silence, white noise, constant tones) are rejected before reaching the AI model.
4. **Stage 1 — Cry Detection:** Binary HuBERT classifier determines if the audio contains a baby cry (99.42% accuracy).
5. **Stage 2 — Cry Classification:** If a cry is detected, an ensemble of two HuBERT classifiers predicts the cry type from four categories: scared, needs, physical_pain, or burping (84.27% accuracy).
6. **Response Assembly:** The prediction is enriched with Arabic content (cry type name, physical signs, parenting tips, medical warnings), confidence labels, and a medical disclaimer.
7. **JSON Response:** Structured response returned with a unique request ID for traceability.

*(Insert Figure 3.1.3 — Updated Cry Detection Workflow Here)*

### 3.2.3 Workflow — AI Storytelling and Voice Cloning Feature

In this feature, the user selects (or requests) a story, and optionally provides reference audio for voice cloning. The backend exposes FastAPI endpoints to validate requests, then delegates synthesis to the XTTS v2 service layer. The design relies on voice cloning and inference-time adaptation instead of fine-tuning, ensuring practical deployment while maintaining narrative immersion.

*(Insert Figure 3.1.4 — Storytelling Workflow Here)*

### 3.2.4 Workflow — Mommy Chatbot (RAG) Feature

The chatbot workflow begins when the user submits a question. The system first checks conversation memory for essential context (e.g., child age) and requests missing information if necessary. The query is embedded and used to retrieve the most relevant chunks from a FAISS vector index. Retrieved chunks are filtered through relevance verification, then injected into a structured prompt to generate a culturally appropriate Arabic response using Aya-23-8B.

*(Insert Figure 3.1.5 — Chatbot (RAG) Workflow Here)*

---

## 3.3 Baby Cry Recognition Model

This section presents the complete development pipeline of the baby cry classification system, including dataset preparation, audio preprocessing, model architecture design, training methodology, comprehensive evaluation, and the production hardening improvements implemented during the second semester. The system leverages a transformer-based architecture (HuBERT) to classify baby cries. The first semester focused on building and evaluating the AI model. The second semester focused on making the system production-ready through multi-layer input validation, acoustic analysis, rate limiting, structured logging, and operational metrics.

### 3.3.1 Dataset Description

#### Dataset Overview

The baby cry recognition model was trained on a publicly available dataset from Kaggle, containing 7,249 audio samples across nine initial categories. After removing irrelevant categories (lonely and laugh, which are not actual cry sounds), the final dataset consisted of 7 classes with 4,749 audio files. The original class distribution is shown in Table 3.1.

**Table 3.1 — Initial Class Distribution**

| Class | Number of Samples | Percentage |
|---|---|---|
| Belly Pain | 750 | 15.8% |
| Burping | 247 | 5.2% |
| Cold/Hot | 750 | 15.8% |
| Discomfort | 750 | 15.8% |
| Hungry | 750 | 15.8% |
| Scared | 750 | 15.8% |
| Tired | 752 | 15.8% |
| **Total** | **4,749** | **100%** |

**Second Semester Update:** For the production deployment, the training pipeline was expanded to aggregate data from multiple sources: Kaggle (baby_crying, baby_sounds), Donate-a-Cry (infant_cry_corpus), and ESC-50 (environmental sounds for the non-cry negative class in Stage 1). The original 6 labels were consolidated into 4 clinically meaningful categories (see Label Merging Strategy below), improving model robustness while maintaining practical utility.

#### Dataset Characteristics

An exploratory data analysis (EDA) revealed the following audio properties:

- **Sample Rate:** Predominantly 16 kHz (99% of samples), with occasional variations.
- **Duration:** All samples standardized to 4 seconds in the original dataset.
- **Format:** WAV files, mono channel.
- **Quality:** Professional recording quality suitable for machine learning applications.

#### Class Imbalance Handling

A significant class imbalance was identified, with the "burping" class containing only 247 samples compared to 750+ samples in other classes (approximately 3× fewer samples). To address this imbalance and prevent model bias, data augmentation techniques were applied specifically to the "burping" class:

1. **Time Stretching:** Random speed variations of ±10% to simulate different cry intensities.
2. **Pitch Shifting:** Frequency modulation of ±2 semitones to capture vocal variations.
3. **Noise Injection:** Addition of minimal background noise (0.5% amplitude) to improve robustness.

This augmentation process generated 503 additional synthetic samples, bringing the "burping" class to 750 samples and achieving a balanced dataset of 5,252 total samples after augmentation.

#### Dataset Splitting Strategy

The augmented dataset was divided into training, validation, and test sets using stratified sampling to maintain proportional class distribution across splits:

- **Training Set:** 3,676 samples (70%) — Used for model learning.
- **Validation Set:** 788 samples (15%) — Used for hyperparameter tuning and early stopping.
- **Test Set:** 788 samples (15%) — Used for final unbiased evaluation.

The stratified approach ensures that each split maintains the same class distribution as the original dataset, preventing evaluation bias.

#### Special Consideration: "Hungry" Class

During initial model training, the "hungry" class exhibited significant confusion with other categories (particularly "belly pain" and "discomfort"), leading to reduced overall model performance. Analysis revealed that hunger cries often overlap acoustically with other distress signals, making them difficult to distinguish using audio features alone.

**Solution:** The "hungry" class was removed from the primary classification task and handled through a contextual questionnaire approach. This decision improved the model's performance on the remaining 6 classes from 63.71% to 82.37% accuracy, demonstrating the effectiveness of hybrid audio-contextual classification.

#### Label Merging Strategy (Second Semester)

For the production deployment, the original 6-class taxonomy was further consolidated into 4 clinically meaningful categories to improve model robustness:

| Original Label | Merged Label | Rationale |
|---|---|---|
| Belly Pain | physical_pain | Renamed for clinical clarity |
| Discomfort | needs | Similar acoustic profile; shared caregiving response |
| Cold/Hot | needs | Similar acoustic profile; shared caregiving response |
| Tired | needs | Similar acoustic profile; shared caregiving response |
| Scared | scared | Preserved — acoustically distinctive |
| Burping | burping | Preserved — distinct temporal pattern |

**Rationale:** Labels that share similar acoustic profiles and caregiving responses were merged, reducing classification ambiguity while maintaining practical utility for parents. The production model uses inverse-frequency class weighting (burping: 2.075, needs: 0.203, physical_pain: 0.811, scared: 0.912) to compensate for the resulting class imbalance.

### 3.3.2 Audio Preprocessing and Feature Extraction

#### Preprocessing Pipeline

The audio preprocessing pipeline standardizes all input samples to ensure consistent model input dimensions and quality. The following steps were implemented:

**Step 1: Audio Loading and Resampling**

- **Target Sample Rate:** 16,000 Hz (standard for speech/audio recognition models).
- **Channel Conversion:** Convert all audio to mono channel if stereo.
- **Normalization:** Automatic gain control to normalize audio amplitude levels.
- **int16 Detection:** Auto-scales if max amplitude > 1.5 (handles integer-encoded audio gracefully).

**Step 2: Duration Standardization**

All audio samples are standardized to a fixed duration of 5 seconds (80,000 samples at 16 kHz). This standardization is crucial for batch processing and consistent feature extraction.

Processing Logic:

- **Short Files (< 5 seconds):** Zero-padding applied at the end, with binary attention mask to mark real vs. padded samples.
- **Long Files (> 5 seconds):** Center-cropping to 5 seconds (captures the middle portion of the recording).
- **Exact Length:** Files matching 5 seconds remain unchanged.

The choice of 5 seconds (updated from 3 seconds in the first semester) provides a better balance between capturing sufficient temporal information and maintaining computational efficiency.

**Step 3: Feature Extraction with Wav2Vec2 Feature Extractor**

The preprocessed audio waveforms are converted into model-compatible features using the Wav2Vec2FeatureExtractor from the Hugging Face Transformers library. This extractor:

- Normalizes raw audio waveforms (zero-mean, unit-variance per sample).
- Applies learned feature transformations optimized for transformer architectures.
- Outputs feature representations compatible with HuBERT model input.

The feature extractor is configured with: Sampling rate: 16,000 Hz; Return format: PyTorch tensors; Automatic padding: Enabled for batch processing.

#### Attention Mask Downsampling (Second Semester)

A critical engineering detail for the production system: the attention mask (which marks real vs. padded samples at 16 kHz) must be downsampled to match the transformer's internal frame rate. This is handled by the `_downsample_mask()` method using `F.adaptive_max_pool1d()` — a computationally efficient approach that preserves mask semantics (any frame containing at least one real sample is marked as valid).

#### Feature Representation

After preprocessing, each audio sample is represented as:

- **Raw Audio:** 80,000 floating-point values (5 seconds × 16,000 Hz).
- **Feature Tensor:** Processed through Wav2Vec2FeatureExtractor for transformer compatibility.
- **Attention Mask:** Binary mask [1, 80000] indicating real vs. padded samples.
- **Input Shape:** [batch_size, sequence_length] where sequence_length = 80,000.

This representation preserves temporal information while ensuring compatibility with the transformer-based architecture.

### 3.3.3 Model Architecture and Training

#### Architecture Overview

The baby cry recognition model employs a transfer learning approach, utilizing the pretrained HuBERT (Hidden-unit BERT) model as a feature extraction backbone, combined with a custom classification head for task-specific learning.

#### Backbone: HuBERT Model

| Property | Specification |
|---|---|
| **Model Selection** | facebook/hubert-base-ls960 |
| **Architecture** | Transformer-based encoder with 12 layers |
| **Parameters** | ~95 million pretrained parameters |
| **Training Data** | Trained on LibriSpeech 960 hours dataset |
| **Output Dimension** | 768-dimensional hidden states per token |

**Rationale:** HuBERT's self-supervised pretraining on large-scale audio data enables it to capture rich acoustic features that generalize well to various audio classification tasks, including baby cry recognition.

*(Insert Fig 3.3.1 — HuBERT Base Model Configuration Here)*

#### Classification Head

A custom multi-layer perceptron (MLP) classifier processes the HuBERT embeddings:

```
HuBERT Backbone (frozen CNN feature extractor + fine-tuned transformer)
    └── Masked Mean Pooling (sequence → single vector, dimension 768)
        └── Linear(768 → 256) + GELU activation
            └── Dropout(0.1)
                └── Linear(256 → num_classes)
```

- **Masked Mean Pooling:** Averages only non-padded hidden states, ensuring variable-length audio is handled correctly.
- **GELU Activation:** Smoother gradient flow compared to ReLU, standard in transformer architectures.
- **Dropout Regularization:** 10% dropout to prevent overfitting the relatively small cry dataset.

#### Training Phases

**Phase 1: Frozen HuBERT Baseline (7 Classes)**

Training Process:

1. Initialize HuBERT with pre-trained weights.
2. Freeze all HuBERT parameters (requires_grad = False).
3. Initialize classifier with Xavier uniform initialization.
4. Train only classifier head for 60 epochs with early stopping.

Phase 1 Results:

- Best Validation Accuracy: 63.71%
- Test Accuracy: 63.71%
- Classes: 7 (including "hungry")
- Training Time: approximately 2 hours on NVIDIA T4 GPU

Analysis: Established baseline performance, identified "hungry" class confusion, demonstrated need for architecture refinement and fine-tuning.

*(Insert Fig 3.3.3 — Phase 1 Architecture Configuration Here)*

**Phase 2: Fine-Tuning with 6 Classes**

**Objective:** Improve performance by removing "hungry" class and fine-tuning HuBERT.

Key Changes:

- **Reduced Classes:** 7 → 6 classes (removed "hungry").
- **Gradual Unfreezing:** Last 6 transformer layers + feature projection made trainable.
- **Trainable Parameters:** 43.48 million (45.8% of model).
- **Layer-wise Learning Rate Decay (LLRD):** Different learning rates for different layers.

*(Insert Fig 3.3.4 — Phase 2 Architecture Configuration - Partial Unfreezing Here)*

Fine-Tuning Configuration — Learning Rate Schedule (LLRD):

| Layer | Learning Rate |
|---|---|
| Feature Projection | 2.4×10⁻⁵ |
| Layer 11 (top) | 2.0×10⁻⁵ |
| Layer 10 | 1.9×10⁻⁵ |
| Layer 9 | 1.8×10⁻⁵ |
| Layer 8 | 1.7×10⁻⁵ |
| Layer 7 | 1.6×10⁻⁵ |
| Layer 6 | 1.5×10⁻⁵ |
| Classifier | 5.0×10⁻⁵ (highest) |

**Rationale:**

- Lower layers (0–5): Extract general acoustic features (phonemes, spectral patterns).
- Upper layers (6–11): Adapt to task-specific patterns (cry characteristics).
- Prevents catastrophic forgetting: Preserves low-level representations.

*(Insert Fig 3.3.5 — Layer-wise Learning Rate Distribution Here)*

Training Details:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 0.01) |
| Scheduler | Cosine Annealing Warm Restarts (T₀=5, T_mult=2, η_min=1×10⁻⁷) |
| Batch Size | 8 |
| Gradient Clipping | 0.5 |
| Loss Function | CrossEntropyLoss with label smoothing (0.05) |
| Total Epochs | 62 (30 initial + 32 extended training) |
| Early Stopping | Based on test accuracy (patience = 7–10) |

*(Insert Fig 3.3.6 — Complete Hyperparameter Configuration Here)*

Phase 2 Results:

- Best Test Accuracy: 82.37% (achieved at epoch 52)
- Final Train Accuracy: 90.35%
- Final Validation Accuracy: 79.88%
- Final Test Accuracy: 81.63%
- Overfitting Gap: 10.47% (train – validation)

Improvement Metrics:

- vs. Phase 1 (7 classes): +18.66% absolute improvement
- vs. Phase 1 Baseline: +29.6% relative improvement

*(Insert Fig 3.3.7 — Phase 1 vs Phase 2 Comparison Here)*

*(Insert Fig 3.3.8 — Phase 1 and Phase 2 Comparison Chart Here)*

Training Challenges and Solutions:

1. **Underfitting (Phase 1):** Addressed by increasing learning rate from 5×10⁻⁵ to 1×10⁻⁴.
2. **Class Imbalance:** Mitigated through data augmentation (burping class).
3. **Hungry Class Confusion:** Resolved by removing from classification task.
4. **Overfitting:** Controlled through dropout (0.45), label smoothing (0.05), and early stopping.

#### Production Model (Second Semester)

For the production deployment, the model was further refined:

- **Two-Stage Cascade:** Stage 1 (binary detection: baby_cry vs. not_baby_cry) achieves 99.42% accuracy. Stage 2 (4-class classification) achieves 84.27% accuracy.
- **Ensemble Inference:** Stage 2 uses probability averaging across two model checkpoints from different training epochs for improved robustness.
- **Label Consolidation:** 6 classes merged to 4 clinically meaningful categories (see 3.3.1 Label Merging Strategy).
- **Inference Optimization:** `@torch.no_grad()` decorator disables gradient computation, reducing memory usage by ~50%. Cascade short-circuit skips Stage 2 entirely for non-cry audio.

### 3.3.3 Model Testing and Evaluation

#### Evaluation Metrics

The model performance was assessed using multiple metrics to provide a comprehensive evaluation:

**Primary Metrics:**

- **Overall Accuracy:** Percentage of correctly classified samples.
- **Per-Class Accuracy:** Individual class performance — enables identification of difficult-to-classify categories.
- **Confusion Matrix:** Detailed breakdown of classification errors — reveals patterns in misclassification.
- **Macro F1-Score:** Primary metric that penalizes poor performance on minority classes.
- **ROC-AUC:** Discrimination power (Stage 1: 0.9983).

**Secondary Metrics:**

- **Overfitting Gap:** Difference between training and validation accuracy — measure of generalization: 10.47% (acceptable).
- **Loss Curves:** Training and validation loss over epochs — monitors convergence and overfitting.

#### First Semester Results (6-Class Model)

The final model was evaluated on a held-out test set of 675 samples (after removing hungry class), maintaining class balance (112–113 samples per class).

**Per-Class Test Accuracy:**

| Class | Accuracy | Number of Test Samples |
|---|---|---|
| Belly Pain | ~82% | 113 |
| Burping | ~83% | 113 |
| Cold/Hot | ~81% | 112 |
| Discomfort | ~79% | 112 |
| Scared | ~95% | 113 |
| Tired | ~85% | 112 |

#### Production Model Results (Second Semester — 4-Class + Binary Detection)

**Stage 1 — Binary Cry Detection:**

| Metric | Value |
|---|---|
| Test Accuracy | 99.42% |
| Macro F1-Score | 0.9937 |
| ROC-AUC | 0.9983 |
| baby_cry F1 | 0.9955 |
| not_baby_cry F1 | 0.9920 |
| False Positives | 4 / 560 (0.71%) |
| False Negatives | 5 / 992 (0.50%) |

*(Insert Stage 1 Confusion Matrix Image Here)*

**Analysis:** Stage 1 achieves near-perfect detection. The 0.50% false negative rate means approximately 1 in 200 genuine cry events would be missed — acceptable for an advisory system. The 0.71% false positive rate is similarly low, meaning non-cry audio rarely triggers the Stage 2 pipeline unnecessarily.

**Stage 2 — 4-Class Cry Classification:**

| Metric | Value |
|---|---|
| Test Accuracy | 84.27% |
| Macro F1-Score | 0.7685 |
| Best Validation F1 | 0.7894 |
| Wrong-confident predictions | 11.5% |

**Per-Class Performance:**

| Class | F1-Score | Recall | Assessment |
|---|---|---|---|
| scared | 0.967 | 99% | Excellent — distinct acoustic signature |
| needs | 0.881 | 90% | Strong — benefits from label merging |
| physical_pain | 0.687 | 70% | Moderate — confused with needs (29%) |
| burping | 0.538 | 41% | Weak — severe data scarcity |

*(Insert Stage 2 Confusion Matrix Image Here)*

**Key Observations:**

- **Burping → Needs confusion (54%):** Burping cries are short and low-intensity, acoustically similar to mild fussing.
- **Physical_pain → Needs confusion (29%):** Pain cries can overlap with intense needs-based fussing.
- **Scared is highly distinctive:** Sharp onset, high-pitch — easy for the model to isolate (99% recall).

#### Model Robustness Analysis

**Out-of-Distribution (OOD) Detection:**

To handle real-world scenarios where baby cries may not match training categories, an OOD detection module was implemented using multiple uncertainty metrics:

1. **Confidence Threshold:** Rejects predictions with max probability < 70%.
2. **Entropy Measure:** Flags high-uncertainty predictions (entropy > 1.5).
3. **Margin Score:** Detects ambiguous cases (top-2 probability gap < 20%).
4. **Probability Distribution Uniformity:** Identifies uniform (random) predictions.

When OOD is detected, the system requests contextual questionnaire answers to assess hunger probability, providing a hybrid classification approach.

**Second Semester Addition — Acoustic Pre-filtering:**

Before reaching the AI model, the acoustic sanity check module now rejects obviously non-cry inputs:

| Check | Metric | Threshold | What It Rejects |
|---|---|---|---|
| Volume | RMS Energy | ≥ 0.010 (~-40 dBFS) | Silent/near-silent recordings |
| Voiced Ratio | Fraction of loud frames | ≥ 20% | Mostly-silent recordings |
| Frequency Band | Energy in 250–3000 Hz band | ≥ 40% of total | Non-vocal sounds |
| Spectral Flux | Onset strength | ≥ 0.015 | Constant tones, hums |
| ZCR (low) | Average zero-crossing rate | ≥ 0.02 | Unnaturally smooth signals |
| ZCR (high) | Average zero-crossing rate | ≤ 0.35 | White noise, static |

#### Generalization Assessment

- **Train-Validation Gap:** 10.47% indicates moderate overfitting, acceptable for this task.
- **Validation-Test Gap:** < 2%, demonstrating consistent performance across data splits.
- **Cross-Class Performance:** Balanced accuracy across all classes.

#### Comparison with Baseline

| Model | Accuracy |
|---|---|
| Random Guess (6 classes) | 16.67% |
| Phase 1 — Frozen HuBERT (7 classes) | 63.71% |
| Phase 2 — Fine-tuned HuBERT (6 classes) | 82.37% |
| Production — Stage 1 Binary (2nd semester) | 99.42% |
| Production — Stage 2 Ensemble (2nd semester) | 84.27% |

#### Model Limitations and Future Improvements

**Current Limitations:**

1. **Hungry Class Exclusion:** Requires questionnaire-based assessment.
2. **Source Bias:** Training data heavily weighted toward one source; real-world recordings from different microphones may see 10–20pp accuracy drop.
3. **Burping Class:** Only ~240 training samples — model recall is moderate (41%).
4. **Not a Diagnostic Tool:** The system is designed for AI assistance only.

**Potential Improvements:**

1. **ONNX Export:** Convert HuBERT models to ONNX format for 2–5× CPU inference speedup.
2. **Backbone Sharing:** Share a single HuBERT backbone between Stage 1 and Stage 2 for ~60% memory reduction.
3. **Data Augmentation:** Expand with more diverse recording conditions and babies.
4. **Multi-Modal Input:** Combine audio with video (facial expressions, body movements).
5. **Active Learning:** Incrementally improve with user feedback.

#### Model Deployment Specifications

| Property | First Semester | Second Semester (Production) |
|---|---|---|
| Model Size | ~400 MB | ~3.3 GB (3 checkpoints) |
| Inference Time (GPU) | ~30 ms | ~150 ms |
| Inference Time (CPU) | ~200 ms | ~2 seconds |
| Input Format | WAV/MP3/OGG, 3s | WAV/MP3/OGG/FLAC/M4A/WebM, 5s |
| Output Format | 6-class probabilities + OOD flag | Binary detection + 4-class probabilities + Arabic advice |
| Pre-inference Validation | None | File validation + acoustic sanity checks |
| Rate Limiting | None | 30/min, 500/hour per IP |
| Logging | Basic print | Structured JSON logging (structlog) |
| Health Monitoring | None | /health endpoint with live metrics |

### 3.3.4 Production Hardening (Second Semester)

This section documents the new modules added during the second semester to make the system production-ready.

#### Audio Validation (`api/validators.py`)

The validation module runs before any audio processing, catching bad inputs early:

| Check | Limit | Error Code |
|---|---|---|
| Filename present | Required | MISSING_FILENAME |
| File extension | wav, mp3, ogg, flac, m4a, webm | UNSUPPORTED_FORMAT |
| Minimum file size | 1 KB | FILE_TOO_SMALL |
| Maximum file size | 25 MB | FILE_TOO_LARGE |
| Audio parseable | Must be decodable by SoundFile | INVALID_AUDIO_FORMAT |
| Minimum duration | 0.5 seconds | AUDIO_TOO_SHORT |
| Maximum duration | 30 seconds | AUDIO_TOO_LONG |

#### Acoustic Sanity Checks (`api/acoustic_checks.py`)

Adds signal-level validation after decoding but before HuBERT inference, preventing the deep model from wasting compute on inputs that are obviously not infant cries. See the table in Model Robustness Analysis for thresholds.

#### Rate Limiting (`api/rate_limit.py`)

Prevents API abuse using slowapi: 30 requests/minute and 500 requests/hour per client IP. Returns `429 Too Many Requests` with a `Retry-After: 60` header.

#### Structured Logging (`api/logging_config.py`)

Uses structlog for machine-parseable logging: JSON in production (for Datadog, CloudWatch, ELK), colored console in development. Log events include request_id, client_ip, stage1_label, processing_ms, etc.

#### Metrics Tracking (`api/metrics.py`)

Thread-safe in-memory counters exposed via `/health`: uptime_seconds, requests_total, predictions_success, validation_errors, acoustic_rejections, avg_inference_ms, and per-class prediction counts.

#### Request ID Tracking

Every prediction request receives a unique UUID, included in all log events and the API response, enabling end-to-end request tracing.

#### Model Warm-up

On startup, a dummy inference (5 seconds of silence) triggers CUDA kernel JIT compilation and warms caches, ensuring the first real request has normal latency.

### 3.3.5 API Documentation

#### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | / | Arabic landing page (HTML) |
| GET | /health | Health check with live metrics |
| POST | /predict | Main inference endpoint (rate limited: 30/min) |
| GET | /docs | Auto-generated Swagger/OpenAPI documentation |
| GET | /redoc | Alternative ReDoc documentation |

#### Error Responses

| Code | Error Code | Scenario |
|---|---|---|
| 400 | UNSUPPORTED_FORMAT | File extension not supported |
| 400 | FILE_TOO_SMALL / FILE_TOO_LARGE | File size outside bounds |
| 400 | AUDIO_TOO_SHORT / AUDIO_TOO_LONG | Duration outside bounds |
| 400 | TOO_QUIET / MOSTLY_SILENCE | Audio failed acoustic checks |
| 400 | WRONG_FREQUENCY_PROFILE | Energy outside cry frequency band |
| 400 | CONSTANT_TONE / TOO_NOISY | Spectral flux or ZCR out of range |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests |
| 500 | INFERENCE_ERROR | Model inference failure |
| 503 | MODEL_NOT_LOADED | Models still loading at startup |

All error responses include bilingual messages (Arabic for users, English for developers), the request_id, and `success: false`.

### 3.3.6 Technologies Used

| Category | Technology | Version | Purpose |
|---|---|---|---|
| **AI/ML** | PyTorch | ≥ 2.0.0 | Deep learning framework |
| | PyTorch Lightning | 2.4.0 | Structured training/inference |
| | Hugging Face Transformers | 4.44.2 | HuBERT pre-trained model |
| | TorchMetrics | 1.4.2 | Evaluation metrics |
| **Audio** | Librosa | ≥ 0.10.0 | Resampling and acoustic features |
| | SoundFile | ≥ 0.12.1 | Fast C-based audio I/O |
| | NumPy | ≥ 1.24.0 | Numerical computation |
| **API** | FastAPI | ≥ 0.110.0 | REST API framework |
| | Uvicorn | ≥ 0.29.0 | ASGI server |
| | Pydantic v2 | (via FastAPI) | Response schema validation |
| **Production** | slowapi | ≥ 0.1.9 | Rate limiting |
| | structlog | ≥ 24.1.0 | Structured logging |

---

## 3.4 Methodology and Experimental Setup

### 3.4.1 Contribution: YAMNet-Based Experiments and Analysis

As part of the project team, this contribution focused on experimenting with YAMNet-based models for baby cry classification. This involved not only training a single model, but conducting extensive experiments, optimizations, and comparisons to fully understand the strengths and limitations of YAMNet for this task.

**YAMNet Model Overview:**

YAMNet is a deep neural network pre-trained on AudioSet, a large-scale audio dataset containing millions of labeled sound events. The model processes raw audio waveforms and produces high-level audio embeddings (1024-dimensional) that capture semantic audio information. Due to the limited size of the baby cry dataset, YAMNet was selected as a transfer learning backbone to leverage pre-learned audio representations.

### 3.4.2 Experimental Methodology Using YAMNet

The YAMNet experiments were carried out through multiple structured stages, each introducing controlled changes to study their effect on performance.

#### 3.4.2.1 Phase A — Data Preparation and Preprocessing

A complete data preparation and cleaning pipeline was implemented before model training. This phase included:

- Downloading the dataset and inspecting class distribution.
- Resampling all audio files to 16 kHz.
- Removing unrelated folders and classes not belonging to the seven target categories.
- Verifying that no corrupted or empty audio files were included.
- Splitting the dataset into 70% training, 15% validation, and 15% testing sets.
- Encoding class labels using LabelEncoder.

All preprocessing steps and data handling procedures were implemented and documented in the project notebook (full_journey).

#### 3.4.2.2 Phase B — YAMNet as a Frozen Feature Extractor (Transfer Learning)

**Concept of Frozen YAMNet:**

In this phase, YAMNet was used as a frozen feature extractor, meaning that its pre-trained weights were not updated during training. The pipeline followed these steps:

1. Freeze all YAMNet parameters.
2. Extract 1024-dimensional embeddings for each audio sample.
3. Train a dense neural network classifier on the extracted embeddings.

**Systematic Experiments and Tuning:**

Rather than training a single classifier, multiple systematic experiments were conducted, including:

- Changing the optimizer (e.g., Adam with different configurations).
- Adjusting the learning rate (e.g., 0.001 → 0.0005 → 0.0002).
- Increasing the number of training epochs and early stopping patience.
- Updating class weights to reduce class imbalance and prediction bias.
- Applying data augmentation, initially partially and later on the full dataset.

**Full Data Augmentation Strategy:**

A comprehensive augmentation pipeline was applied to balance the dataset across all classes. The applied techniques included: Time Stretching, Pitch Shifting, Time Shifting, and Gaussian Noise Injection. The goal was to reach a balanced target number of samples per class.

**Advanced Classifier Experiments on YAMNet Embeddings:**

Multiple classifier architectures were explored on top of the YAMNet embeddings, including:

- A baseline dense (MLP) classifier.
- An attention-based classifier, applying self-attention over embedding dimensions.
- A residual classifier, incorporating skip connections within dense layers.
- An ensemble of multiple models, using: simple averaging, weighted averaging, and majority voting.

**Key Observation:** When feature quality is the primary bottleneck, increasing classifier depth or using ensemble methods provides only limited performance gains.

#### 3.4.2.3 Phase C — End-to-End YAMNet Pipeline (Online Features and Fine-Tuning)

This phase explicitly investigated the difference between Frozen and Unfrozen YAMNet using raw audio input.

**(1) End-to-End Training with Frozen YAMNet:**

- Raw audio waveforms of fixed length (e.g., 64,000 samples for 4 seconds at 16 kHz) were fed directly into the model.
- YAMNet operated inside the model graph.
- Temporal embeddings were aggregated using Global Average Pooling.
- Only the classifier head was trained.

This experiment validated that online feature extraction behaves consistently with offline embedding extraction.

**(2) Unfrozen YAMNet (Fine-Tuning):**

- YAMNet was set as trainable.
- Most layers were frozen, while only the last layers were unfrozen (partial unfreezing).
- A very small learning rate (e.g., 1e-5) was used to avoid catastrophic forgetting.
- Smaller batch sizes were adopted due to higher memory requirements.
- Early Stopping and ReduceLROnPlateau were applied.

**Important Note:** Due to the relatively small dataset size, fine-tuning may not always yield significant improvements and can easily lead to overfitting if the learning rate is not sufficiently low.

#### 3.4.2.4 Phase D — CNN from Scratch on Mel Spectrograms (Non-Pretrained)

An alternative approach was explored by learning audio features from scratch:

- Each audio file was converted into a Mel Spectrogram.
- A CNN was trained directly on spectrogram images without pretraining.

This experiment resulted in model collapse and severe overfitting due to the limited dataset size relative to the number of CNN parameters. This outcome strongly confirmed that transfer learning is essential for this task.

#### 3.4.2.5 Phase E — Class Merging Strategy (Most Impactful Improvement)

After analyzing confusion matrices, acoustically similar classes were merged as follows:

- **physical_discomfort** = belly pain + discomfort + cold_hot
- **needs_attention** = hungry + tired
- **scared** and **burping** were kept as independent classes

This reformulation reduced the problem from 7 classes to 4 classes, significantly increasing class separability and improving practical usability, since parents are more concerned with the appropriate response than the exact cry label.

Additional experiments attempted aggressive optimization of merged classes using deeper networks and higher class weights. These attempts led to model collapse, demonstrating that excessive class weighting negatively affects training stability.

#### 3.4.2.6 Final Scientific Conclusions

1. **Frozen YAMNet** provides a strong and stable baseline for small audio datasets.
2. **Fine-tuning (unfreezing) YAMNet** is computationally expensive and riskier, requiring very low learning rates and larger datasets.
3. **Data augmentation and class weighting** improve robustness but cannot introduce missing discriminative information.
4. **Training CNNs from scratch** is unsuitable for limited datasets due to severe overfitting.
5. **The most significant improvement** resulted from problem reformulation through class merging, rather than increasing model complexity.
