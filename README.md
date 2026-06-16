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

---
---

# MuMZ-main — Chatbot & Storytelling Modules

The `MuMZ-main/` folder contains two additional modules that complement the Baby Cry Classifier:

1. **Mumz Chatbot** (`mums_chatbot-main/`) — An Arabic-language AI assistant for infant care
2. **XTTS Storytelling** (`xtts-main/`) — A text-to-speech system that reads stories aloud in a cloned or default Arabic voice

---

## 🍼 Module 1: Mumz Chatbot — Arabic Baby Care Assistant

**Location:** `MuMZ-main/mums_chatbot-main/`

### What It Is

Mumz is an AI chatbot that answers mothers' questions about infant health, nutrition, sleep, milestones, and early childhood care — targeting children from **birth to 3 years old**. It communicates in **Egyptian Arabic (عامية مصرية)** and combines Retrieval-Augmented Generation (RAG), a multilingual LLM, and a rule-based safety layer to produce trustworthy, age-aware responses.

The entire project lives inside a single Jupyter notebook: `mumz_chatbot_graduation_projectipynb.ipynb`.

### Key Features

| Feature | Description |
|---|---|
| **Full Arabic Support** | Understands Egyptian colloquial Arabic including spelling variations and synonyms |
| **RAG-Powered Answers** | Retrieves relevant knowledge from a curated Qdrant vector database before generating a response |
| **Age-Aware Responses** | Extracts the child's age from natural language (Arabic numerals, words like `سنتين`, `شهرين`, `سنة ونص`) and tailors answers accordingly |
| **Safety Layer** | Detects emergencies (`تشنج`, `اختناق`, `نزيف`), refuses to prescribe medications, and recommends consulting a doctor |
| **Session Memory** | Tracks child info (age, feeding type) and conversation history within a session; supports multiple children |
| **Milestone Tracking** | Provides normal developmental ranges and red flags for walking, talking, crawling, and teething |
| **Smart Suggestions** | Generates contextual follow-up questions after each response |
| **Gradio UI** | Clean, streaming chat interface runnable in Google Colab |

### Architecture

```
User Message (Arabic)
        │
        ▼
┌───────────────────┐
│  Safety & Intent  │  ← Greeting / Danger / Medical / General
│  Classification   │
└────────┬──────────┘
         │
    ┌────┴─────┐
    │          │
 Rule-based  RAG Retrieval
 Response    (Qdrant + LlamaIndex)
    │          │
    └────┬─────┘
         │
         ▼
┌────────────────────┐
│  LLM Generation    │  ← Cohere command-r (streaming)
│  (Aya-Expanse 8B   │     or Aya-Expanse 8B (4-bit)
│   as fallback)     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Post-Processing   │  ← Hallucination filter, deduplication,
│  & Sanitization    │     medical dosage stripping
└────────┬───────────┘
         │
         ▼
    Final Response + Smart Suggestions
```

### How It Works

1. **Intent Classification** — Every message is classified as:
   - **GREETING** — Social messages (`أهلا`, `شكرا`) → fixed warm response
   - **DANGER** — Emergency keywords (`تشنج`, `اختناق`, `نزيف شديد`) → immediate emergency instruction
   - **MEDICAL** — Dosage/prescription requests (`جرعة`, `باراسيتامول`) → politely refused, doctor recommended
   - **GENERAL** — Health, nutrition, sleep, development questions → full RAG + LLM pipeline

2. **RAG Pipeline** — For general questions:
   - The query is rewritten using conversation history and synonym expansion
   - Top-3 relevant chunks are retrieved from Qdrant (similarity threshold `0.65`)
   - Chunks are cleaned, trimmed to 1200 chars, and injected into the LLM prompt
   - The LLM generates a response grounded only in the retrieved context

3. **Safety & Post-Processing** — Before returning the response:
   - Regex-based hallucination patterns catch fabricated doctor names, fake formulas, and incorrect dosages
   - Dosage numbers (`\d+ مل`, `\d+ mg`) are stripped from all outputs
   - Semantic relevance score (cosine similarity) validates that the response addresses the question

### Tech Stack

| Component | Technology |
|---|---|
| **LLM** | `CohereForAI/aya-expanse-8b` (4-bit quantized via bitsandbytes) |
| **Streaming LLM** | Cohere `command-r-08-2024` API |
| **Embeddings** | `intfloat/multilingual-e5-large` (HuggingFace) |
| **Vector Store** | Qdrant (local) |
| **RAG Framework** | LlamaIndex + LangChain |
| **UI** | Gradio |
| **Runtime** | Google Colab (GPU) |

### Notebook Sections

| Section | Description |
|---|---|
| `Installation` | `pip` installs for all dependencies |
| `Imports` | All library imports |
| `Setup & Config` | Google Drive mount, Qdrant client init, HuggingFace login |
| `Constants & Rules` | Greetings, disclaimers, food rules, medical rules, milk quantities |
| `Memory` | `BabyAssistantMemory` class — session state, child info, history |
| `Helper Functions` | Age extraction, Arabic normalization, query rewriting, context cleaning |
| `LLM & RAG` | LLM call wrappers (streaming + non-streaming), retriever setup, `retrieve_context()`, `generate()` |
| `Safety & Intent` | `classify_intent()`, `safety_check()`, `apply_medical_rules()`, `check_milestone()`, out-of-scope detection |
| `Post Processing` | Hallucination pattern detection, response sanitization, semantic relevance scoring |
| `Core Chat` | `ask()` — main orchestration function |
| `UI & Gradio` | `respond()` + Gradio chat interface with streaming |
| `Test` | RAG unit tests, memory tests, eval suite with semantic scoring, results dashboard |

### Prerequisites

- Google Colab account (free tier works; GPU recommended for Aya-Expanse)
- [Cohere API key](https://cohere.com/)
- [HuggingFace API key](https://huggingface.co/settings/tokens)
- Pre-built Qdrant vector database (`rag_data_backup.zip`) stored in Google Drive

### Running the Chatbot

1. Open the notebook in Colab
2. Add secrets in Colab (🔑 sidebar): `HUGGINGFACE_API_KEY` and `COHERE_API_KEY`
3. Upload `rag_data_backup.zip` to Google Drive at `MyDrive/rag_data_backup.zip` containing:
   - `my_qdrant_data/` — Qdrant collection directory
   - `docstore.json` — LlamaIndex document store
4. Run all cells — the notebook installs dependencies, loads models, and launches a Gradio chat interface

### Testing

The notebook includes a built-in evaluation suite covering:
- **Happy path** — Normal nutrition and care questions
- **Safety** — Danger detection, medication refusal
- **Boundary** — Edge cases (very young ages, unclear questions)
- **Out-of-scope** — Adult recipes, non-baby topics
- **Greetings** — Social and thanks messages

Results are visualized in a 4-panel Matplotlib dashboard showing pass/fail by category, response times, and semantic relevance scores.

---

## 📖 Module 2: XTTS Storytelling — Arabic Text-to-Speech with Voice Cloning

**Location:** `MuMZ-main/xtts-main/`

### What It Is

The XTTS module is a **text-to-speech storytelling system** that converts Arabic text into natural-sounding speech. It supports **voice cloning** — a mother can record her own voice and the system will read stories aloud using a synthesized version of that voice. This is designed for bedtime storytelling when a parent is unavailable, providing the child with a familiar voice.

The module consists of two parts:
- **Backend** (`backend/`) — A FastAPI server wrapping the Coqui XTTS v2 model
- **Flutter Client** (`lib/`) — A cross-platform mobile/web app for recording voice samples and triggering synthesis

### Architecture

```
Flutter App (Mobile / Web)
        │
        │  1. Record/upload voice samples
        │  2. Enter story text
        │
        ▼
┌────────────────────────┐
│  FastAPI Backend        │
│  (Docker container)     │
│                         │
│  /upload_speakers       │ ← Upload & convert audio to WAV 16kHz mono
│  /synthesize            │ ← TTS with optional diacritization
│  /generated/{file}      │ ← Download generated audio
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  XTTS v2 Model          │
│  (Coqui TTS)            │
│                         │
│  • Default voice mode   │ ← Uses bundled speaker (saniya-nrl.mp3)
│  • Clone mode           │ ← Clones user's recorded voice
│  • Merge mode           │ ← Merges multiple voice samples
└────────────────────────┘
         │
         ▼
   Generated WAV Audio
   (streamed back to app)
```

### Voice Modes

| Mode | Description |
|---|---|
| **Default** | Uses a pre-bundled Arabic speaker voice (`saniya-nrl.mp3`) |
| **Clone** | Clones the user's own voice from one or more uploaded audio samples |
| **Merge** | Merges multiple voice recordings into a single speaker profile before synthesis |

### Backend Details

The backend is a **Python FastAPI** application containerized with **Docker**:

#### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — shows available endpoints |
| `/upload_speakers` | POST | Accepts multiple audio files (WAV/MP3/FLAC/OGG/M4A/WebM), converts to WAV 16kHz mono, returns server paths |
| `/synthesize` | POST | Synthesizes speech from text. Parameters: `text`, `mode` (default/clone/merge), `speaker_paths`, `use_diacritization` |
| `/generated/{filename}` | GET | Download a previously generated WAV file |
| `/generate` | POST | Legacy endpoint for backward compatibility |

#### Audio Processing Pipeline

When speaker files are uploaded for voice cloning, the backend applies:

1. **Resampling** — Convert to 16kHz mono
2. **VAD (Voice Activity Detection)** — Uses `webrtcvad` to trim non-speech segments
3. **Noise Reduction** — Light noise reduction via `noisereduce` (prop_decrease=0.4)
4. **Silence Trimming** — Conservative trimming at 35 dB threshold
5. **RMS Normalization** — Match loudness levels across files
6. **Merging** — If multiple files, concatenate processed audio into a single speaker reference

#### Arabic Diacritization

Before synthesis, the text is automatically diacritized using **Farasa** (an Arabic NLP toolkit). Diacritization adds short vowel marks (tashkeel) to the Arabic text, which significantly improves pronunciation quality. This can be toggled on/off per request.

#### Key Files

| File | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI application — endpoint definitions, CORS config, request handling |
| `backend/app/tts_service.py` | `TTSService` class — model loading, audio preprocessing, VAD, noise reduction, merging, synthesis |
| `backend/app/utils.py` | Helper functions — file validation, WAV conversion, Farasa diacritization |
| `backend/download_xtts.py` | Script that pre-downloads the XTTS v2 model during Docker build |
| `backend/Dockerfile` | Docker image definition (Python 3.11-slim + ffmpeg + sox + Java for Farasa) |
| `backend/docker-compose.yml` | Docker Compose config — maps port 8000, mounts volumes for uploads/generated audio |
| `backend/requirements.txt` | Python dependencies |
| `backend/saniya-nrl.mp3` | Default Arabic speaker voice sample |

#### Tech Stack (Backend)

| Component | Technology |
|---|---|
| **TTS Model** | `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui TTS) |
| **Framework** | FastAPI + Uvicorn |
| **Audio Processing** | librosa, soundfile, noisereduce, webrtcvad |
| **Diacritization** | Farasa |
| **Containerization** | Docker + Docker Compose |
| **Language** | Python 3.11 |

#### Synthesis Parameters

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.45 | Controls randomness in generation |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `repetition_penalty` | 1.08 | Penalizes repeated tokens |
| `FP16` | Off (CPU mode) | Half-precision for GPU acceleration |

### Flutter Client Details

The Flutter app (`lib/`) provides a mobile and web interface:

#### Features

- **Voice Recording** — Record voice samples directly from the device microphone (mobile via `record` package, web via `MediaRecorder` API)
- **File Picker** — Select existing audio files (WAV, MP3, M4A, OGG, FLAC)
- **Voice Mode Selector** — Dropdown to choose Default Voice, Voice Cloning, or Merge Voices
- **Arabic Diacritization Toggle** — Enable/disable automatic tashkeel
- **Text Input** — Enter the story or text to be spoken
- **Upload & Synthesize** — Two-step flow: upload speaker files, then synthesize
- **Audio Playback** — Play/stop generated audio directly in the app

#### Key Files

| File | Purpose |
|---|---|
| `lib/main.dart` | App entry point — sets up `MaterialApp` with `RecorderPage` |
| `lib/recorder.dart` | Main UI — file picker, recording, upload, synthesis, playback controls |
| `pubspec.yaml` | Flutter dependencies: `http`, `path_provider`, `file_picker`, `record`, `audioplayers` |

### Running the Storytelling Module

#### Backend (Docker)

```bash
cd MuMZ-main/xtts-main/backend
docker-compose up --build
# Backend starts on http://localhost:8000
```

#### Flutter Client

```bash
cd MuMZ-main/xtts-main
flutter pub get
flutter run
```

> **Note:** The Flutter client connects to the backend at the URL configured in `recorder.dart` (`backendBaseUrl` parameter). Update this to match your backend's address.

### Hardware Requirements

| Setup | Notes |
|---|---|
| **CPU-only (default)** | Works but synthesis is slow (~30-60s per utterance). Docker Compose sets `USE_GPU=0` |
| **GPU (CUDA)** | Significantly faster. Set `USE_GPU=1` in `docker-compose.yml` and ensure NVIDIA Docker runtime is available |

---

## ⚠️ Disclaimer

Both modules are **informational and assistive tools only**. The chatbot does not provide medical diagnoses or prescribe medications. The storytelling module is designed for entertainment and comfort. Always consult qualified professionals for medical decisions regarding your child.
