# 🎵 AI-Based Music Composition System Using Transformer Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-4B9CD3?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/ci.yml?style=for-the-badge&label=CI)](https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/actions)

**End-to-End Generative AI project for original music composition using Transformer Networks, RAG, and Music Theory Constraints**

[🎵 Live Demo](#-local-setup) • [📓 Colab Training](https://colab.research.google.com/github/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/blob/main/notebooks/02_model_training.ipynb) • [📖 Documentation](#-architecture) • [🔌 API Docs](#-flask-rest-api)

</div>

---

## 📖 Overview

This project implements a **complete end-to-end AI music composition pipeline** that generates original musical compositions from:
- 🌱 **Seed note sequences** (user-provided starting notes)
- 📝 **Text-based parameters** (genre, mood, key, mode, tempo)
- 📁 **Uploaded MIDI files** (style-guided generation via RAG)

The system combines a **custom Transformer Encoder-Decoder** model with a **Music Theory Constraint Engine** and a **Retrieval-Augmented Generation (RAG) pipeline** to produce coherent, musically valid compositions.

> **✨ Demo Mode Available**: The app runs immediately without a trained model using music-theory-based heuristic generation (Markov chains + scale constraints). No GPU required!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Music Composition System                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │  MIDI Parser │───▶│  Tokenizer   │───▶│  Transformer Model │    │
│  │  (pretty_midi│    │  (512 vocab) │    │  (Enc-Dec, 6 layer)│    │
│  │   + mido)    │    │              │    │  8 heads, 256-dim  │    │
│  └──────────────┘    └──────────────┘    └────────────────────┘    │
│         │                                          │                 │
│         ▼                                          ▼                 │
│  ┌──────────────┐                      ┌────────────────────┐      │
│  │ Preprocessor │                      │  Inference Engine  │      │
│  │ (windowing,  │                      │  • Greedy          │      │
│  │  augmentation│                      │  • Temperature     │      │
│  │  normalization)                     │  • Top-K / Top-P   │      │
│  └──────────────┘                      │  • Beam Search     │      │
│                                        │  • Demo Mode ✨    │      │
│  ┌──────────────────────────────────┐  └────────────────────┘      │
│  │  RAG Pipeline                    │           │                    │
│  │  ┌──────────┐   ┌─────────────┐ │           ▼                   │
│  │  │  Embedder│──▶│ FAISS Index │─┼──▶ ┌────────────────────┐    │
│  │  │ (128-dim)│   │ (Top-K)     │ │    │ Music Theory Engine │    │
│  │  └──────────┘   └─────────────┘ │    │ • Key/Scale valid. │    │
│  └──────────────────────────────────┘    │ • Chord progressions│   │
│                                          │ • Interval rules   │    │
│  ┌──────────────┐  ┌─────────────────┐  └────────────────────┘    │
│  │  Streamlit   │  │   Flask REST API│           │                  │
│  │  Web App     │  │   /api/generate │           ▼                  │
│  │  (Piano Roll │  │   /api/upload   │    ┌────────────────┐       │
│  │   Viz, RAG   │  │   /api/history  │    │  MIDI Output   │       │
│  │   Upload)    │  └─────────────────┘    │  (.mid file)   │       │
│  └──────────────┘                         └────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Music Generation** | Custom Transformer Encoder-Decoder with 6 layers, 8 heads, 256-dim embeddings |
| ✨ **Demo Mode** | Works immediately with heuristic generation — no GPU or training required |
| 🎼 **Music Theory Engine** | Key/scale validation, chord progressions, interval consonance, rhythm checking |
| 🔍 **RAG Pipeline** | Upload MIDI files → 128-D embeddings → FAISS retrieval → style-guided generation |
| 🎛️ **Full Controls** | Genre, mood, key, mode, tempo, length, temperature, top-K, top-P, beam search |
| 📊 **Visualizations** | Piano roll, pitch histograms, attention heatmaps, training metrics |
| ⬇️ **MIDI Download** | Export compositions as standard MIDI files for any DAW |
| 🔌 **REST API** | Flask API for programmatic integration |
| ☁️ **Colab Training** | Train the model on GPU using our Google Colab notebook |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch 2.0+, Custom Transformer, Multi-Head Attention |
| **Music Processing** | pretty_midi, mido, music21, midiutil |
| **Generative AI** | Transformer Encoder-Decoder, Sequence-to-Sequence, Autoregressive |
| **RAG / Vector DB** | FAISS, custom feature embeddings (128-D) |
| **Frontend** | Streamlit 1.28+ |
| **Backend API** | Flask 3.0+, Flask-CORS |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Testing & CI** | pytest, flake8, GitHub Actions |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks.git
cd AI-Based-Music-Composition-System-Using-Transformer-Networks
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app (Demo Mode — no training required!)
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 5. (Optional) Run the Flask API
```bash
# In a separate terminal
python api/app.py
```

API available at `http://localhost:5000/api/`

---

## 🎯 Quick Start

### Generate music from Python
```python
from src.inference.generator import MusicGenerator

# Initialize in demo mode (no model required)
generator = MusicGenerator(demo_mode=True)

# Generate a Classical composition
result = generator.generate(
    genre="Classical",
    mood="Calm",
    key="C",
    mode="major",
    tempo=120,
    num_notes=64,
    temperature=1.0,
    apply_theory=True,
)

print(f"Generated {result.num_notes} notes")
print(f"Theory score: {result.theory_score:.2f}")
print(f"Duration: {result.total_duration:.1f}s")

# Save as MIDI
result.save_midi("my_composition.mid")
```

### Generate from CLI
```bash
python -m src.inference.generator \
  --genre Jazz \
  --mood Energetic \
  --key D \
  --mode dorian \
  --tempo 140 \
  --num_notes 128 \
  --output jazz_composition.mid
```

### Use the Flask API
```bash
# Generate music
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "genre": "Classical",
    "mood": "Calm",
    "key": "C",
    "mode": "major",
    "tempo": 120,
    "num_notes": 64,
    "apply_theory": true
  }'

# Health check
curl http://localhost:5000/api/health

# Upload MIDI for RAG
curl -X POST http://localhost:5000/api/upload-style \
  -F "midi_file=@my_style.mid"
```

---

## 🔌 Flask REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Available genres, moods, keys |
| `POST` | `/api/generate` | Generate music |
| `POST` | `/api/generate/download` | Generate + download MIDI directly |
| `POST` | `/api/upload-style` | Upload MIDI for RAG indexing |
| `GET` | `/api/history` | Generation history |
| `DELETE` | `/api/history` | Clear history |

### Example Response (`POST /api/generate`)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "pitches": [60, 64, 67, 65, 64, 62, 60],
  "durations": [0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 1.0],
  "velocities": [75, 68, 80, 72, 65, 70, 80],
  "midi_b64": "TVRoZAAAAAYAAAABAeA...",
  "theory_score": 0.8750,
  "generation_time_ms": 142.3,
  "method": "demo",
  "key": "C",
  "mode": "major",
  "genre": "Classical",
  "mood": "Calm",
  "tempo": 120,
  "num_notes": 64,
  "timestamp": 1703123456.789
}
```

---

## 🧠 Model Architecture

```
MusicTransformer(
  vocab_size=512, d_model=256, nhead=8
  encoder_layers=6, decoder_layers=6
  parameters=17,932,288
)
```

### Tokenizer Vocabulary (512 tokens)
```
Token ID  │ Type
──────────┼────────────────────────────────────────
0         │ PAD
1         │ BOS (Begin of Sequence)
2         │ EOS (End of Sequence)
3         │ REST
4         │ UNK
5–131     │ NOTE_ON  (MIDI pitches 0–127)
132–258   │ NOTE_OFF (MIDI pitches 0–127)
259–290   │ DURATION (32 log-spaced bins: 50ms–4s)
291–306   │ VELOCITY (16 bins: MIDI 0–127)
307–314   │ TEMPO    (8 bins: 40–240 BPM)
315–511   │ Reserved
```

---

## 🎼 Music Theory Engine

The Music Theory Constraint Engine validates generated notes using:

- **Key Signature Validation**: Checks if notes belong to the target key
- **Scale Membership**: Supports 10+ scales (Major, Minor, Dorian, Blues, Pentatonic...)
- **Chord Progressions**: Validates against common progressions (I-IV-V, ii-V-I, etc.)
- **Interval Consonance**: Checks melodic intervals for dissonance
- **Melody Range**: Ensures notes stay within a singable range (E2–C6)
- **Rhythm Consistency**: Validates against standard note durations

```python
from src.theory.music_theory_engine import MusicTheoryEngine

engine = MusicTheoryEngine(key="C", mode="major")
result = engine.validate_melody([60, 62, 64, 65, 67])
print(f"In-scale ratio: {result.in_scale_ratio:.0%}")
print(f"Theory score:   {result.score:.3f}")

# Apply automatic corrections
improved = engine.apply_constraints([61, 63, 66])  # Non-scale notes
print(improved)  # Snapped to nearest scale pitches
```

---

## 🔍 RAG Pipeline

```python
from src.rag.embedder import MusicEmbedder
from src.rag.retriever import MusicRetriever

# Extract 128-dimensional musical features from MIDI
embedder = MusicEmbedder()
embedding = embedder.embed_midi_file("my_style.mid")

# Build FAISS index and search
retriever = MusicRetriever()
retriever.add_midi(midi_bytes, "my_style.mid")
results = retriever.search(query_embedding, top_k=5)

# Feature dimensions:
# 12 dims  — Pitch class histogram (C, C#, D, ..., B)
# 16 dims  — Rhythm pattern (note duration distribution)
#  8 dims  — Velocity profile (dynamics)
# 12 dims  — Interval histogram (melodic intervals)
# ──────────────────────────────────────────────────
# 48 dims → projected to 128-D via orthogonal matrix
```

---

## ☁️ Train with Google Colab

Train the Transformer model on GPU using the included Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/blob/main/notebooks/02_model_training.ipynb)

The notebook covers:
1. Dataset download (MAESTRO / Lakh MIDI)
2. MIDI preprocessing and tokenization
3. Model training with AMP + warmup scheduling
4. Checkpoint saving to Google Drive
5. Model evaluation and sample generation

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tokenizer.py -v
pytest tests/test_theory_engine.py -v
pytest tests/test_generator.py -v

# Run quick smoke test
python -c "
from src.inference.generator import MusicGenerator
gen = MusicGenerator(demo_mode=True)
result = gen.generate(num_notes=16)
print(f'✅ {result.num_notes} notes generated, theory={result.theory_score:.2f}')
"
```

---

## 📁 Project Structure

```
AI-Based-Music-Composition-System-Using-Transformer-Networks/
├── README.md
├── requirements.txt
├── setup.py
├── pytest.ini
├── .gitignore
├── config/
│   └── config.yaml              # Central configuration
├── src/
│   ├── data/
│   │   ├── midi_parser.py       # MIDI parsing (pretty_midi + mido)
│   │   ├── preprocessor.py      # Windowing, augmentation, normalization
│   │   └── dataset.py           # PyTorch Dataset + DataModule
│   ├── tokenizer/
│   │   └── music_tokenizer.py   # 512-token music vocabulary
│   ├── model/
│   │   ├── transformer.py       # Transformer Encoder-Decoder
│   │   ├── attention.py         # Multi-Head Attention with weight extraction
│   │   └── positional_encoding.py
│   ├── theory/
│   │   └── music_theory_engine.py  # Theory validation + constraints
│   ├── rag/
│   │   ├── embedder.py          # 128-D musical feature embeddings
│   │   └── retriever.py         # FAISS vector store + retrieval
│   ├── inference/
│   │   └── generator.py         # All decoding strategies + demo mode
│   └── utils/
│       ├── visualization.py     # Piano roll, heatmaps, training plots
│       └── audio.py             # MIDI → base64, HTML player
├── training/
│   ├── train.py                 # Full training loop (AMP + warmup)
│   ├── evaluate.py              # Model evaluation + metrics
│   └── config_training.yaml     # Training hyperparameters
├── api/
│   ├── app.py                   # Flask application factory
│   └── routes.py                # All REST endpoints
├── app/
│   ├── streamlit_app.py         # Main Streamlit app
│   └── pages/
│       ├── generate.py          # Music generation UI
│       ├── upload_style.py      # RAG MIDI upload
│       ├── visualize.py         # Visualization dashboard
│       └── about.py             # Project info
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb  # Google Colab training notebook
├── tests/
│   ├── test_tokenizer.py
│   ├── test_theory_engine.py
│   └── test_generator.py
├── data/
│   └── sample/                  # Sample MIDI files for demo
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI
```

---

## 🗺️ Roadmap

- [x] Custom Transformer architecture
- [x] Music tokenizer (512-token vocabulary)
- [x] Music Theory Constraint Engine
- [x] Demo mode (heuristic generation, no GPU needed)
- [x] RAG pipeline with FAISS
- [x] Streamlit web application
- [x] Flask REST API
- [x] GitHub Actions CI
- [x] Google Colab training notebook
- [ ] Pre-trained model checkpoint (MAESTRO dataset)
- [ ] Multi-instrument support
- [ ] Real-time browser audio synthesis
- [ ] MIDI to sheet music rendering
- [ ] Chord voicing and harmonization

---

## 📖 Learning Outcomes

Through this project, I gained hands-on experience in:

- ✅ **Transformer architecture** implementation from scratch in PyTorch
- ✅ **Symbolic music processing** with MIDI files
- ✅ **MIDI preprocessing and tokenization** for sequence modeling
- ✅ **Sequence-to-sequence learning** with teacher forcing
- ✅ **Retrieval-Augmented Generation (RAG)** with FAISS
- ✅ **Vector database** integration and similarity search
- ✅ **Streamlit** interactive application development
- ✅ **Flask REST API** design and implementation
- ✅ **Git and GitHub** workflow with CI/CD
- ✅ **End-to-end AI** project development and deployment

---

## 👨‍💻 Author

**Rohit Birdawade**

[![GitHub](https://img.shields.io/badge/GitHub-rohitbirdawade007-181717?style=for-the-badge&logo=github)](https://github.com/rohitbirdawade007)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) — Piano MIDI data
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) — Multi-genre MIDI data
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer architecture
- [Music Transformer](https://arxiv.org/abs/1809.04281) — Music-specific Transformer insights
- [pretty_midi](https://craffel.github.io/pretty-midi/) — MIDI processing library

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ and 🎵 by [Rohit Birdawade](https://github.com/rohitbirdawade007)

</div>
