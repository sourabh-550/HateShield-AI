<div align="center">

# 🛡️ SafeShield AI
### Multilingual Hate Speech Detection for Hinglish Text

[![Model](https://img.shields.io/badge/🤗_Model-HuggingFace-yellow)](https://huggingface.co/sourabh5500/hate-speech-muril)
[![Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://huggingface.co/spaces/sourabh5500/safeshield-ai)
[![GitHub](https://img.shields.io/badge/GitHub-sourabh--550-black)](https://github.com/sourabh-550)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Fine-tuned MuRIL transformer for hate speech detection in code-mixed Hinglish text*

</div>

---

## 📌 Overview

Most hate speech classifiers are trained on monolingual English — but a significant volume of online abuse in India occurs in **Hinglish** (Hindi-English code-mixed text), which standard models handle poorly.

**SafeShield AI** addresses this gap by fine-tuning Google's **MuRIL** (Multilingual Representations for Indian Languages) on a 29k sample Hinglish dataset, achieving **0.7352 Macro-F1** — a 3% relative gain over the TF-IDF baseline.

---

## 🏗️ Architecture
Raw Hinglish Text

↓

Preprocessing Pipeline

(URLs, mentions, emojis, normalization)

↓

MuRIL Tokenizer (197k vocab)

↓

MuRIL-base-cased (237M params, 12 layers)

↓

[CLS] embedding → Dropout → Linear(768→2)

   ↓

Softmax → P(NOT), P(HOF)

↓

Confidence-based Routing System
---

## 📊 Results

### Model Comparison

| Model | Macro F1 | ROC-AUC | Notes |
|-------|---------|---------|-------|
| TF-IDF + Logistic Regression | 0.7138 | 0.7933 | Baseline |
| TF-IDF + LinearSVC | 0.7048 | - | Worse than LR |
| **MuRIL Fine-tuned ✅** | **0.7352** | - | +3.0% relative gain |

### Per-Class Performance (Test Set — 5,907 samples)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| NOT (Non-hate) | 0.7351 | 0.8015 | 0.7669 |
| HOF (Hate/Offensive) | 0.7444 | 0.6668 | 0.6974 |
| **Macro Avg** | **0.7397** | **0.7342** | **0.7352** |

### Error Analysis

| Error Type | Count | Rate |
|-----------|-------|------|
| True Positives | 1,817 | - |
| True Negatives | 2,513 | - |
| False Negatives | 926 | 33.76% |
| False Positives | 651 | 20.58% |

### Confidence Calibration

| Confidence | Accuracy | Recommendation |
|-----------|----------|----------------|
| ≥ 0.65 | 87-96% | 🟢 Auto-classify |
| 0.60-0.65 | 76% | 🟡 Classify + Log |
| < 0.60 | 56-66% | 🔴 Human Review |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/sourabh-550/hate-speech-detection.git
cd hate-speech-detection
```

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Run FastAPI server
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Run Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

### 5. Or use the model directly
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("sourabh5500/hate-speech-muril")
model = AutoModelForSequenceClassification.from_pretrained("sourabh5500/hate-speech-muril")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    label = model.config.id2label[probs.argmax().item()]
    return {"label": label, "confidence": probs.max().item()}

print(predict("yaar tu bahut bura insaan hai"))
# {'label': 'HOF', 'confidence': 0.71}
```

---

## 🔌 API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "yaar tu bahut bura insaan hai"}'

# Response
{
  "label": "HOF",
  "confidence": 0.71,
  "scores": {"NOT": 0.29, "HOF": 0.71},
  "routing": "auto_classify"
}
```

---

## 📁 Project Structure
hate-speech-detection/

│

├── notebooks/                    # Kaggle training notebooks

│   ├── 01_eda.ipynb             # Exploratory data analysis

│   ├── 02_preprocessing.ipynb   # Text cleaning pipeline

│   ├── 03_baseline.ipynb        # TF-IDF + LR baseline

│   ├── 04_muril_finetune.ipynb  # MuRIL fine-tuning (GPU)

│   ├── 05_muril_improved.ipynb  # Round 2 with Focal Loss

│   └── 06_error_analysis.ipynb  # Error analysis + bias

│

├── src/                          # Core Python modules

│   ├── config.py                # Central configuration

│   ├── data/

│   │   ├── preprocessor.py      # Hinglish text cleaning

│   │   └── loader.py            # Data loading + splits

│   └── models/

│       └── baseline.py          # TF-IDF + LR baseline

│

├── api/                          # FastAPI inference server

│   ├── main.py                  # API endpoints

│   ├── model_loader.py          # Singleton model loader

│   └── schemas.py               # Pydantic schemas

│

├── app/                          # Streamlit demo UI

│   └── streamlit_app.py

│

├── hf_space/                     # HuggingFace Spaces deployment

│   ├── app.py

│   └── requirements.txt

│

├── results/                      # Experiment results + plots

│   ├── README.md                # Results tracker

│   └── *.png                    # Training curves, confusion matrices

│

├── MODEL_CARD.md                 # Model documentation

├── Dockerfile                    # Container deployment

└── requirements.txt

---

## 🧠 Key Technical Decisions

**Why MuRIL over mBERT?**
MuRIL was specifically trained on 17 Indian languages including transliterated text — making it significantly better at understanding Hinglish than standard multilingual BERT.

**Why not remove emojis?**
Emojis carry sentiment signal in social media text. Instead of removing them, we convert to text descriptions (😂 → `:face_with_tears_of_joy:`) so the tokenizer can process them meaningfully.

**Why confidence-based routing?**
Calibration analysis showed the model is 87-96% accurate above 0.65 confidence but only 56% accurate below 0.60 — barely better than random. A three-tier routing system reduces human review workload by ~66% while maintaining safety.

**Why Focal Loss in Round 2?**
Standard CrossEntropy treats all examples equally. Focal Loss down-weights easy examples and focuses training on hard misclassified cases — specifically targeting HOF recall improvement.

---

## ⚠️ Known Limitations

1. **HOF recall: 66.7%** — model misses ~33% of hate speech
2. **Short texts (<50 chars)** — 41.8% false positive rate
3. **Medium texts (101-150 chars)** — worst FN rate (48.7%)
4. **Non-Hinglish languages** — Croatian/Serbian sometimes misclassified
5. **Implicit hate speech** — sarcasm and dog whistles frequently missed

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Model | MuRIL-base-cased (Google) |
| Framework | PyTorch, HuggingFace Transformers |
| Training | Kaggle (Tesla T4 x2 GPU) |
| Baseline | Scikit-learn (TF-IDF + LR) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Model Hosting | HuggingFace Hub |
| Version Control | Git + GitHub |

---

## 📈 MLOps Pipeline
Data Collection → EDA → Preprocessing → Baseline

↓

MuRIL Fine-tuning (Kaggle GPU)

↓

Error Analysis + Calibration

↓

Model Card Documentation

↓

HuggingFace Hub (Model Hosting)

↓

FastAPI (Inference Server)

↓

Streamlit (Demo UI)

↓

Docker (Containerization)

---

## 👨‍💻 Author

**Sourabh Saxena**
B.Tech CSE (AI/ML) — IMS Engineering College
- GitHub: [sourabh-550](https://github.com/sourabh-550)
- HuggingFace: [sourabh5500](https://huggingface.co/sourabh5500)

**Viabhav Sharma**
B.Tech CSE (AI/ML) — IMS Engineering College
- GitHub: [7vaibhav31](https://github.com/7vaibhav31)

**Aditya Chaudhary**
B.Tech CSE (AI/ML) — IMS Engineering College
- GitHub: [adityachaudhary0](https://github.com/adityachaudhary0)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>