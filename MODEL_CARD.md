---
language:
- hi
- en
tags:
- hate-speech-detection
- text-classification
- hinglish
- code-mixed
- muril
- nlp
- responsible-ai
datasets:
- code-mixed-hinglish-hate-speech
metrics:
- f1
- accuracy
model-index:
- name: hate-speech-muril
  results:
  - task:
      type: text-classification
    metrics:
    - type: f1
      value: 0.7352
      name: Macro F1
    - type: accuracy
      value: 0.7390
      name: Accuracy
---

# SafeShield AI — Multilingual Hate Speech Detection

[![Model](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/sourabh5500/hate-speech-muril)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-orange)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Model Description

**SafeShield AI** is a fine-tuned [MuRIL (Multilingual Representations for Indian Languages)](https://huggingface.co/google/muril-base-cased) model for detecting hate speech and offensive content in **Hinglish** (Hindi-English code-mixed) text.

Most hate speech classifiers are trained on monolingual English, but a significant volume of online abuse in India occurs in Hinglish — which standard tokenizers and models handle poorly. This model addresses that gap.

### Model Architecture
Input Text (Hinglish/English)

↓

MuRIL Tokenizer (vocab: 197k tokens, covers Devanagari + Latin)

↓

MuRIL-base-cased (12 layers, 768 hidden, 12 heads, 237M params)

↓

[CLS] token embedding (768-dim)

↓

Dropout (0.1) + Linear(768 → 2)

↓

Softmax → [P(NOT), P(HOF)]

### Labels
| Label | ID | Meaning |
|-------|----|---------|
| NOT | 0 | Non-hate, non-offensive content |
| HOF | 1 | Hate speech or Offensive content |

---

## Training Details

### Dataset
- **Source**: Code-Mixed Hinglish Hate Speech Detection Dataset
- **Size**: 29,533 samples (after cleaning)
- **Split**: 70% train / 10% val / 20% test
- **Class balance**: NOT=53.7% / HOF=46.3% (nearly balanced, 1.16x ratio)
- **Languages**: Hinglish, English, mixed

### Preprocessing Pipeline
- URL removal
- @mention removal
- Hashtag normalization (#BJP → BJP)
- Emoji demojization (😂 → :face_with_tears_of_joy:)
- Repeated character normalization (sooo → soo)
- Repeated punctuation normalization (!!! → !)
- Lowercasing

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Base model | google/muril-base-cased |
| Max sequence length | 128 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Epochs | 4 (early stop at 3) |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Scheduler | Linear warmup + decay |
| Loss | CrossEntropyLoss (balanced weights) |

### Training Infrastructure
- Platform: Kaggle (free tier)
- GPU: Tesla T4 x2
- Training time: ~25 minutes

---

## Evaluation Results

### Test Set Performance (5,907 samples)

| Metric | NOT | HOF | Macro Avg |
|--------|-----|-----|-----------|
| Precision | 0.7351 | 0.7444 | 0.7397 |
| Recall | 0.8015 | 0.6668 | 0.7342 |
| F1-Score | 0.7669 | 0.6974 | **0.7352** |
| Support | 3,164 | 2,743 | 5,907 |

**Overall Accuracy: 73.90%**

### Comparison with Baseline

| Model | Macro F1 | Notes |
|-------|---------|-------|
| TF-IDF + Logistic Regression | 0.7138 | Baseline |
| **MuRIL Fine-tuned (ours)** | **0.7352** | +3.0% relative gain |

### Confusion Matrix
Predicted
            NOT    HOF
Actual  NOT  [ 2513    651 ]
HOF  [  926   1817 ]

---

## Error Analysis & Known Limitations

### Error Breakdown
| Type | Count | Rate | Description |
|------|-------|------|-------------|
| True Positive | 1,817 | - | Correctly caught hate |
| True Negative | 2,513 | - | Correctly passed clean |
| False Negative | 926 | 33.76% | **Missed hate speech** ← most dangerous |
| False Positive | 651 | 20.58% | Over-flagged clean content |

### Known Failure Modes

**1. Implicit/Subtle Hate Speech**
The model struggles with hate speech that doesn't use explicit slurs or offensive vocabulary — sarcasm, dog whistles, and context-dependent toxicity are frequently missed.

**2. Very Short Texts**
Texts under 50 characters have a 41.8% false positive rate. With minimal context, the model over-flags based on surface patterns.

**3. Non-Hinglish Languages**
Croatian, Serbian, and other non-Hinglish texts are sometimes misclassified as hate speech despite MuRIL's multilingual training.

**4. Medium-Length Hate Speech (101-150 chars)**
The worst false negative rate (48.7%) occurs in the 101-150 character range — nearly half of hate speech in this length is missed.

**5. Confidence Calibration**
- Below 0.60 confidence: only 56-66% accurate (borderline territory)
- Above 0.65 confidence: 87-96% accurate (reliable zone)

### Recommended Production Thresholds
```python
if confidence >= 0.65:
    # Auto-classify — reliable (87-96% accurate)
elif 0.60 <= confidence < 0.65:
    # Auto-classify with logging
else:  # confidence < 0.60
    # Route to human review queue
```

---

## How to Use

### Direct Inference
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("sourabh5500/hate-speech-muril")
model = AutoModelForSequenceClassification.from_pretrained(
    "sourabh5500/hate-speech-muril"
)
model.eval()

def predict(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    label_id = probs.argmax().item()
    return {
        "label": model.config.id2label[label_id],
        "confidence": probs[label_id].item(),
        "scores": {
            "NOT": probs[0].item(),
            "HOF": probs[1].item()
        }
    }

# Test
print(predict("yaar tu bahut bura insaan hai"))
print(predict("aaj ka din bahut acha tha"))
```

### Using the API
```bash
curl -X POST "https://your-api-url/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "yaar tu bahut bura insaan hai"}'
```

---

## Bias & Fairness Considerations

1. **Language bias**: Trained primarily on Hinglish/English. Performance degrades on pure Hindi (Devanagari script) or other Indian languages.
2. **Topic bias**: Dataset sourced from Twitter/social media — may not generalize to other platforms (WhatsApp, YouTube comments).
3. **Annotator bias**: Human annotations inherently reflect annotator demographics and perspectives. Hate speech labeling is subjective.
4. **Recency bias**: Training data has a cutoff date. New slang and emerging hate speech patterns won't be captured.
5. **Class bias**: Despite near-balanced dataset (1.16x), HOF recall (0.667) is lower than NOT recall (0.802).

---

## Intended Use

✅ **Appropriate uses:**
- Content moderation assistance (human-in-the-loop)
- Research on Hinglish hate speech
- First-pass filtering with human review for borderline cases
- Academic study of multilingual toxicity detection

❌ **Inappropriate uses:**
- Fully automated moderation without human oversight
- Legal or punitive decisions based solely on model output
- Languages other than Hinglish/English
- Real-time high-stakes moderation without threshold tuning

---

## Project Structure
SafeShield AI/

├── src/

│   ├── data/

│   │   ├── preprocessor.py

│   │   └── loader.py

│   └── models/

│       └── baseline.py

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_preprocessing.ipynb

│   ├── 03_baseline.ipynb

│   ├── 04_muril_finetune.ipynb

│   ├── 05_muril_improved.ipynb

│   └── 06_error_analysis.ipynb

├── api/

├── app/

├── results/

└── MODEL_CARD.md
---

## Citation

```bibtex
@misc{saxena2025safeshield,
  author = {Sourabh Saxena},
  title = {SafeShield AI: Multilingual Hate Speech Detection for Hinglish Text},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/sourabh5500/hate-speech-muril}
}
```

---

## Author

**Sourabh Saxena**
B.Tech CSE (AI/ML) — IMS Engineering College
GitHub: [sourabh-550](https://github.com/sourabh-550)
HuggingFace: [sourabh5500](https://huggingface.co/sourabh5500)

---

*This model is intended as a research tool. Always use human oversight for content moderation decisions. The author is not responsible for misuse of this model.*