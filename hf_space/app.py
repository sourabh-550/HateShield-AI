import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import emoji

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SafeShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; }
    .hof-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .not-box {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

HF_REPO = "sourabh5500/hate-speech-muril"

# ── Preprocessor (inline — no src/ dependency) ────────────
class HinglishPreprocessor:
    def __init__(self):
        self._url_pattern     = re.compile(r'http[s]?://\S+')
        self._mention_pattern = re.compile(r'@\w+')
        self._hashtag_pattern = re.compile(r'#(\w+)')
        self._repeated_char   = re.compile(r'(.)\1{2,}')
        self._repeated_punct  = re.compile(r'([!?.]){2,}')
        self._whitespace      = re.compile(r'\s+')

    def clean(self, text: str):
        if not isinstance(text, str):
            return None
        text = self._url_pattern.sub(' ', text)
        text = self._mention_pattern.sub(' ', text)
        text = self._hashtag_pattern.sub(r'\1', text)
        text = emoji.demojize(text)
        text = text.lower()
        text = self._repeated_char.sub(r'\1\1', text)
        text = self._repeated_punct.sub(r'\1', text)
        text = self._whitespace.sub(' ', text).strip()
        return text if len(text) >= 3 else None

preprocessor = HinglishPreprocessor()

# ── Model Loading (cached — loads only once) ──────────────
@st.cache_resource
def load_model():
    with st.spinner("Loading MuRIL model... (first load takes ~30 seconds)"):
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        model     = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
        model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ── Prediction Function ───────────────────────────────────
def predict(text: str, threshold: float = 0.5):
    clean = preprocessor.clean(text)
    if not clean:
        return None

    inputs = tokenizer(
        clean,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    prob_not = probs[0].item()
    prob_hof = probs[1].item()

    if prob_hof >= threshold:
        label      = "HOF"
        confidence = prob_hof
    else:
        label      = "NOT"
        confidence = prob_not

    if confidence >= 0.65:
        routing = "🟢 Auto-Classify"
    elif confidence >= 0.60:
        routing = "🟡 Classify + Log"
    else:
        routing = "🔴 Human Review"

    return {
        "label"     : label,
        "confidence": confidence,
        "prob_not"  : prob_not,
        "prob_hof"  : prob_hof,
        "routing"   : routing
    }

# ── UI ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ SafeShield AI</p>', unsafe_allow_html=True)
st.markdown("**Multilingual Hate Speech Detection for Hinglish Text**")
st.caption("Fine-tuned MuRIL model | Macro F1: 0.7352 | 29k Hinglish samples")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "ℹ️ About"])

# ── Tab 1 ─────────────────────────────────────────────────
with tab1:
    st.markdown("### Analyze a Single Text")

    col1, col2, col3 = st.columns(3)
    example_text = ""
    with col1:
        if st.button("😊 Non-hate example"):
            example_text = "aaj ka din bahut acha tha yaar, bahut maza aaya"
    with col2:
        if st.button("😠 Hate example"):
            example_text = "tu bahut bura insaan hai, tujhse nafrat hai mujhe"
    with col3:
        if st.button("🤔 Borderline example"):
            example_text = "yaar ye log kuch bhi bolte hain, koi shame nahi"

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold = st.slider(
            "HOF Threshold",
            min_value=0.3, max_value=0.8,
            value=0.5, step=0.05,
            help="Lower = catch more hate (higher recall)"
        )
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Model**: MuRIL-base-cased
        - **Params**: 237M
        - **Test Macro-F1**: 0.7352
        - **Baseline F1**: 0.7138
        - **Dataset**: 29k Hinglish samples
        """)
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[🤗 Model on HF Hub](https://huggingface.co/sourabh5500/hate-speech-muril)")
        st.markdown("[💻 GitHub](https://github.com/sourabh-550)")

    user_text = st.text_area(
        "Enter text (Hinglish/English):",
        value=example_text,
        height=120,
        placeholder="Type Hinglish/English text here...",
        max_chars=512
    )
    st.caption(f"{len(user_text)}/512 characters")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Analyzing..."):
                result = predict(user_text, threshold)

            if result is None:
                st.error("Text too short after preprocessing.")
            else:
                if result['label'] == "HOF":
                    st.markdown(f"""
                    <div class="hof-box">
                        <h3>🚨 Hate/Offensive Content Detected</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="not-box">
                        <h3>✅ Non-Hate Content</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Label", result['label'])
                c2.metric("Confidence", f"{result['confidence']:.1%}")
                c3.metric("NOT Score", f"{result['prob_not']:.1%}")
                c4.metric("HOF Score", f"{result['prob_hof']:.1%}")

                st.info(f"**Routing:** {result['routing']}")

                st.markdown("**Probability Distribution:**")
                st.progress(result['prob_not'], text=f"NOT: {result['prob_not']:.1%}")
                st.progress(result['prob_hof'], text=f"HOF: {result['prob_hof']:.1%}")

# ── Tab 2 ─────────────────────────────────────────────────
with tab2:
    st.markdown("### Analyze Multiple Texts")
    st.caption("One text per line, max 20 texts")

    batch_input = st.text_area(
        "Enter texts (one per line):",
        height=200,
        placeholder="aaj bahut acha din hai\ntujhse nafrat hai\n..."
    )

    if st.button("🔍 Analyze All", type="primary", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]

        if not texts:
            st.warning("Enter at least one text!")
        elif len(texts) > 20:
            st.error("Max 20 texts allowed!")
        else:
            results = []
            progress = st.progress(0)
            for i, text in enumerate(texts):
                r = predict(text, threshold)
                if r:
                    results.append({
                        "Text"      : text[:60] + ("..." if len(text) > 60 else ""),
                        "Label"     : r['label'],
                        "Confidence": f"{r['confidence']:.1%}",
                        "NOT"       : f"{r['prob_not']:.1%}",
                        "HOF"       : f"{r['prob_hof']:.1%}",
                        "Routing"   : r['routing']
                    })
                progress.progress((i+1)/len(texts))

            if results:
                import pandas as pd
                hof_count = sum(1 for r in results if r['Label'] == 'HOF')

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(results))
                c2.metric("🚨 HOF", hof_count)
                c3.metric("✅ NOT", len(results) - hof_count)

                st.dataframe(pd.DataFrame(results),
                             use_container_width=True, hide_index=True)

# ── Tab 3 ─────────────────────────────────────────────────
with tab3:
    st.markdown("### About SafeShield AI")
    st.markdown("""
    **SafeShield AI** detects hate speech in Hinglish (Hindi-English code-mixed) text
    using a fine-tuned MuRIL transformer model.

    #### 📊 Performance
    | Model | Macro F1 |
    |-------|---------|
    | TF-IDF + LR (Baseline) | 0.7138 |
    | **MuRIL Fine-tuned** | **0.7352** |

    #### 🔄 Routing System
    | Confidence | Decision |
    |-----------|----------|
    | ≥ 0.65 | 🟢 Auto-Classify |
    | 0.60-0.65 | 🟡 Classify + Log |
    | < 0.60 | 🔴 Human Review |

    #### ⚠️ Limitations
    - HOF recall: 66.7% (misses ~33% of hate speech)
    - Short texts (<50 chars) have higher false positive rate
    - Best for Hinglish/English — not pure Hindi (Devanagari)

    #### 👨‍💻 Built by
    **Sourabh Saxena** — B.Tech CSE (AI/ML), IMS Engineering College
    """)