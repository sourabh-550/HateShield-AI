# app/streamlit_app.py
import streamlit as st
import requests
import json

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SafeShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
    }
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
    .review-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────
def predict_single(text: str, threshold: float = 0.5):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "threshold": threshold},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "❌ API not running. Start it with: uvicorn api.main:app --port 8000"
    except Exception as e:
        return None, str(e)


def predict_batch(texts: list, threshold: float = 0.5):
    try:
        resp = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": texts, "threshold": threshold},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "❌ API not running."
    except Exception as e:
        return None, str(e)


def get_routing_info(routing: str):
    if routing == "auto_classify":
        return "🟢 Auto-Classify", "Confidence ≥ 0.65 — reliable prediction"
    elif routing == "classify_log":
        return "🟡 Classify + Log", "Confidence 0.60-0.65 — log for review"
    else:
        return "🔴 Human Review", "Confidence < 0.60 — needs human review"


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/🛡️-SafeShield_AI-blue", width=200)
    st.markdown("### ⚙️ Settings")

    threshold = st.slider(
        "HOF Classification Threshold",
        min_value=0.3,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Lower = catch more hate speech (higher recall). Higher = more conservative (higher precision)."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Model**: MuRIL-base-cased
    - **Fine-tuned on**: Hinglish dataset
    - **Test Macro-F1**: 0.7352
    - **Baseline F1**: 0.7138
    - **Params**: 237M
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[🤗 HuggingFace Model](https://huggingface.co/sourabh5500/hate-speech-muril)")
    st.markdown("[💻 GitHub Repo](https://github.com/sourabh-550)")

    # API Health
    st.markdown("---")
    st.markdown("### 🔌 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online\nDevice: {health['device']}")
    except:
        st.error("❌ API Offline")


# ── Main Header ───────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ SafeShield AI</p>', unsafe_allow_html=True)
st.markdown("**Multilingual Hate Speech Detection for Hinglish Text**")
st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "ℹ️ About"])


# ── Tab 1: Single Analysis ────────────────────────────────
with tab1:
    st.markdown("### Analyze a Single Text")

    # Quick examples
    st.markdown("**Try an example:**")
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

    user_text = st.text_area(
        "Enter text (Hinglish/English):",
        value=example_text,
        height=120,
        placeholder="Type or paste Hinglish/English text here...",
        max_chars=512
    )

    char_count = len(user_text)
    st.caption(f"{char_count}/512 characters")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing..."):
                result, error = predict_single(user_text, threshold)

            if error:
                st.error(error)
            else:
                label      = result['label']
                confidence = result['confidence']
                scores     = result['scores']
                routing    = result['routing']

                # Result display
                if label == "HOF":
                    st.markdown(f"""
                    <div class="hof-box">
                        <h3>🚨 Hate/Offensive Content Detected</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="not-box">
                        <h3>✅ Non-Hate Content</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Label", label)
                col2.metric("Confidence", f"{confidence:.1%}")
                col3.metric("NOT Score", f"{scores['NOT']:.1%}")
                col4.metric("HOF Score", f"{scores['HOF']:.1%}")

                # Routing
                routing_label, routing_desc = get_routing_info(routing)
                st.info(f"**Routing Decision:** {routing_label}\n\n{routing_desc}")

                # Confidence bar
                st.markdown("**Probability Distribution:**")
                st.progress(scores['NOT'], text=f"NOT: {scores['NOT']:.1%}")
                st.progress(scores['HOF'], text=f"HOF: {scores['HOF']:.1%}")


# ── Tab 2: Batch Analysis ─────────────────────────────────
with tab2:
    st.markdown("### Analyze Multiple Texts")
    st.caption("Enter one text per line (max 32 texts)")

    batch_input = st.text_area(
        "Enter texts (one per line):",
        height=200,
        placeholder="yaar aaj bahut acha din hai\ntujhse nafrat hai mujhe\nModi ji ne acha kaam kiya\n..."
    )

    if st.button("🔍 Analyze All", type="primary", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]

        if not texts:
            st.warning("Please enter at least one text!")
        elif len(texts) > 32:
            st.error("Maximum 32 texts allowed per batch!")
        else:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result, error = predict_batch(texts, threshold)

            if error:
                st.error(error)
            else:
                # Summary metrics
                st.markdown("### 📊 Batch Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", result['total'])
                col2.metric("🚨 HOF", result['hof_count'])
                col3.metric("✅ NOT", result['not_count'])
                col4.metric("👁️ Need Review", result['review_count'])

                # Results table
                st.markdown("### Results")
                import pandas as pd
                rows = []
                for r in result['results']:
                    routing_label, _ = get_routing_info(r['routing'])
                    rows.append({
                        "Text": r['text'][:60] + ("..." if len(r['text']) > 60 else ""),
                        "Label": r['label'],
                        "Confidence": f"{r['confidence']:.1%}",
                        "NOT": f"{r['scores']['NOT']:.1%}",
                        "HOF": f"{r['scores']['HOF']:.1%}",
                        "Routing": routing_label
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab 3: About ──────────────────────────────────────────
with tab3:
    st.markdown("### About SafeShield AI")
    st.markdown("""
    **SafeShield AI** is a multilingual hate speech detection system fine-tuned 
    on MuRIL (Multilingual Representations for Indian Languages) for Hinglish 
    (Hindi-English code-mixed) text.

    #### 🏗️ Architecture
    - **Base Model**: google/muril-base-cased (237M params)
    - **Task**: Binary classification (NOT / HOF)
    - **Input**: Hinglish/English text (max 128 tokens)

    #### 📊 Performance
    | Model | Macro F1 |
    |-------|---------|
    | TF-IDF + LR (Baseline) | 0.7138 |
    | **MuRIL Fine-tuned** | **0.7352** |

    #### ⚠️ Known Limitations
    - HOF recall is 66.7% — model misses ~33% of hate speech
    - Short texts (<50 chars) have higher false positive rate
    - Best above 0.65 confidence (87-96% accurate)

    #### 🎯 Routing System
    | Confidence | Action |
    |-----------|--------|
    | ≥ 0.65 | Auto-classify |
    | 0.60-0.65 | Classify + Log |
    | < 0.60 | Human Review |
    """)