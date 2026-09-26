"""
CS 315 - Application Development and Emerging Technologies
Activity 3: GenAI App Using a Different Dataset

Movie Review Sentiment Analyzer
--------------------------------
Dataset : Rotten Tomatoes movie review dataset (Cornell "sentence polarity" set),
          loaded via the Hugging Face `datasets` library:
          https://huggingface.co/datasets/rotten_tomatoes
          Citation: Pang, B., & Lee, L. (2005). Seeing stars: Exploiting class
          relationships for sentiment categorization with respect to rating
          scales. Proceedings of ACL 2005.

GenAI   : Hugging Face Inference API (huggingface_hub.InferenceClient), used for:
            1) Sentiment classification of any review (custom or from the dataset)
            2) Short natural-language explanation / keyword extraction of a review
          using hosted models on the Hugging Face Hub.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

You need a free Hugging Face access token (https://huggingface.co/settings/tokens)
set as the HF_TOKEN environment variable, or entered in the sidebar at runtime.
"""

import os
import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import InferenceClient

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------------------------
# 2. LOAD AND CLEAN THE DATASET  (Step 3 of the activity)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Rotten Tomatoes dataset...")
def load_rotten_tomatoes():
    """
    Loads the Rotten Tomatoes / Cornell movie review polarity dataset from the
    Hugging Face Hub and returns a cleaned pandas DataFrame with columns:
        text  - the review text
        label - 0 (negative) or 1 (positive)
    """
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    df = pd.concat([train_df, test_df], ignore_index=True)

    # --- basic cleaning ---
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
    return df


@st.cache_data
def load_local_csv(path):
    """Optional: load a locally saved copy from data/movie_reviews.csv if present."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCAL_CSV = os.path.join(DATA_DIR, "movie_reviews.csv")

if os.path.exists(LOCAL_CSV):
    df = load_local_csv(LOCAL_CSV)
else:
    df = load_rotten_tomatoes()

# ----------------------------------------------------------------------------
# 3. GENAI INTEGRATION - HUGGING FACE INFERENCE API  (Step 4 of the activity)
# ----------------------------------------------------------------------------
def get_client(hf_token: str) -> InferenceClient:
    return InferenceClient(token=hf_token)


SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TEXT_GEN_MODEL = "google/flan-t5-base"


def analyze_sentiment(client: InferenceClient, text: str):
    """Call the Hugging Face Inference API for sentiment classification."""
    result = client.text_classification(text, model=SENTIMENT_MODEL)
    # result looks like: [{'label': 'POSITIVE', 'score': 0.998}, ...]
    top = max(result, key=lambda r: r["score"])
    return top["label"], top["score"]


def extract_keywords_summary(client: InferenceClient, text: str):
    """Use a generative model on the HF Inference API to summarize / pull out
    the key themes and keywords driving the review's sentiment."""
    prompt = (
        "Read this movie review and respond with: "
        "1) a one-sentence summary, 2) three keywords that best capture its tone.\n\n"
        f"Review: {text}"
    )
    response = client.text_generation(
        prompt, model=TEXT_GEN_MODEL, max_new_tokens=80
    )
    return response


# ----------------------------------------------------------------------------
# 4. SIDEBAR - API KEY + SETTINGS
# ----------------------------------------------------------------------------
st.sidebar.header("🔑 Hugging Face Settings")


def get_stored_token() -> str:
    """
    Look for the HF token in this order:
      1. Streamlit secrets (st.secrets["HF_TOKEN"]) - used on Streamlit
         Community Cloud, or a local .streamlit/secrets.toml file.
      2. OS environment variable HF_TOKEN - used when you `export HF_TOKEN=...`
         or load a .env file before running the app.
      3. Empty string - falls back to manual entry in the sidebar.
    """
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        # st.secrets raises if no secrets.toml exists at all - that's fine,
        # it just means we fall through to the environment variable check.
        pass
    return os.environ.get("HF_TOKEN", "")


stored_token = get_stored_token()

if stored_token:
    hf_token = stored_token
    st.sidebar.success("Hugging Face token loaded from secrets/environment ✅")
else:
    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help="Create a free token at https://huggingface.co/settings/tokens",
    )
    st.sidebar.caption(
        "Tip: for deployment, store this as a secret (`HF_TOKEN`) instead of "
        "typing it here every time. See README.md for setup options."
    )

st.sidebar.divider()
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"Total reviews loaded: **{len(df):,}**")
if "sentiment" in df.columns:
    st.sidebar.write(df["sentiment"].value_counts())
st.sidebar.caption(
    "Source: Rotten Tomatoes / Cornell movie review polarity dataset "
    "(Pang & Lee, 2005), via Hugging Face Datasets — "
    "https://huggingface.co/datasets/rotten_tomatoes"
)

# ----------------------------------------------------------------------------
# 5. MAIN UI  (Step 5 of the activity: Build the Streamlit Interface)
# ----------------------------------------------------------------------------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "A GenAI-powered app that explores the Rotten Tomatoes review dataset and "
    "uses the Hugging Face Inference API to analyze sentiment on demand."
)

tab1, tab2, tab3 = st.tabs(
    ["📚 Explore Dataset", "🤖 Analyze a Review", "📈 Sentiment Trends"]
)

# --- TAB 1: Explore dataset ---
with tab1:
    st.subheader("Browse the Rotten Tomatoes Reviews")
    sentiment_filter = st.selectbox(
        "Filter by sentiment", ["All", "Positive", "Negative"]
    )
    sample_size = st.slider("Number of reviews to show", 5, 50, 10)

    filtered = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]
    st.dataframe(filtered.sample(min(sample_size, len(filtered)))[["text", "sentiment"]])

# --- TAB 2: Analyze a custom or sampled review ---
with tab2:
    st.subheader("Run GenAI Sentiment Analysis")
    source = st.radio("Choose a review to analyze:", ["Write my own", "Pick from dataset"])

    if source == "Write my own":
        review_text = st.text_area(
            "Enter a movie review:",
            "This film was a stunning, emotional journey with brilliant acting.",
        )
    else:
        random_row = df.sample(1).iloc[0]
        review_text = random_row["text"]
        st.info(f"Sampled review (true label: {random_row['sentiment']}):")
        st.write(review_text)

    if st.button("Analyze Review", type="primary"):
        if not hf_token:
            st.error("Please enter your Hugging Face API token in the sidebar.")
        else:
            client = get_client(hf_token)
            with st.spinner("Calling Hugging Face Inference API..."):
                try:
                    label, score = analyze_sentiment(client, review_text)
                    st.success(f"**Predicted sentiment:** {label}  (confidence: {score:.2%})")

                    st.write("**GenAI summary & keywords:**")
                    summary = extract_keywords_summary(client, review_text)
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error calling the Hugging Face API: {e}")

# --- TAB 3: Visualize results  (Step 6 of the activity) ---
with tab3:
    st.subheader("Overall Sentiment Distribution")
    counts = df["sentiment"].value_counts()
    st.bar_chart(counts)

    st.subheader("Review Length by Sentiment")
    df["review_length"] = df["text"].str.split().apply(len)
    avg_len = df.groupby("sentiment")["review_length"].mean()
    st.bar_chart(avg_len)

    st.caption(
        "These charts summarize patterns across the Rotten Tomatoes dataset, "
        "such as the balance of positive vs. negative reviews and how review "
        "length relates to sentiment."
    )
