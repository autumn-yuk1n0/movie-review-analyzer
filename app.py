"""
CS 315 - Application Development and Emerging Technologies
Activity 3: GenAI App Using a Different Dataset

Movie Review Sentiment Analyzer
--------------------------------
Dataset : Rotten Tomatoes critic reviews (with movie titles), a cleaned-up
          Hugging Face repackaging of the Kaggle "Rotten Tomatoes Movies and
          Critic Reviews" dataset, loaded via the Hugging Face `datasets`
          library: https://huggingface.co/datasets/frankier/processed_multiscale_rt_critics
          Original source: Stefano Leone, "Rotten Tomatoes Movies and Critic
          Reviews Dataset," Kaggle, scraped from rottentomatoes.com (2020-10-31).

GenAI   : Hugging Face Inference API (huggingface_hub.InferenceClient), used for:
            1) Sentiment classification of any review (custom or from the dataset)
            2) Short natural-language explanation / keyword extraction of a review
            3) An open-ended chatbot
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
@st.cache_data(show_spinner="Loading Rotten Tomatoes critic reviews (with movie titles)...")
def load_rotten_tomatoes():
    """
    Loads a cleaned, titled Rotten Tomatoes critic reviews dataset from the
    Hugging Face Hub and returns a pandas DataFrame with columns:
        movie_title - the movie being reviewed
        critic_name - the critic who wrote the review
        text        - the review text
        sentiment   - "Positive" or "Negative", derived from the review's
                      normalized score (label / scale_points >= 0.5 -> Positive)

    Source: frankier/processed_multiscale_rt_critics on the Hugging Face Hub,
    itself a cleaned-up repackaging of Stefano Leone's "Rotten Tomatoes Movies
    and Critic Reviews Dataset" on Kaggle (scraped from rottentomatoes.com).
    """
    ds = load_dataset("frankier/processed_multiscale_rt_critics")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    df = pd.concat([train_df, test_df], ignore_index=True)

    # --- basic cleaning ---
    df = df.rename(columns={"review_content": "text"})
    df = df.dropna(subset=["text", "movie_title", "label", "scale_points"])
    df["text"] = df["text"].astype(str).str.strip()
    df["movie_title"] = df["movie_title"].astype(str).str.strip()
    df = df[(df["text"].str.len() > 0) & (df["movie_title"].str.len() > 0)]
    df = df[df["scale_points"] > 0]  # avoid divide-by-zero
    df = df.drop_duplicates(subset=["text", "movie_title"]).reset_index(drop=True)

    # Derive a Positive/Negative sentiment label from the normalized score
    df["normalized_score"] = df["label"] / df["scale_points"]
    df["sentiment"] = df["normalized_score"].apply(
        lambda x: "Positive" if x >= 0.5 else "Negative"
    )

    # This dataset has ~670k rows - sample down to a manageable size so the
    # app stays fast and light on Streamlit Cloud's free tier.
    if len(df) > 20000:
        df = df.sample(20000, random_state=42).reset_index(drop=True)

    keep_cols = ["movie_title", "critic_name", "text", "sentiment"]
    return df[keep_cols]


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
    # provider="hf-inference" routes to Hugging Face's own free serverless
    # inference, instead of paid third-party partners (Together, Novita, etc.).
    # It's rate-limited rather than billed against your $ credit balance,
    # which suits an app meant to be asked lots of questions for free.
    return InferenceClient(token=hf_token, provider="hf-inference")


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


CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def ask_chatbot(client: InferenceClient, messages: list):
    """
    Send the running conversation to a Hugging Face chat model and return its
    reply. `messages` is a list of {"role": "user"/"assistant", "content": str}
    dicts, following the same format used by chat-style LLM APIs.
    """
    completion = client.chat_completion(
        messages=messages,
        model=CHAT_MODEL,
        max_tokens=300,
    )
    return completion.choices[0].message.content


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
    "Source: Rotten Tomatoes critic reviews, via the Hugging Face dataset "
    "frankier/processed_multiscale_rt_critics — a cleaned repackaging of "
    "Stefano Leone's \"Rotten Tomatoes Movies and Critic Reviews Dataset\" "
    "on Kaggle (scraped from rottentomatoes.com)."
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
    ["📚 Explore Dataset", "🤖 Analyze a Review", "💬 Chatbot"]
)

# --- TAB 1: Explore dataset ---
with tab1:
    st.subheader("Browse the Rotten Tomatoes Critic Reviews")
    col_a, col_b = st.columns(2)
    with col_a:
        sentiment_filter = st.selectbox(
            "Filter by sentiment", ["All", "Positive", "Negative"]
        )
    with col_b:
        title_search = st.text_input("Search by movie title (optional)", "")

    sample_size = st.slider("Number of reviews to show", 5, 50, 10)

    filtered = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]
    if title_search.strip():
        filtered = filtered[
            filtered["movie_title"].str.contains(title_search.strip(), case=False, na=False)
        ]

    if filtered.empty:
        st.warning("No reviews match that title search. Try a different movie name.")
    else:
        st.dataframe(
            filtered.sample(min(sample_size, len(filtered)))[
                ["movie_title", "critic_name", "text", "sentiment"]
            ]
        )

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
        st.info(
            f"Sampled review for **{random_row['movie_title']}** "
            f"(critic: {random_row['critic_name']}, true sentiment: {random_row['sentiment']}):"
        )
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

# --- TAB 3: Chatbot - ask anything about the dataset or movies in general ---
with tab3:
    st.subheader("Ask the Chatbot Anything")
    st.caption(
        "Powered by the Hugging Face Inference API. Ask about the dataset, "
        "sentiment analysis, movies in general, or anything else."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show the conversation so far
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Type your question here...")

    if user_question:
        if not hf_token:
            st.error("Please enter your Hugging Face API token in the sidebar.")
        else:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.write(user_question)

            client = get_client(hf_token)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = ask_chatbot(client, st.session_state.chat_history)
                        st.write(reply)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as e:
                        st.error(f"Error calling the Hugging Face API: {e}")

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
