# 🎬 Movie Review Sentiment Analyzer
**CS 315 – Application Development and Emerging Technologies — Activity 3**

A GenAI-powered Streamlit app that analyzes movie review sentiment using the
Rotten Tomatoes dataset and the Hugging Face Inference API.

---

## 1. App Idea
- **Dataset:** Rotten Tomatoes movie reviews (Cornell "sentence polarity" dataset),
  loaded directly from the Hugging Face Hub.
- **Purpose:** Sentiment analysis and keyword/summary extraction on movie
  reviews — both reviews sampled from the dataset and reviews the user types
  in themselves.

## 2. Environment Setup
Project structure:
```
movie-review-analyzer/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your Hugging Face token
├── README.md
└── data/                # Optional: place a local movie_reviews.csv here
                          # to override the auto-downloaded dataset
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Get a **free Hugging Face API token**: https://huggingface.co/settings/tokens
(select "Read" access). The app looks for it in this order, so use whichever
fits how you're running it:

1. **Streamlit secrets** (recommended) — copy
   `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set:
   ```toml
   HF_TOKEN = "hf_your_actual_token_here"
   ```
   This is also exactly what you'll paste into **Settings → Secrets** when
   you deploy to Streamlit Community Cloud (see Step 8 below).
2. **Environment variable** — export it before running the app:
   ```bash
   export HF_TOKEN=hf_your_actual_token_here
   streamlit run app.py
   ```
   (or copy `.env.example` to `.env`, fill it in, and load it with a tool
   like `python-dotenv` or your shell's `source .env` before running).
3. **Manual entry** — if neither of the above is set, the app shows a
   password-style input box in the sidebar where you can paste the token
   for that session only.

`.env` and `.streamlit/secrets.toml` are both listed in `.gitignore` so you
never accidentally commit your real token to GitHub.

## 3. Load and Clean the Dataset
`app.py` uses the `datasets` library to pull the Rotten Tomatoes dataset
straight from the Hugging Face Hub (no manual CSV download needed):
```python
from datasets import load_dataset
ds = load_dataset("rotten_tomatoes")
```
It's then converted to a pandas DataFrame, cleaned (nulls dropped, whitespace
stripped, duplicates removed), and labeled `Positive` / `Negative`.

If you'd rather work from your own CSV export, drop a file named
`data/movie_reviews.csv` (with a `text` column) into the `data/` folder —
the app will automatically use it instead.

## 4. Integrate GenAI for Analysis
GenAI tasks run through the **Hugging Face Inference API**
(`huggingface_hub.InferenceClient`), not OpenAI:
- **Sentiment classification** — `distilbert-base-uncased-finetuned-sst-2-english`
- **Summary / keyword extraction** — `google/flan-t5-base` (generative model)

## 5. Streamlit Interface
Three tabs:
- **Explore Dataset** — filter and browse sample reviews
- **Analyze a Review** — type your own review or sample one from the dataset,
  then run it through the Hugging Face models
- **Sentiment Trends** — charts summarizing the dataset

## 6. Visualize the Results
Built with Streamlit's native `st.bar_chart` (Positive vs. Negative counts,
and average review length by sentiment).

## 7. Test and Iterate
Try it with:
- Short vs. long reviews
- Sarcastic or mixed-sentiment reviews (a good stress test for the model)
- Reviews sampled directly from the dataset, to compare the model's
  prediction against the dataset's true label

Use what you learn to refine prompts to the generative model or swap in a
different Hugging Face model.

## 8. Deploy Your App
1. Push this project to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) (Streamlit Community
   Cloud) and sign in with GitHub.
3. Click **New app**, select the repo/branch and `app.py` as the entry point.
4. Under **Advanced settings → Secrets**, paste the same `HF_TOKEN = "..."`
   line shown in `.streamlit/secrets.toml.example` above.
5. Deploy — Streamlit Cloud will install `requirements.txt` automatically.

## 9. Next Goals
- Add filters by genre or release year (requires a richer dataset with metadata)
- Add a chatbot tab (using a Hugging Face conversational model) that answers
  questions about the dataset, e.g. "What's a common theme in negative reviews?"

---

## 📖 Dataset Citation
This app uses the **Rotten Tomatoes movie review dataset**, hosted on Hugging
Face Datasets: https://huggingface.co/datasets/rotten_tomatoes

> Pang, B., & Lee, L. (2005). *Seeing stars: Exploiting class relationships
> for sentiment categorization with respect to rating scales.* Proceedings of
> the 43rd Annual Meeting of the Association for Computational Linguistics
> (ACL 2005).

## 🤖 GenAI API Used
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
  via the `huggingface_hub` Python library — models:
  `distilbert-base-uncased-finetuned-sst-2-english` and `google/flan-t5-base`.
