# 🎬 Movie Review Sentiment Analyzer
**CS 315 – Application Development and Emerging Technologies — Activity 3**

A GenAI-powered Streamlit app that analyzes movie review sentiment using
real, titled Rotten Tomatoes critic reviews and the Hugging Face Inference API.

---

## 1. App Idea
- **Dataset:** Rotten Tomatoes critic reviews — including movie title, critic
  name, and review score — loaded directly from the Hugging Face Hub.
- **Purpose:** Sentiment analysis and keyword/summary extraction on movie
  reviews — both reviews sampled from the dataset (with their real movie
  titles shown) and reviews the user types in themselves.

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
`app.py` uses the `datasets` library to pull a titled Rotten Tomatoes critic
reviews dataset straight from the Hugging Face Hub (no manual CSV download
needed):
```python
from datasets import load_dataset
ds = load_dataset("frankier/processed_multiscale_rt_critics")
```
It's then converted to a pandas DataFrame, cleaned (nulls dropped, whitespace
stripped, duplicates removed), and given a `Positive`/`Negative` sentiment
label derived from each review's normalized score. The full dataset has
~670,000 rows, so the app samples down to 20,000 for speed on Streamlit
Cloud's free tier — this is easy to adjust in `load_rotten_tomatoes()`.

If you'd rather work from your own CSV export, drop a file named
`data/movie_reviews.csv` (with `text` and `movie_title` columns) into the
`data/` folder — the app will automatically use it instead.

## 4. Integrate GenAI for Analysis
GenAI tasks run through the **Hugging Face Inference API**
(`huggingface_hub.InferenceClient`), not OpenAI:
- **Sentiment classification** — `distilbert-base-uncased-finetuned-sst-2-english`
- **Summary / keyword extraction** — `google/flan-t5-base` (generative model)

## 5. Streamlit Interface
Three tabs:
- **Explore Dataset** — filter by sentiment, search by movie title, and
  browse sample reviews (with movie title and critic name shown)
- **Analyze a Review** — type your own review or sample one from the dataset
  (shown with its real movie title), then run it through the Hugging Face models
- **Chatbot** — an open-ended chat interface (built with `st.chat_message` /
  `st.chat_input`) powered by a Hugging Face chat model, so users can ask
  questions about the dataset, sentiment analysis, or movies in general

## 6. Visualize the Results
The **Explore Dataset** tab's table view lets users filter and browse
reviews by sentiment. (An earlier version of this app also included bar
charts of sentiment distribution — removed in favor of the chatbot, per the
project's Next Goals in Step 9 below.)

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
- ~~Include a chatbot for answering user questions about the dataset~~ — done,
  see the Chatbot tab
- Bring back sentiment distribution charts as a separate tab alongside the chatbot

---

## 📖 Dataset Citation
This app uses a cleaned, titled **Rotten Tomatoes critic reviews dataset**,
hosted on Hugging Face Datasets:
https://huggingface.co/datasets/frankier/processed_multiscale_rt_critics

That dataset is itself a cleaned-up repackaging of the original data
scraped from rottentomatoes.com and published on Kaggle:

> Leone, S. (2020). *Rotten Tomatoes Movies and Critic Reviews Dataset.*
> Kaggle. https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

## 🤖 GenAI API Used
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
  via the `huggingface_hub` Python library — models:
  `distilbert-base-uncased-finetuned-sst-2-english` (sentiment classification),
  `google/flan-t5-base` (review summary/keywords), and
  `HuggingFaceH4/zephyr-7b-beta` (open-ended chatbot).
