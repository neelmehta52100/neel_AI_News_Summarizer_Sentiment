"""
End-to-end pipeline:

1. Scrape Google Research blog + TechCrunch AI articles (or load existing CSVs).
2. Combine into data/combined_AI_articles.csv
3. Generate ~150-word English summaries with an LLM.
4. Translate summaries + titles into German.
5. Run 3-class sentiment on summaries.
6. Save final dataset as data/combined_sentiment.csv for news_ai_app.py.
"""

import os
import time
import random
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GOOGLE_CSV = DATA_DIR / "google_research_blog_articles.csv"
TC_CSV = DATA_DIR / "techcrunch_ai_articles_clean.csv"
COMBINED_AI_CSV = DATA_DIR / "combined_AI_articles.csv"
COMBINED_SENTIMENT_CSV = DATA_DIR / "combined_sentiment.csv"

MAX_GOOGLE_ARTICLES = 50
MAX_TC_ARTICLES = 50

# HF models (same as your notebook)
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MT_MODEL = "Helsinki-NLP/opus-mt-en-de"

# -------------------------------------------------------------------
# HELPER: BASIC HTTP FETCH
# -------------------------------------------------------------------

def fetch_page(url, session=None, sleep_range=(1, 3)):
    """Fetch a URL with a desktop UA + small sleep to be polite."""
    if session is None:
        session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    time.sleep(random.uniform(*sleep_range))
    resp = session.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

# -------------------------------------------------------------------
# SCRAPING: GOOGLE RESEARCH BLOG
# -------------------------------------------------------------------

def is_google_article_url(url: str) -> bool:
    """
    Simple heuristic: Google Research blog posts usually live under /blog/
    and look like full article paths, not listing/search URLs.
    """
    parsed = urlparse(url)
    path = parsed.path
    if not path.startswith("/blog"):
        return False
    # Avoid tag/search/contact etc.
    if any(bad in path for bad in ["search", "about", "contact"]):
        return False
    return True

def extract_google_article_links(listing_html, base="https://research.google"):
    soup = BeautifulSoup(listing_html, "html.parser")
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(base, href)
        if not is_google_article_url(full_url):
            continue
        if full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return links

def parse_google_article(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Title: often in <h1> or meta tags
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Content: very approximate – you can refine based on your previous notebook
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    content = "\n".join(paragraphs)

    # Basic date heuristic: look for <time> or date-like text
    time_tag = soup.find("time")
    date_text = time_tag.get_text(strip=True) if time_tag else ""

    return {
        "source": "Google Research",
        "title": title,
        "link": url,
        "date": date_text,
        "content": content,
    }

def scrape_google_research_blog(
    base_url: str = "https://research.google/blog/",
    max_articles: int = MAX_GOOGLE_ARTICLES,
    output_csv: Path = GOOGLE_CSV,
):
    print(f"[Google] Scraping up to {max_articles} articles from {base_url}")
    session = requests.Session()
    listing_html = fetch_page(base_url, session=session)
    article_urls = extract_google_article_links(listing_html)

    # Limit to max_articles
    article_urls = article_urls[:max_articles]
    articles = []

    for i, url in enumerate(article_urls, start=1):
        try:
            print(f"[Google] {i}/{len(article_urls)}: {url}")
            html = fetch_page(url, session=session)
            art = parse_google_article(html, url)
            articles.append(art)
        except Exception as e:
            print(f"[Google] ERROR on {url}: {e}")

    df = pd.DataFrame(articles)
    df.to_csv(output_csv, index=False)
    print(f"[Google] Saved {len(df)} articles to {output_csv}")
    return df

# -------------------------------------------------------------------
# SCRAPING: TECHCRUNCH AI TAG
# -------------------------------------------------------------------

def is_tc_article_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path
    # e.g. /2025/01/01/some-ai-article/
    return bool(re.match(r"^/\d{4}/\d{2}/\d{2}/.+", path))

def extract_tc_article_links(listing_html, base="https://techcrunch.com"):
    soup = BeautifulSoup(listing_html, "html.parser")
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(base, href)
        if not is_tc_article_url(full_url):
            continue
        if full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return links

def parse_tc_article(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Content (again: simple heuristic)
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    content = "\n".join(paragraphs)

    # Date
    time_tag = soup.find("time")
    date_text = time_tag.get_text(strip=True) if time_tag else ""

    return {
        "source": "TechCrunch",
        "title": title,
        "link": url,
        "date": date_text,
        "content": content,
    }

def scrape_techcrunch_ai_tag(
    start_url: str = "https://techcrunch.com/tag/artificial-intelligence/",
    max_articles: int = MAX_TC_ARTICLES,
    output_csv: Path = TC_CSV,
):
    print(f"[TC] Scraping up to {max_articles} articles from {start_url}")
    session = requests.Session()
    seen_urls = set()
    all_articles = []
    current_url = start_url

    while len(all_articles) < max_articles and current_url:
        try:
            print(f"[TC] Listing page: {current_url}")
            html = fetch_page(current_url, session=session)
        except Exception as e:
            print(f"[TC] ERROR fetching listing {current_url}: {e}")
            break

        article_urls = extract_tc_article_links(html)
        article_urls = [u for u in article_urls if u not in seen_urls]

        for u in article_urls:
            if len(all_articles) >= max_articles:
                break
            try:
                print(f"[TC] Article {len(all_articles)+1}/{max_articles}: {u}")
                ah = fetch_page(u, session=session)
                art = parse_tc_article(ah, u)
                all_articles.append(art)
                seen_urls.add(u)
            except Exception as e:
                print(f"[TC] ERROR on article {u}: {e}")

        # Simple pagination: look for "Next" link
        soup = BeautifulSoup(html, "html.parser")
        next_link = None
        for a in soup.find_all("a", href=True):
            if "next" in a.get_text(strip=True).lower():
                next_link = urljoin("https://techcrunch.com", a["href"])
                break

        if not next_link:
            break
        current_url = next_link

    df = pd.DataFrame(all_articles)
    df.to_csv(output_csv, index=False)
    print(f"[TC] Saved {len(df)} articles to {output_csv}")
    return df

# -------------------------------------------------------------------
# COMBINING DATASETS
# -------------------------------------------------------------------

def load_or_scrape_sources():
    # Google
    if GOOGLE_CSV.exists():
        print(f"[Google] Loading existing CSV: {GOOGLE_CSV}")
        df_google = pd.read_csv(GOOGLE_CSV)
    else:
        df_google = scrape_google_research_blog()

    # TechCrunch
    if TC_CSV.exists():
        print(f"[TC] Loading existing CSV: {TC_CSV}")
        df_tc = pd.read_csv(TC_CSV)
    else:
        df_tc = scrape_techcrunch_ai_tag()

    return df_google, df_tc

def combine_sources(df_google, df_tc):
    print("[Combine] Normalising columns and concatenating...")
    # Ensure common columns
    for df in (df_google, df_tc):
        for col in ["source", "title", "link", "date", "content"]:
            if col not in df.columns:
                df[col] = ""

    combined = pd.concat([df_google, df_tc], ignore_index=True)
    combined.to_csv(COMBINED_AI_CSV, index=False)
    print(f"[Combine] Combined {len(combined)} articles -> {COMBINED_AI_CSV}")
    return combined

# -------------------------------------------------------------------
# LLM SUMMARISATION (≈150 words)
# -------------------------------------------------------------------

def create_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    print(f"[Summarizer] Using device: {'GPU' if device == 0 else 'CPU'}")
    return pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        tokenizer=SUMMARIZER_MODEL,
        device=device,
    )

def summarize_to_150_words(summarizer, text, target_words=150):
    if not isinstance(text, str) or not text.strip():
        return ""

    shortened = text.strip()
    # (optional) rough char-level cut to avoid super-long texts
    if len(shortened) > 4000:
        shortened = shortened[:4000]

    result = summarizer(
        shortened,
        max_length=256,
        min_length=80,
        do_sample=False,
        truncation=True, 
    )
    summary = result[0]["summary_text"]

    # Optional word-trimming to ~150 words
    words = summary.split()
    if len(words) > target_words * 1.3:
        summary = " ".join(words[: int(target_words * 1.3)])

    return summary


def add_llm_summaries(df):
    summarizer = create_summarizer()
    summaries = []
    for i, text in enumerate(df["content"]):
        print(f"[Summariser] {i+1}/{len(df)}", end="\r", flush=True)
        summary = summarize_to_150_words(summarizer, text)
        summaries.append(summary)
    print()  # newline
    df["summary_llm_150words"] = summaries
    return df

# -------------------------------------------------------------------
# MACHINE TRANSLATION (EN -> DE) FOR SUMMARIES & TITLES
# -------------------------------------------------------------------

def create_mt_model():
    tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[MT] Using device: {device}")
    return tokenizer, model, device

def translate_batch(tokenizer, model, device, text_list, max_length=256):
    clean_texts = [(t if isinstance(t, str) and t.strip() else "") for t in text_list]
    if not any(clean_texts):
        return [""] * len(clean_texts)

    enc = tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

    decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    return decoded

def add_german_summaries_and_titles(df):
    tokenizer, model, device = create_mt_model()

    # --- German summaries ---
    print("[MT] Translating summaries to German...")
    de_summaries = []
    batch_size = 16
    for start in range(0, len(df), batch_size):
        batch = df["summary_llm_150words"].iloc[start:start+batch_size].tolist()
        translated = translate_batch(tokenizer, model, device, batch, max_length=256)
        de_summaries.extend(translated)
        print(f"[MT] Summaries {min(len(de_summaries), len(df))}/{len(df)}", end="\r")

    df["summary_llm_german"] = de_summaries[: len(df)]
    print()

    # --- German titles ---
    print("[MT] Translating titles to German...")
    if "title_german" not in df.columns:
        df["title_german"] = ""

    batch_size = 32
    indices = df[
        df["title"].astype(str).str.strip().ne("")
    ].index.tolist()

    de_titles = {}
    for i in range(0, len(indices), batch_size):
        idx_batch = indices[i : i + batch_size]
        texts = df.loc[idx_batch, "title"].tolist()
        translated = translate_batch(tokenizer, model, device, texts, max_length=64)
        for j, idx in enumerate(idx_batch):
            de_titles[idx] = translated[j]
        print(f"[MT] Titles {min(i+batch_size, len(indices))}/{len(indices)}", end="\r")

    for idx, de_t in de_titles.items():
        df.at[idx, "title_german"] = de_t
    print()

    return df

# -------------------------------------------------------------------
# SENTIMENT (3-class) ON SUMMARIES
# -------------------------------------------------------------------

def create_sentiment_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    print(f"[Sentiment] Using device: {'GPU' if device == 0 else 'CPU'}")
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return pipe

def add_three_class_sentiment(df):
    sentiment_pipe = create_sentiment_pipeline()
    labels_all = []
    scores_all = []
    batch_size = 16

    texts = df["summary_llm_150words"].fillna("").astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        print(f"[Sentiment] {min(start+batch_size, len(texts))}/{len(texts)}", end="\r")
        results = sentiment_pipe(batch)
        labels_all.extend([r["label"] for r in results])
        scores_all.extend([float(r["score"]) for r in results])

    df["sentiment"] = labels_all[: len(df)]
    df["sentiment_score"] = scores_all[: len(df)]
    print()
    return df

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():
    print("=== AI News Pipeline: START ===")

    # 1) Scrape or load individual sources
    df_google, df_tc = load_or_scrape_sources()

    # 2) Combine into one dataset
    df = combine_sources(df_google, df_tc)

    # 3) LLM summarisation
    df = add_llm_summaries(df)

    # 4) German translations (summaries + titles)
    df = add_german_summaries_and_titles(df)

    # 5) 3-class sentiment on summaries
    df = add_three_class_sentiment(df)

    # 6) Save final CSV
    df.to_csv(COMBINED_SENTIMENT_CSV, index=False)
    print(f"[Final] Saved: {COMBINED_SENTIMENT_CSV.resolve()}")

    print("=== AI News Pipeline: DONE ===")

if __name__ == "__main__":
    main()
