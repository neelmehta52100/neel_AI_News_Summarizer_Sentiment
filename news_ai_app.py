# Main entry point for the AI News Dashboard.
# This script acts as the presentation layer. It loads the processed CSV data and show it into an interactive Streamlit interface for exploring summaries and sentiment.

import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path


# Page configuration and styling

st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide",
)

# Global CSS for dark, card-style look
st.markdown(
    """
    <style>
    body {
        background-color: #020617;
    }
    .main {
        background-color: #020617;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .news-card {
        border-radius: 14px;
        padding: 1.0rem 1.2rem;
        margin-bottom: 0.9rem;
        border: 1px solid #1f2937;
        box-shadow: 0 12px 25px rgba(15,23,42,0.7);
    }
    /* NEW: sentiment-specific background colours */
    .news-card-pos {
        background: rgba(22, 163, 74, 0.15);   
    }
    .news-card-neg {
        background: rgba(220, 38, 38, 0.15);   
    }
    .news-card-neu {
        background: #0b1120;                  
    }

    .news-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: #e5e7eb;
    }
    .news-title a {
        color: #e5e7eb;
        text-decoration: none;
    }
    .news-title a:hover {
        text-decoration: underline;
    }
    .news-meta {
        font-size: 0.80rem;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }
    .news-summary {
        font-size: 0.90rem;
        color: #e5e7eb;
        line-height: 1.45;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-pos {
        background: #16a34a;
        color: white;
    }
    .badge-neg {
        background: #dc2626;
        color: white;
    }
    .badge-neu {
        background: #4b5563;
        color: white;
    }
    .badge-score {
        background: #0f172a;
        color: #93c5fd;
        border: 1px solid #1d4ed8;
    }
    .filters-box {
        background: #020617;
        border-radius: 12px;
        padding: 0.8rem 1rem 0.4rem 1rem;
        border: 1px solid #1f2937;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper functions
@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and other useful columns."""
    df = pd.read_csv(csv_path)

    # Clean labels
    if "sentiment" in df.columns:
        df["sentiment_clean"] = df["sentiment"].astype(str).str.lower().str.strip()
    else:
        df["sentiment_clean"] = ""

    # making sure required text columns exist
    for col in [
        "summary_llm_150words",
        "summary_llm_german",
        "title",
        "title_german",       
        "link",
        "sentiment_score",
        "content",
    ]:
        if col not in df.columns:
            df[col] = ""

    # Add source column from URL since missing in our CSV file (Google / TechCrunch / Other)
    # Add / clean source column (Google / TechCrunch / Other)
    def detect_source(link: str) -> str:
        url = str(link).lower()
        if "techcrunch" in url:
            return "TechCrunch"
        elif "research.google" in url or "google" in url:
            return "Google Research"
        else:
            return "Other"

    if "source" not in df.columns:
        # No column at all → build it from URL
        df["source"] = df["link"].apply(detect_source)
    else:
        # Column exists but may be NaN or empty strings
        df["source"] = df["source"].fillna("")
        # If *all* values are empty/NaN/blank, recompute from URL
        if not df["source"].astype(str).str.strip().any():
            df["source"] = df["link"].apply(detect_source)
        else:
            # Otherwise just clean up blanks
            df["source"] = df["source"].replace("", "Unknown")


    # Here I am making sure orig_words / llm_words / compression exists otherwise add it
    if "orig_words" not in df.columns:
        df["orig_words"] = df["content"].astype(str).str.split().str.len()
    if "llm_words" not in df.columns:
        df["llm_words"] = df["summary_llm_150words"].astype(str).str.split().str.len()
    if "llm_compression_ratio" not in df.columns:
        df["llm_compression_ratio"] = df["llm_words"] / df["orig_words"].replace(0, 1)

    return df


def sentiment_badge_html(label: str) -> str:
    """Return colored HTML badge for sentiment label."""
    if not isinstance(label, str):
        label = ""
    lab = label.lower()
    if "neg" in lab:
        css_class = "badge badge-neg"
        text = "Negative"
    elif "pos" in lab:
        css_class = "badge badge-pos"
        text = "Positive"
    else:
        css_class = "badge badge-neu"
        text = "Neutral"
    return f'<span class="{css_class}">{text}</span>'


def card_class_for_sentiment(sentiment_clean: str) -> str:
    """Return CSS class for card background based on sentiment."""
    s = (sentiment_clean or "").lower()
    if "neg" in s:
        return "news-card news-card-neg"
    elif "pos" in s:
        return "news-card news-card-pos"
    else:
        return "news-card news-card-neu"


def score_badge_html(score: float) -> str:
    """Return nice badge for sentiment score (0–1 -> 0–100%)."""
    try:
        if score > 1.0:
            score = score / 100.0
    except Exception:
        score = 0.0
    pct = f"{score * 100:0.1f}%"
    return f'<span class="badge badge-score">Score: {pct}</span>'


# Load CSV data otherwise asking user to makesure file exists
data_path = Path("data") / "combined_sentiment.csv"
if not data_path.exists():
    st.error(f"CSV not found at {data_path}. Make sure the file is in data/.")
    st.stop()

df = load_data(data_path)


# Sidebar controls (filters, language, Sentiment, others.)

st.sidebar.title("⚙️ Controls")

# Search box
search_query = st.sidebar.text_input("Search in titles", "")

# Language selection
language_choice = st.sidebar.radio(
    "Language",          # <--- label changed
    ["English", "German"],
    index=0,
    help="Choose which language to display on cards.",
)

# Sentiment filter with tooltip about score meaning
sentiment_options = ["All", "Positive", "Neutral", "Negative"]
sentiment_choice = st.sidebar.radio(
    "Sentiment filter",
    sentiment_options,
    index=0,
    help="Filter cards by predicted sentiment. "
         "The score shown on each card is the model's confidence (0-100%) in that label."
)

# Source filter (Google vs TechCrunch, plus others)
all_sources = sorted(df["source"].unique())
source_choice = st.sidebar.multiselect(
    "Source filter",
    options=all_sources,
    default=all_sources,
    help="Select which news sources to include in the view."
)


# Information on dataset
st.sidebar.markdown("---")
st.sidebar.write(f"**Total articles:** {len(df)}")


# Main title
st.markdown(
    "<h1 style='color:#e5e7eb; margin-bottom:0.2rem;'>📰 AI News Summarizer</h1>",
    unsafe_allow_html=True,
)
# Giving myself credit :)
st.markdown(
    "<p style='color:#9ca3af; font-size:0.85rem; margin-top:0;'>This project is done by Neel Mehta.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#9ca3af; font-size:0.9rem;'>Explore AI news articles with summaries, sentiment analysis, and German translations.</p>",
    unsafe_allow_html=True,
)

# Tabs: Browse vs Analytics
tab_browse, tab_analytics = st.tabs(["📄 Browse articles", "📊 Analytics"])


# TAB 1: Browse Articles

with tab_browse:
    # Filter summary box
    st.markdown('<div class="filters-box">', unsafe_allow_html=True)
    summary_bits = []

    if sentiment_choice != "All":
        summary_bits.append(f"Sentiment: **{sentiment_choice}**")
    if source_choice and len(source_choice) != len(all_sources):
        summary_bits.append("Sources: " + ", ".join(f"`{s}`" for s in source_choice))
    if search_query.strip():
        summary_bits.append(f"Search: `\"{search_query.strip()}\"`")
    summary_bits.append(f"Language: **{language_choice}**")

    st.markdown(
        " • ".join(summary_bits) if summary_bits else "Showing all articles.",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Friendly explanation of the score
    st.caption(
        "NOTE: *Score* = model confidence for the predicted sentiment (0-100%), "
        "not how “positive” or “negative” the news is."
    )

    # Apply filters
    filtered = df.copy()

    # Sentiment filter
    if sentiment_choice != "All":
        target = sentiment_choice.lower()
        filtered = filtered[filtered["sentiment_clean"].str.contains(target, na=False)]

    # Source filter
    if source_choice:
        filtered = filtered[filtered["source"].isin(source_choice)]

    # Search filter
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[filtered["title"].astype(str).str.lower().str.contains(q)]

    # Decide which summary column to show
    summary_col = (
        "summary_llm_150words" if language_choice == "English" else "summary_llm_german"
    )
    # Decide which title column to show
    title_col = "title" if language_choice == "English" else "title_german"

    # Display cards
    if filtered.empty:
        st.warning("No articles match your filters/search. Try changing the filters.")
    else:
        st.markdown(
            f"<p style='color:#9ca3af; font-size:0.85rem;'>Showing {len(filtered)} articles.</p>",
            unsafe_allow_html=True,
        )

        for _, row in filtered.iterrows():
            title = row.get(title_col, "Untitled")
            link = row.get("link", "")
            summary = row.get(summary_col, "")

            sent_label = row.get("sentiment", "")
            sent_score = row.get("sentiment_score", 0.0)
            sent_clean = row.get("sentiment_clean", "")

            sent_html = sentiment_badge_html(sent_label)
            score_html = score_badge_html(sent_score)
            card_class = card_class_for_sentiment(sent_clean)  # NEW

            card_html = f"""
            <div class="{card_class}">
                <div class="news-title">
                    <a href="{link}" target="_blank">{title}</a>
                </div>
                <div class="news-meta">
                    {sent_html} {score_html} &nbsp;|&nbsp;
                    <span style="font-size:0.75rem; color:#9ca3af;">Source: {row.get('source','')}</span>
                </div>
                <div class="news-summary">
                    {summary}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# TAB 2: Analytics

with tab_analytics:
    st.subheader("Summary vs original article length")

    col1, col2 = st.columns(2)

    # Histogram of original article word counts
    with col1:
        st.markdown("**Original article length**")
        hist_orig = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("orig_words", bin=alt.Bin(maxbins=30), title="Words per article"),
                y=alt.Y("count()", title="Number of articles"),
            )
            .properties(height=250)
        )
        st.altair_chart(hist_orig, use_container_width=True)

    # Histogram of LLM summary word counts
    with col2:
        st.markdown("**LLM summary length (llm_words)**")
        hist_llm = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("llm_words", bin=alt.Bin(maxbins=30), title="Words per summary"),
                y=alt.Y("count()", title="Number of articles"),
            )
            .properties(height=250)
        )
        st.altair_chart(hist_llm, use_container_width=True)

    st.markdown("---")

    # Compression ratio (summary / original length)
    st.markdown("**Compression ratio (summary / original length)**")
    scatter_comp = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("orig_words", title="Original words per article"),
            y=alt.Y("llm_compression_ratio", title="LLM summary words / original words"),
            color=alt.Color("source", title="Source"),
            tooltip=["title", "orig_words", "llm_words", "llm_compression_ratio", "source"],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(scatter_comp, use_container_width=True)

    st.markdown("---")

    # Sentiment distribution by source
    st.subheader("Sentiment distribution by source")
    sent_counts = (
        df.groupby(["source", "sentiment_clean"])
        .size()
        .reset_index(name="count")
    )

    bar_sent = (
        alt.Chart(sent_counts)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title="Source"),
            y=alt.Y("count:Q", title="Number of articles"),
            color=alt.Color(
                "sentiment_clean:N",
                title="Sentiment",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#16a34a", "#4b5563", "#dc2626"],
                ),
            ),
        )
        .properties(height=280)
    )
    st.altair_chart(bar_sent, use_container_width=True)

    st.markdown("---")

  # Stats table
    st.subheader("Length & compression statistics")
    stats_cols = ["orig_words", "llm_words", "llm_compression_ratio"]
    stats_df = df[stats_cols].describe().T
    stats_df["median"] = df[stats_cols].median()
    st.dataframe(stats_df.round(2))
