# 📊 Feedback Intelligence System

A production-style Python system that aggregates app store reviews from multiple sources,
runs multi-method sentiment analysis, detects trends, and generates PDF stakeholder reports —
all inside a clean Streamlit dashboard.

---

## 🗂️ Project Structure

```
feedback_intelligence/
├── app.py                        ← Main Streamlit dashboard (run this)
├── requirements.txt
├── sample_data/
│   └── sample_feedback.csv       ← Demo dataset (20 records)
└── src/
    ├── ingestion/
    │   ├── play_store.py         ← Google Play Store scraper
    │   ├── app_store.py          ← Apple App Store RSS fetcher
    │   └── csv_importer.py       ← CSV / Excel importer
    ├── processing/
    │   ├── cleaner.py            ← Text normalisation
    │   ├── sentiment.py          ← VADER + TextBlob ensemble
    │   └── categorizer.py        ← Keyword-based auto-tagging & priority scoring
    ├── intelligence/
    │   └── trend_detector.py     ← Rolling sentiment, trend direction, anomaly detection
    └── reports/
        └── pdf_generator.py      ← ReportLab PDF report builder
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
cd feedback_intelligence
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time, auto on first run)

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Usage

### Data Sources

| Source | How to use | API Key? |
|--------|-----------|----------|
| **Google Play Store** | Enter app package name (e.g. `com.spotify.music`) | ❌ None |
| **Apple App Store** | Enter numeric App ID from App Store URL | ❌ None |
| **CSV / Excel** | Upload any file with a text/review column | ❌ None |
| **Sample Data** | Click "Load Sample Dataset" | ❌ None |

### Dashboard Tabs

1. **📈 Overview** — Sentiment donut, trend line, source breakdown, categories
2. **🔬 Deep Analysis** — Rating distribution, keyword frequency, day-of-week sentiment, scatter plots
3. **🚨 Issues & Alerts** — Trending issues, critical alerts, anomaly detection
4. **📋 Raw Data** — Searchable table with CSV export
5. **📄 PDF Report** — One-click professional weekly report download

---

## 🧠 How It Works

### Sentiment Analysis (Multi-Method Ensemble)
- **VADER** (65% weight) — rule-based, optimised for short reviews and social media
- **TextBlob** (35% weight) — ML-based pattern analysis
- Combined into a score from **-1 (very negative)** to **+1 (very positive)**
- Confidence score computed from method agreement × signal strength

### Categorisation
Keyword-rule engine tags each review into one of:
`Bug / Crash` · `Performance` · `UI / Design` · `Feature Request` ·
`Pricing / Subscription` · `Customer Support` · `Positive Experience` · `General`

### Priority Scoring (0–100)
Combines sentiment score + urgency keyword matching:
- 🔴 Critical (≥ 80) · 🟠 High (60–79) · 🟡 Medium (40–59) · 🟢 Low (< 40)

### Trend Detection
- Aggregates to daily sentiment averages
- Computes 7-day rolling mean
- Fits linear regression to determine **improving / declining / stable**
- Z-score anomaly detection flags days with abnormal feedback volume

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `pandas / numpy` | Data processing |
| `vaderSentiment` | Rule-based sentiment |
| `textblob` | ML-based sentiment |
| `nltk` | NLP utilities |
| `google-play-scraper` | Play Store reviews |
| `requests` | App Store RSS fetch |
| `reportlab` | PDF generation |
| `matplotlib` | PDF chart rendering |
| `scikit-learn` | TF-IDF (keyword extraction) |

---

## PDF Report Contents

The generated report includes:
- Executive summary table (total, sentiment %, avg score, trend direction)
- Sentiment pie chart + trend line chart
- Category distribution bar chart
- Top trending issues table
- Critical feedback samples
- Auto-dated filename

---

## Common Issues

**`ModuleNotFoundError`** — run `pip install -r requirements.txt` inside your venv.

**Play Store returns empty** — check the package name is correct. Some apps restrict scraping.

**App Store returns empty** — verify the numeric app ID. Try country=`us` first.

**NLTK errors** — run the NLTK download block in step 4 above.

---
## Video link
https://youtu.be/QKpPAdECUTE
*Built for the Feedback Intelligence System capstone project.*
