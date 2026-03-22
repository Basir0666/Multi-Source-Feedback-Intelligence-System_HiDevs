"""
Feedback Intelligence System
Multi-source feedback aggregation · Sentiment Analysis · Trend Detection · PDF Reports

Run with:  streamlit run app.py
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Local imports ────────────────────────────────────────────────────────────
from src.ingestion.app_store import fetch_app_store_reviews
from src.ingestion.csv_importer import load_csv_feedback
from src.ingestion.play_store import fetch_play_store_reviews
from src.intelligence.trend_detector import (
    compute_rolling_sentiment,
    detect_anomalies,
    detect_sentiment_trend,
    get_trending_issues,
)
from src.processing.categorizer import categorize_dataframe, extract_top_keywords
from src.processing.cleaner import clean_dataframe
from src.processing.sentiment import analyze_dataframe
from src.reports.pdf_generator import generate_pdf_report

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .main-header {
        background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%);
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .badge-pos { color:#10B981; font-weight:600; }
    .badge-neg { color:#EF4444; font-weight:600; }
    .badge-neu { color:#F59E0B; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("df", pd.DataFrame()), ("processed", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Clean → Sentiment → Categorise."""
    with st.spinner("🔄 Running analysis pipeline (this may take a moment)…"):
        df = clean_dataframe(df)
        df = analyze_dataframe(df)
        df = categorize_dataframe(df)
    return df


def append_and_reprocess(new_df: pd.DataFrame) -> None:
    """Merge new data into session state and mark for reprocessing."""
    if new_df.empty:
        return
    if not st.session_state.df.empty:
        combined = pd.concat([st.session_state.df, new_df], ignore_index=True)
        # Drop duplicates on id if present
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset="id")
        st.session_state.df = combined
    else:
        st.session_state.df = new_df
    st.session_state.processed = False


# ── Sample data ────────────────────────────────────────────────────────────────
def build_sample_data() -> pd.DataFrame:
    """Generate 300 realistic synthetic feedback records for demo."""
    rng = np.random.default_rng(42)

    texts = [
        # Positive
        "The app is absolutely fantastic! Best I've ever used, super intuitive and fast.",
        "Love the new update — performance improvements are very noticeable.",
        "Great customer support. They resolved my issue within a couple of hours!",
        "Very clean UI and easy to navigate. Highly recommended to everyone.",
        "Finally an app that does exactly what it promises. Five stars.",
        "Smooth experience, no crashes in three months of daily use.",
        "The feature updates keep getting better. Team is clearly listening to users.",
        "Excellent tool for tracking goals. Exactly what I needed.",
        "Fast, reliable, and intuitive. Could not ask for more from an app.",
        "Worth every penny of the subscription. Premium features are top notch.",
        # Neutral
        "It's okay, does the job but nothing special about it.",
        "Average experience. Some features are good, others are lacking.",
        "App works fine but could use improvements in the search function.",
        "Decent but the competition has better features at this price point.",
        "Not bad, not great. Gets the job done most of the time.",
        # Negative
        "App keeps crashing every time I open the settings page. Please fix this!",
        "Terrible performance since the latest update. It is extremely slow now.",
        "Lost all my data after the update. No backup option available. Horrible.",
        "Charged me twice for the subscription. Customer service never responded.",
        "Battery drain is insane. This app kills my phone battery in two hours.",
        "Dark mode is completely broken since the last update. Eyes are straining.",
        "The login keeps failing with an error message. Cannot access my account.",
        "Too many ads in the free version. Makes the whole app completely unusable.",
        "UI feels outdated and confusing. The navigation is a total mess.",
        "Push notifications are not working at all. Missing important alerts.",
        "App freezes on the checkout screen. Lost three orders because of this bug.",
        "Why did you remove the data export feature? Terrible product decision.",
        "App force closes randomly during use. Very frustrating for daily workflow.",
        "Sync feature is broken. Data does not update correctly across devices.",
        "Price increased but features were removed. Not worth it at all anymore.",
    ]

    sources = ["Google Play Store", "App Store", "Survey Export"]
    base = datetime.now() - timedelta(days=60)
    records = []

    for i in range(300):
        text = texts[i % len(texts)]
        days_offset = int(rng.integers(0, 61))
        records.append({
            "id": f"sample_{i}",
            "text": text,
            "rating": int(rng.integers(1, 6)),
            "date": base + timedelta(days=days_offset, hours=int(rng.integers(0, 24))),
            "source": sources[i % len(sources)],
            "author": f"User_{i+1}",
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔧 Data Sources")

    # ── Google Play Store ──────────────────────────────────────────────────
    st.markdown("### 🤖 Google Play Store")
    play_id = st.text_input(
        "App package name",
        value="com.spotify.music",
        help="e.g. com.whatsapp, com.instagram.android",
        key="play_id",
    )
    play_count = st.slider("Max reviews", 50, 500, 150, 50, key="play_count")
    if st.button("📥 Fetch Play Store Reviews", use_container_width=True):
        with st.spinner(f"Fetching reviews for {play_id}…"):
            fetched = fetch_play_store_reviews(play_id, count=play_count)
        if not fetched.empty:
            append_and_reprocess(fetched)
            st.success(f"✅ Loaded {len(fetched)} Play Store reviews")
        else:
            st.error("❌ No reviews returned. Check app ID or network.")

    st.markdown("---")

    # ── Apple App Store ────────────────────────────────────────────────────
    st.markdown("### 🍎 Apple App Store")
    apple_id = st.text_input(
        "Numeric App Store ID",
        value="324684580",
        help="Find in App Store URL: .../id<number>",
        key="apple_id",
    )
    apple_country = st.selectbox(
        "Country", ["us", "gb", "in", "au", "ca", "de", "fr"], key="apple_country"
    )
    apple_pages = st.slider("Pages (~50 reviews each)",
                            1, 5, 2, key="apple_pages")
    if st.button("📥 Fetch App Store Reviews", use_container_width=True):
        with st.spinner(f"Fetching App Store reviews…"):
            fetched = fetch_app_store_reviews(
                apple_id, country=apple_country, pages=apple_pages)
        if not fetched.empty:
            append_and_reprocess(fetched)
            st.success(f"✅ Loaded {len(fetched)} App Store reviews")
        else:
            st.error("❌ No reviews returned. Verify app ID and country.")

    st.markdown("---")

    # ── CSV Upload ─────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload CSV / Excel")
    uploaded = st.file_uploader(
        "Drop file here", type=["csv", "xlsx", "xls"], key="csv_upload"
    )
    csv_label = st.text_input(
        "Source label", value="Survey Export", key="csv_label")
    if uploaded and st.button("📥 Load File", use_container_width=True):
        with st.spinner("Parsing file…"):
            fetched = load_csv_feedback(uploaded, source_name=csv_label)
        if not fetched.empty:
            append_and_reprocess(fetched)
            st.success(f"✅ Loaded {len(fetched)} records from file")
        else:
            st.error("❌ Could not parse file. Ensure it has a text/review column.")

    st.markdown("---")

    # ── Demo data ──────────────────────────────────────────────────────────
    st.markdown("### 🎲 Demo")
    if st.button("🔄 Load Sample Dataset", use_container_width=True):
        st.session_state.df = build_sample_data()
        st.session_state.processed = False
        st.success("✅ 300 sample records loaded!")

    # ── Filters (only shown when data is present) ──────────────────────────
    if not st.session_state.df.empty and st.session_state.processed:
        st.markdown("---")
        st.markdown("### 🔍 Filters")

        df_raw = st.session_state.df
        min_d = df_raw["date"].min().date()
        max_d = df_raw["date"].max().date()

        date_sel = st.date_input("Date range", value=(min_d, max_d),
                                 min_value=min_d, max_value=max_d)

        source_opts = ["All"] + sorted(df_raw["source"].unique().tolist())
        source_sel = st.selectbox("Source", source_opts)

        sent_sel = st.multiselect(
            "Sentiment", ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
        )

        cat_opts = ["All"] + sorted(df_raw["category"].unique().tolist())
        cat_sel = st.selectbox("Category", cat_opts)

        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.df = pd.DataFrame()
            st.session_state.processed = False
            st.rerun()
    else:
        # Defaults when no data yet
        date_sel = None
        source_sel = "All"
        sent_sel = ["Positive", "Neutral", "Negative"]
        cat_sel = "All"


# ── Run pipeline if needed ────────────────────────────────────────────────────
if not st.session_state.df.empty and not st.session_state.processed:
    st.session_state.df = run_pipeline(st.session_state.df)
    st.session_state.processed = True
    st.rerun()


# ── Apply filters ─────────────────────────────────────────────────────────────
def get_filtered() -> pd.DataFrame:
    df = st.session_state.df.copy()
    if df.empty:
        return df
    try:
        if date_sel and len(date_sel) == 2:
            df = df[(df["date"].dt.date >= date_sel[0]) &
                    (df["date"].dt.date <= date_sel[1])]
    except Exception:
        pass
    if source_sel != "All":
        df = df[df["source"] == source_sel]
    if sent_sel:
        df = df[df["sentiment"].isin(sent_sel)]
    if cat_sel != "All":
        df = df[df["category"] == cat_sel]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.85rem;">📊 Feedback Intelligence System</h1>
    <p style="margin:0.3rem 0 0;opacity:.88;font-size:.95rem;">
        Real-time multi-source feedback analysis · Sentiment tracking ·
        Issue prioritisation · PDF reporting
    </p>
</div>
""", unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────────────────────────
if st.session_state.df.empty:
    st.info("👈 **Get started:** fetch reviews from the sidebar, upload a CSV, or click **Load Sample Dataset**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "#### 🤖 Google Play Store\nEnter any app package name to fetch the latest reviews automatically — no API key needed.")
    with c2:
        st.markdown(
            "#### 🍎 Apple App Store\nEnter a numeric App Store ID to pull RSS-based reviews. Works globally across countries.")
    with c3:
        st.markdown(
            "#### 📂 CSV / Excel Upload\nUpload any survey or feedback export. Column names are auto-detected and mapped.")
    st.stop()

df = get_filtered()

if df.empty:
    st.warning(
        "⚠️ No records match the current filters. Adjust the sidebar filters.")
    st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
total = len(df)
pos_n = (df["sentiment"] == "Positive").sum()
neg_n = (df["sentiment"] == "Negative").sum()
neu_n = (df["sentiment"] == "Neutral").sum()
avg_sc = round(df["score"].mean(), 3)
avg_rat = round(df["rating"].mean(),
                2) if df["rating"].notna().any() else "N/A"
crit_n = df["priority_label"].str.contains("Critical", na=False).sum()

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("📝 Total",    f"{total:,}")
k2.metric("😊 Positive", f"{pos_n:,}", f"{round(pos_n/total*100, 1)}%")
k3.metric("😞 Negative", f"{neg_n:,}", f"{round(neg_n/total*100, 1)}%")
k4.metric("😐 Neutral",  f"{neu_n:,}", f"{round(neu_n/total*100, 1)}%")
k5.metric("📊 Avg Score", f"{avg_sc:.3f}")
k6.metric("⭐ Avg Rating", f"{avg_rat}")
k7.metric("🔴 Critical",  f"{crit_n}")

st.markdown("---")

COLOR_MAP = {"Positive": "#10B981",
             "Neutral": "#F59E0B", "Negative": "#EF4444"}

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_deep, tab_issues, tab_data, tab_report = st.tabs([
    "📈 Overview",
    "🔬 Deep Analysis",
    "🚨 Issues & Alerts",
    "📋 Raw Data",
    "📄 PDF Report",
])


# ════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab_overview:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        # Sentiment donut
        sc = df["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(
            sc, values="Count", names="Sentiment", hole=0.55,
            color="Sentiment", color_discrete_map=COLOR_MAP,
            title="Sentiment Distribution",
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, height=330,
                              margin=dict(t=45, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        # Trend line
        daily = compute_rolling_sentiment(df)
        if not daily.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily["date_only"], y=daily["avg_sentiment"],
                mode="lines", name="Daily avg",
                line=dict(color="#9CA3AF", width=1), opacity=0.55,
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily["date_only"], y=daily["rolling_sentiment"],
                mode="lines", name="7-day rolling",
                line=dict(color="#4F46E5", width=2.5),
            ))
            fig_trend.add_hline(
                y=0, line_dash="dash", line_color="#EF4444", opacity=0.35,
                annotation_text="Neutral", annotation_position="bottom right",
            )
            fig_trend.update_layout(
                title="Sentiment Trend Over Time",
                xaxis_title="Date", yaxis_title="Score",
                height=330, legend=dict(orientation="h", y=-0.25),
                margin=dict(t=45, b=15, l=10, r=10),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # Source breakdown + category bar
    col_a, col_b = st.columns(2)

    with col_a:
        src_sent = df.groupby(["source", "sentiment"]
                              ).size().reset_index(name="count")
        fig_src = px.bar(
            src_sent, x="source", y="count", color="sentiment",
            color_discrete_map=COLOR_MAP, barmode="group",
            title="Sentiment by Source",
        )
        fig_src.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig_src, use_container_width=True)

    with col_b:
        cat_c = df["category"].value_counts().reset_index()
        cat_c.columns = ["Category", "Count"]
        fig_cat = px.bar(
            cat_c.head(8), x="Count", y="Category",
            orientation="h", title="Top Feedback Categories",
            color="Count", color_continuous_scale="Blues",
        )
        fig_cat.update_layout(
            height=320, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_cat, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 2 — DEEP ANALYSIS
# ════════════════════════════════════════════════════════
with tab_deep:
    trend = detect_sentiment_trend(df)

    t1, t2, t3 = st.columns(3)
    d = trend.get("direction", "stable")
    emoji_map = {"improving": "📈", "declining": "📉", "stable": "➡️"}
    t1.metric(f"{emoji_map.get(d, '➡️')} Trend", d.title())
    t2.metric("Current Avg Sentiment", f"{trend.get('current_avg', 0):.3f}")
    t3.metric("Period Change", f"{trend.get('change', 0):+.3f}")
    st.info(f"📊 **Trend Analysis:** {trend.get('description', 'N/A')}")

    col_l, col_r = st.columns(2)

    with col_l:
        # Rating histogram
        rdf = df[df["rating"].notna()].copy()
        if not rdf.empty:
            rdf["rating"] = rdf["rating"].astype(int)
            rc = rdf["rating"].value_counts().sort_index().reset_index()
            rc.columns = ["Rating", "Count"]
            fig_r = px.bar(
                rc, x="Rating", y="Count",
                title="Rating Distribution (1–5 Stars)",
                color="Rating",
                color_continuous_scale=["#EF4444", "#F59E0B", "#F59E0B",
                                        "#10B981", "#10B981"],
            )
            fig_r.update_layout(height=310, coloraxis_showscale=False)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No numeric ratings available.")

    with col_r:
        # Confidence distribution
        fig_conf = px.histogram(
            df, x="confidence", nbins=20,
            title="Sentiment Confidence Distribution",
            color_discrete_sequence=["#4F46E5"],
        )
        fig_conf.update_layout(height=310)
        st.plotly_chart(fig_conf, use_container_width=True)

    # Top keywords
    st.markdown("### 🔑 Top Keywords")
    col_kw = "clean_text" if "clean_text" in df.columns else "text"
    kws = extract_top_keywords(df[col_kw].tolist(), top_n=25)
    if kws:
        kw_df = pd.DataFrame(kws, columns=["Keyword", "Frequency"])
        fig_kw = px.bar(
            kw_df, x="Frequency", y="Keyword",
            orientation="h", title="Most Frequent Keywords",
            color="Frequency", color_continuous_scale="Purples",
        )
        fig_kw.update_layout(
            height=480, yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    # Day-of-week sentiment
    if len(df) > 14:
        dow_df = df.copy()
        dow_df["day"] = dow_df["date"].dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = dow_df.groupby("day")["score"].mean().reindex(order).fillna(0)
        fig_dow = px.bar(
            x=pivot.index, y=pivot.values,
            title="Average Sentiment by Day of Week",
            labels={"x": "Day", "y": "Avg Score"},
            color=pivot.values,
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
        )
        fig_dow.update_layout(height=310, coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Subjectivity vs polarity scatter
    if "textblob_subjectivity" in df.columns:
        fig_sc = px.scatter(
            df.sample(min(500, len(df))),
            x="textblob_polarity", y="textblob_subjectivity",
            color="sentiment", color_discrete_map=COLOR_MAP,
            title="Polarity vs Subjectivity (TextBlob)",
            opacity=0.65,
        )
        fig_sc.update_layout(height=350)
        st.plotly_chart(fig_sc, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 3 — ISSUES & ALERTS
# ════════════════════════════════════════════════════════
with tab_issues:
    st.markdown("### 🔥 Trending Negative Issues (last 30 days)")
    issues = get_trending_issues(df, days=30, top_n=5)

    if issues:
        for i, iss in enumerate(issues, 1):
            with st.expander(
                f"#{i}  {iss['category']}  —  {iss['count']} reports", expanded=(i == 1)
            ):
                ia, ib, ic = st.columns(3)
                ia.metric("Volume", iss["count"])
                ib.metric("Avg Sentiment", f"{iss['avg_sentiment']:.3f}")
                ic.metric("Priority Score", f"{iss['avg_priority']:.0f} / 100")

                samples = df[
                    (df["category"] == iss["category"]) &
                    (df["sentiment"] == "Negative")
                ].head(3)
                if not samples.empty:
                    st.markdown("**Sample Reviews:**")
                    for _, row in samples.iterrows():
                        st.markdown(f"> 💬 *\"{row['text'][:200]}\"*")
    else:
        st.success("✅ No major trending issues detected in the selected period.")

    # Critical alerts
    st.markdown("---")
    st.markdown("### 🚨 Critical Feedback Alerts")
    crit_df = (
        df[df["priority_label"].str.contains("Critical", na=False)]
        .sort_values("priority_score", ascending=False)
        .head(15)
    )
    if not crit_df.empty:
        st.error(
            f"⚠️ **{len(crit_df)} critical items** require immediate attention!")
        for _, row in crit_df.iterrows():
            cols = st.columns([4, 1, 1, 1])
            cols[0].write(f"💬 {str(row['text'])[:160]}…")
            cols[1].write(f"🏷️ {row['category']}")
            cols[2].write(f"📡 {row['source']}")
            cols[3].write(f"🔥 {row['priority_score']}/100")
            st.divider()
    else:
        st.success("✅ No critical alerts at this time.")

    # Anomaly detection
    st.markdown("---")
    st.markdown("### 📡 Volume Anomaly Detection")
    anomalies = detect_anomalies(df)
    if not anomalies.empty:
        st.warning(
            f"⚠️ Detected **{len(anomalies)} days** with abnormal feedback volume!")
        st.dataframe(
            anomalies[["date_only", "count", "avg_sentiment", "z_score"]].rename(columns={
                "date_only": "Date", "count": "Volume",
                "avg_sentiment": "Avg Sentiment", "z_score": "Z-Score",
            }),
            use_container_width=True,
        )
    else:
        st.info("📊 No volume anomalies detected in current data range.")


# ════════════════════════════════════════════════════════
#  TAB 4 — RAW DATA
# ════════════════════════════════════════════════════════
with tab_data:
    st.markdown(f"### 📋 All Feedback — {total:,} records")

    search = st.text_input(
        "🔍 Search reviews", placeholder="Filter by keyword…")

    show_cols = [c for c in [
        "date", "source", "text", "sentiment", "score", "confidence",
        "category", "priority_label", "rating",
    ] if c in df.columns]

    disp = df[show_cols].copy()
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d %H:%M")
    if "score" in disp.columns:
        disp["score"] = disp["score"].round(3)

    if search:
        mask = disp["text"].str.contains(search, case=False, na=False)
        disp = disp[mask]

    st.dataframe(disp, use_container_width=True, height=520)

    csv_dl = disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_dl,
        file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════
#  TAB 5 — PDF REPORT
# ════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### 📄 Generate Weekly Insight Report")

    app_name = st.text_input("App / Company Name", value="My App")
    r1, r2 = st.columns(2)
    with r1:
        rpt_start = st.date_input(
            "Report start", value=datetime.now().date() - timedelta(days=7))
    with r2:
        rpt_end = st.date_input("Report end", value=datetime.now().date())

    if st.button("🖨️ Generate PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating professional PDF…"):
            try:
                t_info = detect_sentiment_trend(df)
                t_issues = get_trending_issues(df, days=30)
                dr_str = f"{rpt_start.strftime('%d %b')} – {rpt_end.strftime('%d %b %Y')}"

                pdf_bytes = generate_pdf_report(
                    df=df,
                    trend_info=t_info,
                    trending_issues=t_issues,
                    date_range=dr_str,
                    company_name=app_name,
                )
                st.success("✅ Report generated!")
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                # Preview card
                st.markdown("#### 📋 Report Summary")
                st.info(
                    f"**{app_name}** — Feedback Intelligence Report\n\n"
                    f"📅 Period: {dr_str}  \n"
                    f"📝 Total records analysed: **{total:,}**  \n"
                    f"😊 Positive: **{pos_n:,}** ({round(pos_n/total*100, 1)}%)  \n"
                    f"😞 Negative: **{neg_n:,}** ({round(neg_n/total*100, 1)}%)  \n"
                    f"📈 Trend: **{t_info.get('direction', 'N/A').title()}**  \n"
                    f"🔥 Top Issue: **{t_issues[0]['category'] if t_issues else 'None'}**"
                )
            except Exception as exc:
                st.error(f"❌ PDF generation failed: {exc}")
                logger.exception("PDF error")
