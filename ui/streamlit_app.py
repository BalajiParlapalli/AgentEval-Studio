"""
AgentEval Studio — Streamlit Dashboard
Talks to the FastAPI backend at API_BASE_URL.
"""
import os
import json
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import StringIO

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="AgentEval Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] .css-1d391kg { color: #ffffff; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #7c83fd; }
    .metric-card .label { font-size: 0.8rem; color: #888; margin-top: 4px; }
    .pass-badge { color: #2ecc71; font-weight: 600; }
    .fail-badge { color: #e74c3c; font-weight: 600; }
    .stProgress > div > div { background: #7c83fd; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Make sure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def score_color(v: float) -> str:
    if v >= 0.75:
        return "🟢"
    if v >= 0.5:
        return "🟡"
    return "🔴"


def fmt_score(v: float) -> str:
    return f"{score_color(v)} {v:.2%}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=60)
    st.title("AgentEval Studio")
    st.caption("Evaluate · Compare · Improve")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📂 Datasets", "🚀 New Run", "📊 Results", "🏆 Leaderboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("v1.0 · LLM-judge (Groq/Gemini)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🧪 AgentEval Studio")
    st.caption("Open-source evaluation and observability dashboard for RAG & AI agents.")
    st.divider()

    runs = api("GET", "/runs/") or []
    datasets = api("GET", "/datasets/") or []

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total_runs = len(runs)
    done_runs = [r for r in runs if r["status"] == "done"]
    avg_pass = (
        sum(r["summary"].get("pass_rate", 0) for r in done_runs) / len(done_runs)
        if done_runs else 0
    )
    avg_faith = (
        sum(r["summary"].get("avg_faithfulness", 0) for r in done_runs) / len(done_runs)
        if done_runs else 0
    )

    for col, label, value in [
        (col1, "Total Runs", str(total_runs)),
        (col2, "Datasets", str(len(datasets))),
        (col3, "Avg Pass Rate", f"{avg_pass:.0%}"),
        (col4, "Avg Faithfulness", f"{avg_faith:.0%}"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if done_runs:
        st.subheader("Recent runs")
        rows = []
        for r in done_runs[:10]:
            s = r["summary"]
            rows.append({
                "Run": r["name"],
                "Version": r["app_version"],
                "Pass Rate": s.get("pass_rate", 0),
                "Faithfulness": s.get("avg_faithfulness", 0),
                "Relevancy": s.get("avg_answer_relevancy", 0),
                "Recall": s.get("avg_context_recall", 0),
                "Latency (ms)": s.get("avg_latency_ms", 0),
                "Cases": s.get("total_cases", 0),
            })
        df = pd.DataFrame(rows)

        # Radar chart for latest run
        if len(done_runs) >= 1:
            latest = done_runs[0]["summary"]
            cats = ["Faithfulness", "Answer Relevancy", "Context Recall", "ROUGE-L", "Keyword Coverage"]
            vals = [
                latest.get("avg_faithfulness", 0),
                latest.get("avg_answer_relevancy", 0),
                latest.get("avg_context_recall", 0),
                latest.get("avg_rouge_l", 0),
                latest.get("avg_keyword_coverage", 0),
            ]
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(124, 131, 253, 0.2)",
                line=dict(color="#7c83fd"),
                name=done_runs[0]["name"],
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Latest Run — Score Radar",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Score columns
        styled = df.style.format({
            "Pass Rate": "{:.0%}",
            "Faithfulness": "{:.0%}",
            "Relevancy": "{:.0%}",
            "Recall": "{:.0%}",
            "Latency (ms)": "{:.0f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No completed runs yet. Upload a dataset and start a run to see results here.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Datasets
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Datasets":
    st.title("📂 Datasets")
    tab1, tab2 = st.tabs(["📋 Existing Datasets", "➕ Add Dataset"])

    with tab1:
        datasets = api("GET", "/datasets/") or []
        if not datasets:
            st.info("No datasets yet. Upload one using the 'Add Dataset' tab.")
        else:
            for ds in datasets:
                with st.expander(f"**{ds['name']}** — {ds['row_count']} cases · `{ds['id'][:8]}…`"):
                    st.caption(ds.get("description") or "No description.")
                    st.caption(f"Created: {ds['created_at'][:19]}")
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if st.button("👁 Preview", key=f"prev_{ds['id']}"):
                            cases = api("GET", f"/datasets/{ds['id']}/cases")
                            if cases:
                                st.dataframe(pd.DataFrame(cases), use_container_width=True)
                    with col2:
                        if st.button("🗑 Delete", key=f"del_{ds['id']}"):
                            api("DELETE", f"/datasets/{ds['id']}")
                            st.rerun()

    with tab2:
        st.subheader("Upload a golden dataset")
        st.markdown("""
        Supported formats: **CSV** or **JSON**

        **Required columns:** `id`, `input`, `expected_answer`
        **Optional:** `expected_keywords` (comma-separated), `reference_context`, `app_version`, `category`
        """)

        with st.expander("📥 Download sample dataset template"):
            sample = [
                {
                    "id": "q001",
                    "input": "What is the main argument in the uploaded document?",
                    "expected_answer": "The document argues that renewable energy reduces carbon emissions.",
                    "expected_keywords": "renewable, carbon, emissions",
                    "reference_context": "Renewable energy sources like solar and wind significantly cut CO2.",
                    "app_version": "v1",
                    "category": "retrieval"
                },
                {
                    "id": "q002",
                    "input": "What evidence supports the economic benefits of solar power?",
                    "expected_answer": "Solar power reduces long-term energy costs and creates jobs.",
                    "expected_keywords": "cost, jobs, solar",
                    "reference_context": "Studies show solar installations lower electricity bills and generate employment.",
                    "app_version": "v1",
                    "category": "reasoning"
                }
            ]
            st.download_button(
                "⬇️ sample_dataset.json",
                data=json.dumps(sample, indent=2),
                file_name="sample_dataset.json",
                mime="application/json",
            )

        ds_name = st.text_input("Dataset name", placeholder="RAG Debate Arena — Benchmark v1")
        ds_desc = st.text_input("Description (optional)")

        input_method = st.radio(
            "Input method",
            ["📋 Paste JSON", "📄 Paste CSV"],
            horizontal=True,
            help="File upload is disabled on HF Spaces (403). Paste your data directly instead.",
        )
        st.info("💡 Tip: download the sample template above, copy its contents, and paste here.")

        pasted_json = None
        pasted_csv = None
        if input_method == "📋 Paste JSON":
            pasted_json = st.text_area(
                "Paste JSON array here",
                height=220,
                placeholder='[{"id":"q001","input":"Your question?","expected_answer":"Expected answer.","expected_keywords":"kw1,kw2"}]',
            )
        else:
            pasted_csv = st.text_area(
                "Paste CSV content here (with header row)",
                height=220,
                placeholder="id,input,expected_answer,expected_keywords\nq001,What is X?,X is Y.,kw1,kw2",
            )

        if st.button("Create Dataset", type="primary") and ds_name:
            if not pasted_json and not pasted_csv:
                st.error("Please paste your dataset content.")
            else:
              with st.spinner("Parsing and saving..."):
                import io, csv as _csv
                if pasted_json:
                    content_bytes = pasted_json.encode()
                    fname = "pasted.json"
                else:
                    content_bytes = pasted_csv.encode()
                    fname = "pasted.csv"
                try:
                    if fname.endswith(".json"):
                        raw = json.loads(content_bytes)
                        if isinstance(raw, dict) and "cases" in raw:
                            raw = raw["cases"]
                    elif fname.endswith(".csv"):
                        reader = _csv.DictReader(io.StringIO(content_bytes.decode()))
                        raw = list(reader)
                    else:
                        st.error("Only .json or .csv supported.")
                        raw = None
                    if raw is not None:
                        cases = []
                        for i, row in enumerate(raw):
                            kws = row.get("expected_keywords", [])
                            if isinstance(kws, str):
                                kws = [k.strip() for k in kws.split(",") if k.strip()]
                            cases.append({
                                "id": row.get("id", f"case_{i}"),
                                "input": row.get("input", row.get("question", "")),
                                "expected_answer": row.get("expected_answer", row.get("answer", "")),
                                "expected_keywords": kws,
                                "reference_context": row.get("reference_context", ""),
                                "app_version": row.get("app_version", "v1"),
                                "category": row.get("category", "general"),
                            })
                        payload = {"name": ds_name, "description": ds_desc, "cases": cases}
                        result = api("POST", "/datasets/", json=payload)
                        if result:
                            st.success(f"✅ Dataset '{result['name']}' created with {result['row_count']} cases.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Parse error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: New Run
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 New Run":
    st.title("🚀 Start New Eval Run")

    datasets = api("GET", "/datasets/") or []
    if not datasets:
        st.warning("Upload a dataset first.")
        st.stop()

    dataset_options = {f"{d['name']} ({d['row_count']} cases)": d["id"] for d in datasets}

    with st.form("run_form"):
        st.subheader("Run configuration")
        col1, col2 = st.columns(2)
        with col1:
            run_name = st.text_input("Run name", placeholder="RAG Arena — Prompt v2 test")
            dataset_label = st.selectbox("Dataset", list(dataset_options.keys()))
            app_version = st.text_input("App version tag", value="v1", placeholder="v1 / prompt-v2 / gemini-flash")
        with col2:
            target_url = st.text_input(
                "Target endpoint URL",
                placeholder="https://your-rag-app.hf.space/api/query",
                help="POST endpoint that accepts {question, app_version} and returns {answer, contexts}",
            )
            pass_threshold = st.slider("Pass threshold (overall score)", 0.0, 1.0, 0.6, 0.05)
            st.caption("🔐 LLM judge key set via HF Space secret: GROQ_API_KEY (Groq) or GEMINI_API_KEY (Gemini AIzaSy).")

        st.warning("⚠️ If your target app is on HF Free tier, it sleeps after inactivity. The first run may show 1-2 errors — this is normal. Wait 30 seconds and re-run immediately.")
        st.markdown("**Target API contract** — your RAG app should accept:")
        st.code('POST /your/endpoint\n{"question": "...", "app_version": "..."}\n→ {"answer": "...", "contexts": ["chunk1", ...], "token_count": 123}', language="json")

        submitted = st.form_submit_button("▶️ Start Run", type="primary")

    if submitted:
        if not run_name or not target_url:
            st.error("Run name and target URL are required.")
        else:
            payload = {
                "name": run_name,
                "dataset_id": dataset_options[dataset_label],
                "app_version": app_version,
                "target_url": target_url,
                "pass_threshold": pass_threshold,
            }

            result = api("POST", "/runs/", json=payload)
            if result:
                st.success(f"✅ Run **{run_name}** started (ID: `{result['id'][:8]}…`)")
                st.info("Evaluation is running in the background. Check **Results** page for status.")

                # Live status polling
                with st.spinner("Waiting for completion..."):
                    for _ in range(60):
                        time.sleep(3)
                        status = api("GET", f"/runs/{result['id']}")
                        if status and status["status"] in ("done", "failed"):
                            break

                if status and status["status"] == "done":
                    s = status["summary"]
                    st.balloons()
                    st.success(f"Run complete! Pass rate: **{s.get('pass_rate', 0):.0%}** · Faithfulness: **{s.get('avg_faithfulness', 0):.0%}**")
                else:
                    st.warning("Run still in progress. Check the Results page.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results":
    st.title("📊 Run Results")

    runs = api("GET", "/runs/") or []
    if not runs:
        st.info("No runs yet.")
        st.stop()

    run_options = {f"{r['name']} [{r['status']}] · {r['created_at'][:10]}": r["id"] for r in runs}
    selected_label = st.selectbox("Select run", list(run_options.keys()))
    run_id = run_options[selected_label]

    run = api("GET", f"/runs/{run_id}")
    if not run:
        st.stop()

    # Status
    status_emoji = {"done": "✅", "running": "⏳", "pending": "🕐", "failed": "❌"}.get(run["status"], "❓")
    st.caption(f"{status_emoji} Status: **{run['status']}** · Version: `{run['app_version']}` · Target: `{run['target_url']}`")

    if run["status"] == "running":
        st.info("Run in progress — refresh to update.")
        if st.button("🔄 Refresh"):
            st.rerun()
        st.stop()

    if run["status"] != "done":
        st.warning("Run not completed.")
        st.stop()

    s = run["summary"]

    # Summary metrics
    cols = st.columns(5)
    metrics = [
        ("Pass Rate", s.get("pass_rate", 0)),
        ("Faithfulness", s.get("avg_faithfulness", 0)),
        ("Answer Relevancy", s.get("avg_answer_relevancy", 0)),
        ("Context Recall", s.get("avg_context_recall", 0)),
        ("ROUGE-L", s.get("avg_rouge_l", 0)),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, f"{val:.0%}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", s.get("total_cases", 0))
    col2.metric("Passed", s.get("passed", 0))
    col3.metric("Avg Latency", f"{s.get('avg_latency_ms', 0):.0f} ms")

    st.divider()

    # Per-case results
    results = api("GET", f"/runs/{run_id}/results") or []
    if results:
        df = pd.DataFrame(results)

        st.subheader("Score distribution")
        score_cols = ["overall_score", "faithfulness", "answer_relevancy", "context_recall", "rouge_l"]
        fig = px.box(
            df[score_cols].melt(var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Metric",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-case breakdown")
        display_df = df[[
            "test_case_id", "overall_score", "faithfulness",
            "answer_relevancy", "context_recall", "rouge_l",
            "latency_ms", "token_count", "passed"
        ]].copy()
        display_df["passed"] = display_df["passed"].map({1: "✅", 0: "❌"})

        st.dataframe(
            display_df.style.format({
                "overall_score": "{:.0%}",
                "faithfulness": "{:.0%}",
                "answer_relevancy": "{:.0%}",
                "context_recall": "{:.0%}",
                "rouge_l": "{:.0%}",
                "latency_ms": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Drill into a single case
        st.divider()
        st.subheader("🔍 Inspect case")
        case_ids = [r["test_case_id"] for r in results]
        chosen = st.selectbox("Select case", case_ids)
        case_data = next((r for r in results if r["test_case_id"] == chosen), None)
        if case_data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input**")
                st.info(case_data["input_text"])
                st.markdown("**Expected answer**")
                st.success(case_data["expected_answer"])
            with col2:
                st.markdown("**Model response**")
                st.warning(case_data["actual_response"])
                if case_data["retrieved_contexts"]:
                    st.markdown("**Retrieved contexts**")
                    for i, ctx in enumerate(case_data["retrieved_contexts"], 1):
                        st.caption(f"Chunk {i}: {str(ctx)[:300]}")

            st.markdown("**Scores**")
            score_row = {
                k: case_data[k]
                for k in ["overall_score", "faithfulness", "answer_relevancy", "context_recall", "rouge_l", "keyword_coverage"]
            }
            fig2 = px.bar(
                x=list(score_row.keys()),
                y=list(score_row.values()),
                color=list(score_row.values()),
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
                labels={"x": "Metric", "y": "Score"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                coloraxis_showscale=False,
                height=250,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Debug: show raw response from RAG app
        st.divider()
        with st.expander("🔧 Debug — Raw API response for selected case"):
            if case_data:
                st.json({
                    "actual_response": case_data["actual_response"],
                    "retrieved_contexts": case_data["retrieved_contexts"],
                    "token_count": case_data["token_count"],
                    "latency_ms": case_data["latency_ms"],
                    "raw_scores": case_data["raw_scores"],
                })
                st.info("💡 If scores are near 0%, check that your RAG app returns the correct JSON schema shown in 'New Run' page.")

        # Export
        st.divider()
        st.download_button(
            "⬇️ Export results as CSV",
            data=df.to_csv(index=False),
            file_name=f"run_{run_id[:8]}_results.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Leaderboard (V2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Leaderboard":
    st.title("🏆 Version Leaderboard")
    st.caption("Compare prompt/model versions side-by-side.")

    runs = api("GET", "/runs/") or []
    done_runs = [r for r in runs if r["status"] == "done"]

    if len(done_runs) < 1:
        st.info("Complete at least one run to see the leaderboard.")
        st.stop()

    rows = []
    for r in done_runs:
        s = r["summary"]
        rows.append({
            "Rank": 0,
            "Run Name": r["name"],
            "Version": r["app_version"],
            "Overall ↑": s.get("avg_overall", 0) or (
                s.get("avg_faithfulness", 0) * 0.3 +
                s.get("avg_answer_relevancy", 0) * 0.25 +
                s.get("avg_context_recall", 0) * 0.15 +
                s.get("avg_rouge_l", 0) * 0.15 +
                s.get("avg_keyword_coverage", 0) * 0.15
            ),
            "Pass Rate": s.get("pass_rate", 0),
            "Faithfulness": s.get("avg_faithfulness", 0),
            "Answer Relevancy": s.get("avg_answer_relevancy", 0),
            "Context Recall": s.get("avg_context_recall", 0),
            "ROUGE-L": s.get("avg_rouge_l", 0),
            "Avg Latency (ms)": s.get("avg_latency_ms", 0),
            "Cases": s.get("total_cases", 0),
            "Date": r["created_at"][:10],
        })

    df = pd.DataFrame(rows).sort_values("Overall ↑", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df.set_index("Rank")

    # Winner banner
    best = df.iloc[0]
    st.success(f"🥇 **Best version:** `{best['Version']}` from run **{best['Run Name']}** — Overall score: **{best['Overall ↑']:.0%}**")

    # Head-to-head comparison chart
    if len(df) >= 2:
        st.subheader("Head-to-head comparison")
        versions = df["Version"].tolist()[:6]
        score_keys = ["Faithfulness", "Answer Relevancy", "Context Recall", "ROUGE-L"]
        fig = go.Figure()
        for _, row in df.head(6).iterrows():
            fig.add_trace(go.Bar(
                name=f"{row['Version']} ({row['Run Name']})",
                x=score_keys,
                y=[row[k] for k in score_keys],
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Latency vs quality scatter
    if len(df) >= 2:
        st.subheader("Latency vs Quality")
        fig2 = px.scatter(
            df.reset_index(),
            x="Avg Latency (ms)",
            y="Overall ↑",
            color="Version",
            size="Cases",
            hover_data=["Run Name", "Pass Rate"],
            text="Version",
            labels={"Overall ↑": "Overall Score"},
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            yaxis=dict(tickformat=".0%"),
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full leaderboard")
    st.dataframe(
        df.style.format({
            "Overall ↑": "{:.0%}",
            "Pass Rate": "{:.0%}",
            "Faithfulness": "{:.0%}",
            "Answer Relevancy": "{:.0%}",
            "Context Recall": "{:.0%}",
            "ROUGE-L": "{:.0%}",
            "Avg Latency (ms)": "{:.0f}",
        }),
        use_container_width=True,
    )

    # Export
    st.download_button(
        "⬇️ Export leaderboard as CSV",
        data=df.reset_index().to_csv(index=False),
        file_name="leaderboard.csv",
        mime="text/csv",
    )
