# frontend/app.py
"""
  1) final report (markdown)
  2) trace
  3) subtasks + internal/web evidence

"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import sys
import uuid

sys.path.append(str(Path(__file__).resolve().parents[1]))
# from backend.agent.pipeline import run_pipeline
from backend.agent.pipeline import run_planner, run_from_state
from backend.agent.state import init_state


st.set_page_config(page_title="Project Nexus", layout="wide")

st.title("Project Nexus: Discovery Agent")
st.caption("AI-powered research assistant for enterprise initiatives.")

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Settings")

    user_role = st.selectbox("User role", options=["standard", "admin"], index=0)
    st.markdown("---")
    st.write("Tips")
    st.markdown("- **Be specific:** The more context you provide, the better the historical matching.\n- **Access Control:** Switch to `admin` to simulate querying restricted (C2) initiatives.")



# =========================
# Main input
# =========================
user_query = st.text_area(
    "Describe your project idea or business challenge:", 
    placeholder="e.g., We need to build an internal HR chatbot using LLMs to reduce L1 support tickets...", 
    height=120
)

col_a, col_b = st.columns([1, 3])
with col_a:
    run_btn = st.button("Run", type="primary", use_container_width=True)
with col_b:
    st.write("")

# =========================
# Run pipeline
# =========================
if run_btn:
    if not user_query.strip():
        st.warning("Please enter a project idea.")
        st.stop()

    with st.status("Running...", expanded=True) as status:
        # ---- Phase 1: Plan ----
        status.update(label="Phase 1/3 — Planning subtasks", state="running")
        subtasks, rejected, reason = run_planner(user_query)

        if rejected:
            st.warning(f"Input rejected: {reason}")
            status.update(label="Done (rejected)", state="complete")
            st.stop()

        # visualize subtasks
        st.subheader("Planned subtasks")
        st.table(
            [
                {
                    "intent": t.get("intent", ""),
                    "question": t.get("question", ""),
                }
                for t in subtasks
            ]
        )

        # ---- Phase 2: Execute tools ----
        status.update(label="Phase 2/3 — Retrieving evidence ", state="running")

        trace_id = uuid.uuid4().hex[:12]  
        allow_c2 = (user_role == "admin") 

        state = init_state(
            user_query=user_query,
            trace_id=trace_id,
            user_role=user_role,
            allow_c2=allow_c2,
        )
        state["subtasks"] = subtasks

        # ---- Phase 3: Summarize + Guardrails + Trace ----
        status.update(label="Phase 3/3 — Synthesizing report", state="running")
        report, trace_path = run_from_state(state)

        status.update(label="Done", state="complete")


    st.success("Done.")

    # --- Output report ---
    st.subheader("Report")
    st.markdown(report)

    # --- Trace path ---
    st.subheader("Trace")
    st.code(trace_path)

    # --- Load trace JSON for transparency ---
    st.subheader("Execution Details (Trace)")
    try:
        p = Path(trace_path)
        trace = json.loads(p.read_text(encoding="utf-8"))
        # =========================
        # Pipeline progress
        # =========================
        st.markdown("### Pipeline Progress")

        events = trace.get("events", []) or []

        if not events:
            st.info("No pipeline events recorded.")
        else:
            status_icon = {
                "started": "🟡",
                "completed": "🟢",
                "failed": "🔴",
            }

            for ev in events:
                icon = status_icon.get(ev.get("status"), "⚪")
                step = ev.get("step", "")
                message = ev.get("message", "")
                ts = ev.get("ts", "")

                st.write(f"{icon} **{step}** — {message}")
                st.caption(f"time: {ts}")

        # High-level metrics / warnings
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Latency (s)", trace.get("metrics", {}).get("latency_s", "—"))
        with mcol2:
            st.metric("Internal evidence", trace.get("metrics", {}).get("internal_count", "—"))
        with mcol3:
            st.metric("Web evidence", trace.get("metrics", {}).get("web_count", "—"))

        warnings = trace.get("warnings", [])
        if warnings:
            st.warning("Warnings: " + ", ".join(warnings))

        if trace.get("blocked"):
            st.error(f"Output blocked: {trace.get('block_reason','')}".strip())

        # Subtasks
        st.markdown("### Subtasks")
        subtasks = trace.get("subtasks", [])
        if not subtasks:
            st.info("No subtasks produced (possibly rejected input).")
        else:
            for t in subtasks:
                sid = t.get("id", "")
                intent = t.get("intent", "")
                question = t.get("question", "")
                search_query = t.get("search_query", "")
                title = f"{sid} — {intent}"

                with st.expander(title, expanded=False):
                    st.markdown(f"**Question**: {question}")
                    st.markdown(f"**Search query**: `{search_query}`")

                    evidence = (t.get("evidence") or {})
                    internal = evidence.get("internal", []) or []
                    web = evidence.get("web", []) or []
                    notes = evidence.get("notes", []) or []

                    # Evidence notes
                    if notes:
                        st.markdown("**Notes**")
                        for n in notes:
                            st.write(f"- {n}")

                    # Internal evidence table
                    st.markdown("**Internal evidence**")
                    if internal:
                        for item in internal:
                            st.write(
                                f"- **{item.get('project_id','')}** | "
                                f"{item.get('title','')} | "
                                f"score={item.get('score','')} | "
                                f"status={item.get('status','')}"
                            )
                            if item.get("summary"):
                                st.caption(item["summary"])
                    else:
                        st.write("_None_")

                    # Web evidence list
                    st.markdown("**Web evidence**")
                    if web:
                        for i, w in enumerate(web, start=1):
                            st.write(f"- [WEB:{i}] {w.get('title','')}")
                            if w.get("url"):
                                st.caption(w["url"])
                            if w.get("snippet"):
                                st.caption(w["snippet"])
                    else:
                        st.write("_None_")

        # Optional: show full raw trace json
        with st.expander("Show raw trace JSON", expanded=False):
            st.json(trace)

    except Exception as e:
        st.error(f"Failed to load trace file: {type(e).__name__}: {e}")
        st.info("You can still use the report and trace path shown above.")