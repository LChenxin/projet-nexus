# backend/agent/pipeline.py

"""
Pipeline single DAG
1) planner: user_query -> subtasks (id/intent/question/search_query)
2) executor: for each subtask execute internal_search / web_search，get evidence
3) summarizer:  evidence to brief（with citations）
4) guardrails: 最小校验（结构、citation、C2 泄露）
5) logging: dump state 到 logs/{trace_id}.json

"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend import config
from backend.llm import chat
from backend.agent.state import AgentState, EvidencePack, SubTask, init_state
from backend.agent import prompts
from backend.tools.internal_search import get_internal_search
from backend.tools.web_search import web_search


# =========================
# 1) Utility: trace id + logging
# =========================
def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _dump_state(state: AgentState) -> Path:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOGS_DIR / f"{state['trace_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return out_path


# =========================
# 2) Planner: user_query -> subtasks JSON
# =========================
from typing import Tuple, Optional

def run_planner(user_query: str) -> Tuple[List[SubTask], bool, str]:
    messages = [
        {"role": "system", "content": prompts.PLANNER_SYSTEM},
        {"role": "user", "content": prompts.PLANNER_USER_TEMPLATE.format(user_query=user_query)},
    ]
    resp = chat(messages)
    text = resp["text"].strip()

    try:
        plan = json.loads(text)

        # add: early reject 
        if plan.get("rejected") is True:
            reason = (plan.get("reason") or "Invalid request").strip()
            return [], True, reason

        subtasks = plan.get("subtasks", [])
        cleaned: List[SubTask] = []
        for i, t in enumerate(subtasks):
            cleaned.append(
                SubTask(
                    id=t.get("id") or f"T{i+1}",
                    intent=t.get("intent") or "other",
                    question=t.get("question") or "",
                    search_query=t.get("search_query") or t.get("question") or user_query,
                )
            )

        if not cleaned:
            raise ValueError("empty_plan")

        return cleaned, False, ""

    except Exception:
        # fallback
        fallback = [
            SubTask(
                id="T1",
                intent="internal_similar_projects",
                question="What similar internal initiatives exist and what were their outcomes?",
                search_query=user_query,
            ),
            SubTask(
                id="T2",
                intent="lessons_learned",
                question="What key lessons learned and failure modes should we consider from internal history?",
                search_query=f"{user_query} postmortem lessons learned latency hallucination monitoring",
            ),
            SubTask(
                id="T3",
                intent="external_examples",
                question="What external examples or best practices exist for this type of solution?",
                search_query=f"{user_query} case study best practices",
            ),
        ]
        return fallback, False, ""


# =========================
# 3) Executor: deterministic tool calls
# =========================
def _internal_retrieve(query: str) -> Tuple[List[Dict[str, Any]], int]:
    searcher = get_internal_search()

    # internal_search dict: {query, results, count}
    res = searcher.search(query)
    internal_results = res.get("results", [])
    # 0 for now
    dropped_c2 = 0
    return internal_results, dropped_c2


def _web_retrieve(query: str) -> Dict[str, Any]:
    return web_search(query, top_k=config.WEB_TOP_K)


def _apply_retrieval_gate_internal(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []

    if not items:
        notes.append("No internal matches found.")
        return [], notes

    # Rank by score
    top1 = items[0].get("score", 0.0)
    accepted = [x for x in items if x.get("score", 0.0) >= config.RETRIEVAL_THRESHOLD]

    if top1 < max(config.RETRIEVAL_THRESHOLD, 0.30):
        notes.append(f"Weak internal evidence (top score={top1:.2f}).")
    return accepted, notes


def _apply_retrieval_gate_web(web_res: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Quality gate for web retrieval, lightweight for now
    """
    notes: List[str] = []
    items = web_res.get("results", []) or []
    cleaned = []
    for r in items:
        if r.get("title") and r.get("url"):
            cleaned.append(r)

    if not cleaned:
        if "error" in web_res:
            notes.append(f"Web search error: {web_res['error']}")
        else:
            notes.append("No web results found.")
    return cleaned, notes


def run_executor(state: AgentState) -> AgentState:

    aggregated_internal: List[Dict[str, Any]] = []
    aggregated_web: List[Dict[str, Any]] = []
    aggregated_notes: List[str] = []
    dropped_c2_total = 0

    for t in state.get("subtasks", []):
        q = (t.get("search_query") or t.get("question") or state["user_query"]).strip()

        # --- Internal retrieval ---
        internal_items, dropped_c2 = _internal_retrieve(q)
        dropped_c2_total += dropped_c2
        internal_items, notes_i = _apply_retrieval_gate_internal(internal_items)

        # --- Web retrieval ---
        need_web = t.get("intent") in {"external_examples", "risks", "recommendation_inputs"}
        if not need_web and (not internal_items):
            need_web = True

        web_items: List[Dict[str, Any]] = []
        notes_w: List[str] = []
        if need_web and config.ENABLE_WEB_SEARCH:
            web_res = _web_retrieve(q)
            web_items, notes_w = _apply_retrieval_gate_web(web_res)

        # --- Evidence pack ---
        evidence: EvidencePack = EvidencePack(
            internal=internal_items,
            web=web_items,
            internal_query=q,
            web_query=q,
            internal_dropped_c2=dropped_c2,
            notes=(notes_i + notes_w),
        )
        t["evidence"] = evidence

        # --- Aggregate evidence ---
        aggregated_internal.extend(internal_items)
        aggregated_web.extend(web_items)
        aggregated_notes.extend(notes_i + notes_w)

    state["aggregated_evidence"] = EvidencePack(
        internal=aggregated_internal,
        web=aggregated_web,
        internal_dropped_c2=dropped_c2_total,
        notes=aggregated_notes,
    )

    # metrics（最小）
    state["metrics"]["internal_count"] = len(aggregated_internal)
    state["metrics"]["web_count"] = len(aggregated_web)
    state["metrics"]["dropped_c2"] = dropped_c2_total
    state["metrics"]["warnings_count"] = len(state.get("warnings", []))

    return state


# =========================
# 4) Summarizer: evidence -> report
# =========================
def run_summarizer(state: AgentState) -> AgentState:

    evidence = state.get("aggregated_evidence", {"internal": [], "web": [], "notes": []})

    internal_json = json.dumps(evidence.get("internal", []), ensure_ascii=False, indent=2)
    web_json = json.dumps(evidence.get("web", []), ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": prompts.SUMMARIZER_SYSTEM},
        {"role": "user", "content": prompts.SUMMARIZER_USER_TEMPLATE.format(
            user_query=state["user_query"],
            internal_evidence_json=internal_json,
            web_evidence_json=web_json,
        )},
    ]

    resp = chat(messages, temperature=0.2)
    state["report_draft"] = resp["text"].strip()
    state["final_report"] = state["report_draft"]
    return state


# =========================
# 5) Guardrails: minimal checks
# =========================
_REQUIRED_HEADERS = [
    "## Problem framing",
    "## Internal similar initiatives",
    "## Key lessons learned",
    "## External examples",
    "## Risks and guardrails",
    "## Recommendation",
    "## Sources",
]


def run_guardrails(state: AgentState) -> AgentState:

    text = state.get("final_report", "")

    # 1) structure check
    for h in _REQUIRED_HEADERS:
        if h not in text:
            state["warnings"].append(f"missing_section:{h}")

    # 2) citation 
    if not re.search(r"\[INT:P\d{3}\]|\[WEB:\d+\]", text):
        state["warnings"].append("missing_citations")

    # 3) C2 leak check (minimal)
    if not state.get("allow_c2", False):
        if "P004" in text or "Sales RAG Agent Architecture" in text:
            state["blocked"] = True
            state["block_reason"] = "confidentiality_violation"
            state["final_report"] = (
                "Output blocked due to confidentiality policy (C2 content detected). "
                "Please run with appropriate access clearance."
            )

    return state


# =========================
# 6) Public API: run pipeline
# =========================
def run_pipeline(user_query: str, user_role: str = "standard") -> Tuple[str, str]:
    """
    Returns:
      final_report (str), trace_path (str)
    """
    trace_id = _new_trace_id()
    allow_c2 = (user_role == "admin") and config.INTERNAL_ALLOW_C2

    state = init_state(user_query=user_query, trace_id=trace_id, user_role=user_role, allow_c2=allow_c2)

    t0 = time.time()

    subtasks, rejected, reason = run_planner(user_query)

    if rejected:
        state["final_report"] = f"Input rejected: {reason}"
        state["warnings"].append("planner_rejected")
        trace_path = _dump_state(state)
        return state["final_report"], str(trace_path)

    state["subtasks"] = subtasks
    
    if not state["subtasks"]:
        state["final_report"] = "Please provide a concrete work-related project idea (1–2 sentences)."
        state["warnings"].append("empty_subtasks")
        state["metrics"]["latency_s"] = round(time.time() - t0, 3)
        trace_path = _dump_state(state)
        return state["final_report"], str(trace_path)

    if state.get("blocked"):
        state["metrics"]["latency_s"] = round(time.time() - t0, 3)
        trace_path = _dump_state(state)
        return state["final_report"], str(trace_path)
    
    state = run_executor(state)

    # Summarizer
    state = run_summarizer(state)

    # Guardrails
    state = run_guardrails(state)

    state["metrics"]["latency_s"] = round(time.time() - t0, 3)

    trace_path = _dump_state(state)
    return state["final_report"], str(trace_path)