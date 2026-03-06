# backend/agent/pipeline.py

"""
Pipeline single DAG
1) planner: user_query -> subtasks (id/intent/question/search_query)
2) executor: for each subtask execute internal_search / web_search,get evidence
3) summarizer:  evidence to brief(with citations)
4) guardrails: Minimal checks on the final report (structure, citations, C2 leak)
5) logging: dump state to logs/{trace_id}.json

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
# 1) Utility
# =========================
def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _dump_state(state: AgentState) -> Path:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOGS_DIR / f"{state['trace_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return out_path

def _add_event(state: AgentState, step: str, status: str, message: str) -> None:
    state.setdefault("events", []).append(
        {
            "step": step,
            "status": status,
            "message": message,
            "ts": round(time.time(), 3),
        }
    )

def _finalize_and_dump(state: AgentState, t0: float) -> Tuple[str, str]:
    """
    Finalize common metrics, append pipeline completion event, dump trace,
    and return final_report + trace path.
    """
    state["metrics"]["latency_s"] = round(time.time() - t0, 3)
    state["metrics"]["warnings_count"] = len(state.get("warnings", []))

    _add_event(state, "pipeline", "completed", "Run finished")
    trace_path = _dump_state(state)
    return state.get("final_report", ""), str(trace_path)



# =========================
# 2) Planner
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

    except Exception as e:
        print(f"[Demo Debug] Planner error: {e}")
        print(f"[Demo Debug] Raw output: {text}")
        return [], True, "Failed to parse planner output into valid JSON."

# =========================
# 3) Executor
# =========================
def _internal_retrieve(query: str, allow_c2: bool) -> Tuple[List[Dict[str, Any]], int]:
    try:
        searcher = get_internal_search()
        res = searcher.search(query,allow_c2=allow_c2)
        internal_results = res.get("results", [])
        dropped_c2 = res.get("dropped_c2", 0) # Add: count C2
        return internal_results, dropped_c2
    except Exception as e:
        print(f"[Demo Debug] Internal search failed for '{query}': {e}")
        return [], 0


def _web_retrieve(query: str) -> Dict[str, Any]:
    try:
        return web_search(query, top_k=config.WEB_TOP_K)
    except Exception as e:
        print(f"[Demo Debug] Web search failed for '{query}': {e}")
        return {"results": [], "error": str(e)}


def _apply_retrieval_gate_internal(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []

    if not items:
        notes.append("No internal matches found.")
        return [], notes

    # Rank by score
    top1 = items[0].get("score", 0.0)
    accepted = items

    if top1 < max(config.RETRIEVAL_THRESHOLD, 0.30):
        notes.append(f"Weak internal evidence (top score={top1:.2f}).")
    return accepted, notes


def _apply_retrieval_gate_web(web_res: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    lightweight quality gate for web retrieval
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
        allow_c2 = state.get("allow_c2", False)
        internal_items, dropped_c2 = _internal_retrieve(q, allow_c2=allow_c2) 
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

    unique_internal = list({item.get("project_id"): item for item in aggregated_internal if item.get("project_id")}.values())
    unique_web = list({item.get("url"): item for item in aggregated_web if item.get("url")}.values())  
    unique_notes = list(dict.fromkeys(aggregated_notes))
    state["aggregated_evidence"] = EvidencePack(
        internal=unique_internal,
        web=unique_web,
        internal_dropped_c2=dropped_c2_total,
        notes=unique_notes,
    )

    state["metrics"]["internal_count"] = len(unique_internal)
    state["metrics"]["web_count"] = len(unique_web)
    state["metrics"]["dropped_c2"] = dropped_c2_total
    state["metrics"]["warnings_count"] = len(state.get("warnings", []))

    return state


# =========================
# 4) Summarizer
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
# 5) Guardrails (minimal checks)
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


# ====================
# 6) API
# ====================

def plan_pipeline(user_query: str, user_role: str = "standard") -> AgentState:
    """
    Stage 1 only:
    - initialize state
    - run planner
    - store subtasks in state
    - record planner events / warnings

    This allows the frontend to visualize subtasks before continuing.
    """
    trace_id = _new_trace_id()
    allow_c2 = (user_role == "admin") and config.INTERNAL_ALLOW_C2
    state = init_state(user_query=user_query, trace_id=trace_id, user_role=user_role, allow_c2=allow_c2)

    _add_event(state, "planner", "started", "Planning subtasks")
    subtasks, rejected, reason = run_planner(user_query)

    # planner hard rejection
    if rejected:
        _add_event(state, "planner", "failed", f"Planner rejected input: {reason}")
        state["final_report"] = f"Input rejected: {reason}"
        state["warnings"].append("planner_rejected")
        return state
    
    
    state["subtasks"] = subtasks
    _add_event(state, "planner", "completed", f"Generated {len(subtasks)} subtasks")

    if not state["subtasks"]:
        state["warnings"].append("empty_subtasks")
        state["final_report"] = "Please provide a concrete work-related project idea (1–2 sentences)."

    return state

def continue_pipeline(state: AgentState) -> Tuple[str, str]:
    """
    Continue execution from an already-initialized/planned state.
    This is the backend API the frontend should call after visualizing subtasks.
    """
    t0 = time.time()

    # Early exits if planning already ended the run
    if state.get("warnings") and "planner_rejected" in state.get("warnings", []):
        return _finalize_and_dump(state, t0)

    if not state.get("subtasks"):
        _add_event(state, "executor", "failed", "No valid subtasks generated")
        if not state.get("final_report"):
            state["final_report"] = "Please provide a concrete work-related project idea (1–2 sentences)."
        return _finalize_and_dump(state, t0)

    if state.get("blocked"):
        _add_event(state, "executor", "failed", f"Execution blocked: {state.get('block_reason', '')}")
        return _finalize_and_dump(state, t0)

    _add_event(state, "executor", "started", "Running retrieval")
    state = run_executor(state)
    _add_event(
        state,
        "executor",
        "completed",
        f"Retrieved {state['metrics']['internal_count']} internal and {state['metrics']['web_count']} web results",
    )

    _add_event(state, "summarizer", "started", "Drafting report")
    state = run_summarizer(state)
    _add_event(state, "summarizer", "completed", "Report drafted")

    _add_event(state, "guardrails", "started", "Checking output safety and structure")
    state = run_guardrails(state)

    if state.get("blocked"):
        _add_event(state, "guardrails", "failed", f"Output blocked: {state.get('block_reason', 'unknown')}")
    else:
        _add_event(state, "guardrails", "completed", "Checks passed")

    return _finalize_and_dump(state, t0)

def run_from_state(state: AgentState) -> Tuple[str, str]:
    return continue_pipeline(state)

def run_pipeline(user_query: str, user_role: str = "standard") -> Tuple[str, str]:
    """
    Full end-to-end API:
      1) plan
      2) continue execution

    Returns:
      final_report (str), trace_path (str)
    """
    # >>> CHANGED
    state = plan_pipeline(user_query=user_query, user_role=user_role)
    return continue_pipeline(state)


    