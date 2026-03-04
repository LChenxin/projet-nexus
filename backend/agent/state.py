# backend/agent/state.py

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict


# =========================
# 1) Evidence schemas
# =========================
class InternalEvidenceItem(TypedDict, total=False):
    project_id: str
    title: str
    score: float
    summary: str
    status: str
    confidentiality: str 


class WebEvidenceItem(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    source: Literal["web"]


class EvidencePack(TypedDict, total=False):
    internal: List[InternalEvidenceItem]
    web: List[WebEvidenceItem]

    #  debug / governance
    internal_query: str
    web_query: str
    internal_dropped_c2: int 
    notes: List[str]
    sources_flat: List[Dict[str, Any]]


# =========================
# 2) Task decomposition
# =========================
class SubTask(TypedDict, total=False):
    id: str                
    question: str           
    intent: Literal[
        "internal_similar_projects",
        "lessons_learned",
        "external_examples",
        "risks",
        "recommendation_inputs",
        "other",
    ]
    search_query: str      # optimized query for retrieval
    evidence: EvidencePack  

# =========================
# 3) Main State
# =========================
class AgentState(TypedDict, total=False):
    # ---- Run identity ----
    trace_id: str
    user_query: str

    # ---- Access control (mock) ----
    user_role: Literal["standard", "admin"]
    allow_c2: bool

    # ---- Planning ----
    subtasks: List[SubTask]

    # ---- Aggregated evidence ----
    aggregated_evidence: EvidencePack

    # ---- Drafts & final output ----
    report_draft: str
    final_report: str

    # ---- Guardrails / validation ----
    warnings: List[str]     
    blocked: bool
    block_reason: str

    # ---- Observability ----
    errors: List[str]       
    metrics: Dict[str, Any] 


# =========================
# 4) State factory helpers
# =========================
def init_state(
    user_query: str,
    trace_id: str,
    user_role: Literal["standard", "admin"] = "standard",
    allow_c2: bool = False,
) -> AgentState:
    return AgentState(
        trace_id=trace_id,
        user_query=user_query,
        user_role=user_role,   # "standard" by default
        allow_c2=allow_c2,
        subtasks=[],
        aggregated_evidence={"internal": [], "web": [], "notes": []},
        report_draft="",
        final_report="",
        warnings=[],
        blocked=False,
        errors=[],
        metrics={
            "internal_count": 0,
            "web_count": 0,
            "dropped_c2": 0,
            "warnings_count": 0,
        },
    )