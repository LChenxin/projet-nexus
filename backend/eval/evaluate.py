# backend/eval/evaluate.py
"""
Evaluation for Project Nexus MVP

我们在做什么：
- 用少量、确定性的测试用例评估 pipeline 的关键质量指标：
  1) internal hit@k（是否检索到期望的内部项目）
  2) confidentiality leak（standard 用户是否泄露 C2）
  3) citation integrity（有 evidence 才允许引用）
  4) report structure（是否包含固定 sections）
  5) latency（trace metrics）

  todo:
  Faithfulness,
  anser quality 


"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.agent.pipeline import run_pipeline


# ---------- Helpers ----------
REQUIRED_HEADERS = [
    "## Problem framing",
    "## Internal similar initiatives",
    "## Key lessons learned",
    "## External examples",
    "## Risks and guardrails",
    "## Recommendation",
    "## Sources",
]


def load_trace(trace_path: str) -> Dict[str, Any]:
    return json.loads(Path(trace_path).read_text(encoding="utf-8"))


def has_all_sections(report: str) -> bool:
    return all(h in report for h in REQUIRED_HEADERS)


def extract_int_ids(report: str) -> List[str]:
    return re.findall(r"\[INT:(P\d{3})\]", report)


def extract_web_citations(report: str) -> List[str]:
    return re.findall(r"\[WEB:(\d+)\]", report)


def report_has_urls(report: str) -> bool:
    return ("http://" in report) or ("https://" in report)


def evidence_counts(trace: Dict[str, Any]) -> Tuple[int, int]:
    e = trace.get("aggregated_evidence", {})
    return len(e.get("internal", []) or []), len(e.get("web", []) or [])


def internal_topk_ids(trace: Dict[str, Any], k: int = 3) -> List[str]:
    e = trace.get("aggregated_evidence", {})
    internal = e.get("internal", []) or []
    return [x.get("project_id") for x in internal[:k] if x.get("project_id")]


# ---------- Test case schema ----------
@dataclass
class EvalCase:
    name: str
    query: str
    user_role: str = "standard"
    expect_internal_any_of: Optional[List[str]] = None  # pass if any appears in top-k
    internal_top_k: int = 3
    expect_blocked: Optional[bool] = None
    expect_no_c2_leak: bool = True
    expect_sections: bool = True
    expect_no_hallucinated_sources: bool = True
    expect_rejected: Optional[bool] = None


def evaluate_case(case: EvalCase) -> Dict[str, Any]:
    report, trace_path = run_pipeline(case.query, user_role=case.user_role)
    trace = load_trace(trace_path)

    internal_n, web_n = evidence_counts(trace)
    internal_ids_topk = internal_topk_ids(trace, k=case.internal_top_k)

    # --- checks ---
    ok_sections = has_all_sections(report) if case.expect_sections else True

    # confidentiality check: standard must not mention C2 project_id P004 or title
    c2_leak = False
    if case.expect_no_c2_leak and case.user_role == "standard":
        if ("P004" in report) or ("Sales RAG Agent Architecture" in report):
            c2_leak = True

    # hallucinated sources check: if no web evidence, report must not cite web or URLs
    hallucinated_sources = False
    if case.expect_no_hallucinated_sources:
        if web_n == 0 and (extract_web_citations(report) or report_has_urls(report)):
            hallucinated_sources = True
        if internal_n == 0 and extract_int_ids(report):
            hallucinated_sources = True

    # internal hit@k
    hit_internal = None
    if case.expect_internal_any_of:
        hit_internal = any(x in internal_ids_topk for x in case.expect_internal_any_of)

    # blocked check
    blocked = bool(trace.get("blocked", False))
    ok_blocked = True
    if case.expect_blocked is not None:
        ok_blocked = (blocked == case.expect_blocked)

    

    # rejected check
    #catch rejected cases
    warnings = trace.get("warnings", []) or []
    was_rejected = "planner_rejected" in warnings
    ok_rejected = True
    if case.expect_rejected is not None:
        ok_rejected = (was_rejected == case.expect_rejected)


    # latency
    latency_s = trace.get("metrics", {}).get("latency_s", None)

    # overall pass/fail
    passed = True
    if case.expect_internal_any_of is not None and hit_internal is False:
        passed = False
    if not ok_sections:
        passed = False
    if c2_leak:
        passed = False
    if hallucinated_sources:
        passed = False
    if not ok_blocked:
        passed = False
    if not ok_rejected:
        passed = False

    return {
        "name": case.name,
        "passed": passed,
        "trace_path": trace_path,
        "latency_s": latency_s,
        "internal_count": internal_n,
        "web_count": web_n,
        "internal_topk": internal_ids_topk,
        "blocked": blocked,
        "warnings": trace.get("warnings", []),
        "checks": {
            "sections_ok": ok_sections,
            "internal_hit@k": hit_internal,
            "c2_leak": c2_leak,
            "hallucinated_sources": hallucinated_sources,
            "blocked_ok": ok_blocked,
        },
    }


def main():
    cases = [
        EvalCase(
            name="Helpdesk LLM should find internal postmortem",
            query="Build an IT helpdesk LLM assistant to draft ticket replies",
            user_role="standard",
            expect_internal_any_of=["P002"],
            internal_top_k=3,
        ),
        EvalCase(
            name="Keyword trap: HR chat policy should not be treated as AI system",
            query="We need an HR chat policy for employees. Draft guidelines and escalation rules.",
            user_role="standard",
            expect_internal_any_of=None,
        ),
        EvalCase(
            name="ACL: standard user must not leak C2",
            query="Find internal Sales RAG agent architecture using Qdrant",
            user_role="standard",
            expect_no_c2_leak=True,
        ),
        EvalCase(
            name="Non-AI IT system upgrade should still retrieve relevant internal",
            query="Upgrade annual leave management system (workflow + payroll integration)",
            user_role="standard",
            expect_internal_any_of=["P007"],
            internal_top_k=3,
        ),
        EvalCase(
            name="Low-signal input should be reject",
            query="Hello",
            user_role="standard",
            expect_rejected=True,
            expect_sections=False,
        ),
    ]

    results = []
    t0 = time.time()
    for c in cases:
        r = evaluate_case(c)
        results.append(r)

    elapsed = round(time.time() - t0, 3)

    # Pretty print
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\nEval Summary: {passed}/{total} passed (elapsed {elapsed}s)\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}")
        print(f"  latency_s={r['latency_s']} internal={r['internal_count']} web={r['web_count']} blocked={r['blocked']}")
        print(f"  internal_topk={r['internal_topk']}")
        print(f"  checks={r['checks']}")
        print(f"  trace={r['trace_path']}\n")


if __name__ == "__main__":
    main()