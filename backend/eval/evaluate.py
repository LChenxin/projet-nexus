#evaluate.py

import json
import pytest
from pathlib import Path
from backend.agent.pipeline import run_pipeline

# --- Helper ---
def get_trace(trace_path: str) -> dict:
    return json.loads(Path(trace_path).read_text(encoding="utf-8"))

def get_internal_ids(trace: dict) -> list:
    evidence = trace.get("aggregated_evidence", {}).get("internal", [])
    return [item.get("project_id") for item in evidence if item.get("project_id")]

# --- Tests ---

def test_happy_path_retrieval_and_structure():
    """ TEST Pipeline: Normal query should retrieve relevant internal project and structure report correctly"""
    report, trace_path = run_pipeline("Build an IT helpdesk LLM assistant to draft ticket replies", user_role="standard")
    trace = get_trace(trace_path)

    # 1. State assertions
    assert not trace.get("blocked"), "Normal query should not be blocked"
    assert "planner_rejected" not in trace.get("warnings", []), "Normal query should not be rejected by planner"

    # 2. Recall assertions
    retrieved_ids = get_internal_ids(trace)
    assert "P002" in retrieved_ids, f"Expected to retrieve P002, actually retrieved: {retrieved_ids}"

    # 3. Structure assertions
    assert "## Recommendation" in report
    assert "## Sources" in report

def test_c2_data_silently_filtered_for_standard_user():
    """ TEST Unauthorized Access : C2 data should be silently filtered for standard users"""
    report, trace_path = run_pipeline("Find internal Sales RAG agent architecture", user_role="standard")
    trace = get_trace(trace_path)

    # 1. Physical isolation assertions 
    retrieved_ids = get_internal_ids(trace)
    assert "P004" not in retrieved_ids, "Internal search should not retrieve C2 project P004 for standard user"

    # 2. State assertions (should not trigger Block, but return normally to prevent side-channel inference)
    assert trace.get("blocked") is False, "Underlying data has been filtered, Guardrail should not be triggered"
    
    # 3. Content assertions (final report should not contain confidential keywords)
    assert "P004" not in report, "Report should not contain P004"
    assert "Sales RAG Agent Architecture" not in report, "Report should not contain confidential title"

def test_c2_data_accessible_for_admin():
    """ TEST Privileged Access : Admin should be able to access C2 data"""
    report, trace_path = run_pipeline("Find internal Sales RAG agent architecture", user_role="admin")
    trace = get_trace(trace_path)

    # 1. Access assertions
    assert not trace.get("blocked"), "Admin permissions should not be blocked"
    
    # 2. Recall and citation assertions
    assert "P004" in get_internal_ids(trace), "Admin should be able to retrieve P004"
    assert "[INT:P004]" in report, "Report should contain correct reference label for C2 project"

def test_early_reject_meaningless_input():
    """ TEST Early Reject : Meaningless input should be rejected by Planner"""
    report, trace_path = run_pipeline("Hello asdfghjkl", user_role="standard")
    trace = get_trace(trace_path)

    # 1. Reject assertions
    assert "planner_rejected" in trace.get("warnings", [])
    assert report.startswith("Input rejected:")
    
    # 2. Resource consumption assertions (should not execute Executor)
    assert trace.get("metrics", {}).get("internal_count", 0) == 0

def test_keyword_trap_hr_chat_policy():
    """ TEST Keyword Trap : HR Chat Policy should not be mistaken for an AI agent project"""
    report, trace_path = run_pipeline(
        "We need HR chat policy guidelines for employees",
        user_role="standard"
    )
    trace = get_trace(trace_path)

    retrieved_ids = get_internal_ids(trace)

    # Should be able to retrieve P003, as it is indeed relevant
    assert "P003" in retrieved_ids

    # But should not classify it as an AI/LLM system
    assert "not a software project" in report.lower() or "policy" in report.lower()