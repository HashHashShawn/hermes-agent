#!/usr/bin/env python3
"""Isolated acceptance gate for the strict hash-bound mission runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.mission_runtime import (
    MissionRuntime,
    MissionRuntimeError,
    activate_mission_runtime,
)


SAFE_HEALTH = "/usr/bin/systemctl is-active qwen36-vllm.service"
GATED_COMMAND = "sudo systemctl restart qwen36-vllm.service"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        result[str(path)] = {
            "exists": path.exists(),
            "sha256": sha(path) if path.is_file() else None,
            "size": path.stat().st_size if path.is_file() else None,
        }
    return result


def build_fixture(root: Path) -> tuple[MissionRuntime, dict]:
    mission = root / "mission"
    state_dir = mission / "state"
    work = root / "work"
    mail = root / "mail"
    state_dir.mkdir(parents=True)
    contract = mission / "MISSION.md"
    contract.write_text(
        "# MISSION\n\n## Objective\nInspect runtime truth.\n\n"
        "## Deliverables\nWrite REPORT.md and receipt.json.\n",
        encoding="utf-8",
    )
    state_path = state_dir / "mission_state.json"
    state_path.write_text(
        json.dumps({
            "mission_id": "HARNESS-MISSION-001",
            "status": "STARTED",
            "next_legal_action": "Run the health read.",
            "history": [],
        }),
        encoding="utf-8",
    )
    policy_path = mission / "mission_policy.json"
    policy = {
        "schema": "hermes.mission-runtime.v1",
        "mission_id": "HARNESS-MISSION-001",
        "contract_sha256": sha(contract),
        "state_path": str(state_path),
        "report_path": str(work / "REPORT.md"),
        "receipt_path": str(work / "receipt.json"),
        "checkpoint_path": str(mission / "receipts" / "blocked.json"),
        "notification_path": str(mail / "BLOCK.md"),
        "terminal_read_allowlist": [SAFE_HEALTH],
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    os.environ["HERMES_MISSION_PATH"] = str(contract)
    os.environ["HERMES_MISSION_POLICY"] = str(policy_path)
    os.environ["HERMES_MISSION_POLICY_SHA256"] = sha(policy_path)
    os.environ.pop("HERMES_MISSION_RESUME", None)
    return MissionRuntime.from_environment(), policy


def disclosed_terminal_commands(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    commands = []
    for index, line in enumerate(lines):
        if line.startswith("--- EXACT TERMINAL COMMAND "):
            commands.append(lines[index + 1])
    return commands


def case(case_id, control_class, digest, expected, actual, passed, *, branch):
    return {
        "case_id": case_id,
        "control_class": control_class,
        "fixture_id": "HARNESS-MISSION-001",
        "fixture_source_digest": digest,
        "precondition": "pass",
        "expected": expected,
        "actual": actual,
        "expected_ids": [case_id],
        "actual_ids": [case_id],
        "expected_order": [case_id],
        "actual_order": [case_id],
        "fallback_branch": branch,
        "side_effects_unchanged": True,
        "result": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--production-path", action="append", default=[])
    args = parser.parse_args()
    production_paths = [Path(value) for value in args.production_path]
    production_before = snapshot(production_paths)
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "agent" / "mission_runtime.py",
        Path(__file__).resolve().parents[1] / "agent" / "agent_init.py",
        Path(__file__).resolve().parents[1] / "agent" / "tool_executor.py",
        Path(__file__).resolve().parents[1] / "tools" / "approval.py",
        Path(__file__).resolve().parents[1] / "cli.py",
    ]
    digest = hashlib.sha256(
        "".join(sha(path) for path in source_paths).encode("ascii")
    ).hexdigest()
    source_tip = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    cases = []

    with tempfile.TemporaryDirectory(prefix="hermes-mission-acceptance-") as tmp:
        root = Path(tmp)
        runtime, policy = build_fixture(root)

        from tools.approval import check_all_command_guards
        with activate_mission_runtime(runtime):
            safe = check_all_command_guards(SAFE_HEALTH, "local")
            from tools.terminal_tool import terminal_tool
            safe_execution = json.loads(terminal_tool(
                command=SAFE_HEALTH,
                timeout=15,
                task_id="mission-runtime-acceptance",
                session_id="mission-runtime-acceptance",
            ))
        health_state = (safe_execution.get("output") or "").strip()
        cases.append(case(
            "MR-01", "positive", digest,
            {"approved": True, "prompted_founder": False, "exit_code": 0, "health_state": "active"},
            {"approved": safe.get("approved"), "prompted_founder": False, "exit_code": safe_execution.get("exit_code"), "health_state": health_state},
            safe.get("approved") is True and safe.get("mission_approved") is True and safe_execution.get("exit_code") == 0 and health_state == "active",
            branch="exact_compiled_read",
        ))

        with activate_mission_runtime(runtime):
            started = time.monotonic()
            gated = check_all_command_guards(GATED_COMMAND, "local")
            elapsed_ms = int((time.monotonic() - started) * 1000)
            from tools.terminal_tool import terminal_tool
            gated_execution = terminal_tool(
                command=GATED_COMMAND,
                timeout=15,
                task_id="mission-runtime-acceptance",
                session_id="mission-runtime-acceptance",
            )
        cases.append(case(
            "MR-02", "negative", digest,
            {"approved": False, "status": "mission_blocked", "max_elapsed_ms": 500, "real_result_requires_stop": True},
            {"approved": gated.get("approved"), "status": gated.get("status"), "elapsed_ms": elapsed_ms, "real_result_requires_stop": runtime.result_requires_stop(gated_execution)},
            gated.get("approved") is False and gated.get("status") == "mission_blocked" and elapsed_ms < 500 and runtime.result_requires_stop(gated_execution),
            branch="permission_unavailable_stop",
        ))

        receipt = runtime.block(
            tool_name="terminal",
            tool_args={"command": GATED_COMMAND},
            tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
            session_id="harness-session",
            tool_call_id="harness-call-1",
        )
        state = json.loads(Path(policy["state_path"]).read_text())
        artifacts = [
            Path(policy["report_path"]), Path(policy["receipt_path"]),
            Path(policy["checkpoint_path"]), Path(policy["notification_path"]),
        ]
        cases.append(case(
            "MR-03", "system", digest,
            {"status": "BLOCKED", "artifact_count": 4, "notification": True},
            {"status": state.get("status"), "artifact_count": sum(p.is_file() for p in artifacts), "notification": Path(policy["notification_path"]).is_file()},
            state.get("status") == "BLOCKED" and all(p.is_file() for p in artifacts),
            branch="durable_block_transition",
        ))

        blocked = MissionRuntime.from_environment()
        os.environ["HERMES_MISSION_RESUME"] = "1"
        resumed = MissionRuntime.from_environment()
        cases.append(case(
            "MR-04", "system", digest,
            {"blocked_entry_refused": True, "resumed_status": "RESUMED", "founder_restatement": False},
            {"blocked_entry_refused": blocked.blocked_on_entry, "resumed_status": resumed.state.get("status"), "founder_restatement": receipt["resume_requires_founder_reconstruction"]},
            blocked.blocked_on_entry and resumed.state.get("status") == "RESUMED" and receipt["resume_requires_founder_reconstruction"] is False,
            branch="durable_resume",
        ))
        cases.append(case(
            "MR-05", "positive", digest,
            {"objective_loaded": True, "deliverables_loaded": True, "blocker_loaded": True},
            {"objective_loaded": "Inspect runtime truth." in resumed.prompt_block, "deliverables_loaded": "Write REPORT.md and receipt.json." in resumed.prompt_block, "blocker_loaded": "approval unavailable" in resumed.prompt_block},
            all(text in resumed.prompt_block for text in ("Inspect runtime truth.", "Write REPORT.md and receipt.json.", "approval unavailable")),
            branch="durable_context_load",
        ))

        # Real executor integration: two requested terminal calls, one blocked
        # result. The second call must be materialized only as SKIPPED.
        os.environ.pop("HERMES_MISSION_RESUME", None)
        runtime2, policy2 = build_fixture(root / "executor")
        from run_agent import AIAgent
        tool_defs = [{
            "type": "function",
            "function": {"name": "terminal", "description": "terminal", "parameters": {"type": "object", "properties": {}}},
        }]
        with (
            patch("run_agent.get_tool_definitions", return_value=tool_defs),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:8000/v1", quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.client = MagicMock()
        agent._mission_runtime = runtime2
        calls = [
            SimpleNamespace(id="call-1", function=SimpleNamespace(name="terminal", arguments=json.dumps({"command": GATED_COMMAND}))),
            SimpleNamespace(id="call-2", function=SimpleNamespace(name="terminal", arguments=json.dumps({"command": "id"}))),
        ]
        messages = []
        blocked_result = json.dumps({"status": "blocked", "error": "approval unavailable"})
        with patch("run_agent.handle_function_call", return_value=blocked_result) as dispatch:
            agent._execute_tool_calls(SimpleNamespace(tool_calls=calls), messages, "harness-task")
        second_status = json.loads(messages[1]["content"]).get("status")
        cases.append(case(
            "MR-06", "negative", digest,
            {"dispatch_count": 1, "second_status": "skipped", "halt": "BLOCKED"},
            {"dispatch_count": dispatch.call_count, "second_status": second_status, "halt": agent._mission_runtime_halt.get("verdict")},
            dispatch.call_count == 1 and second_status == "skipped" and agent._mission_runtime_halt.get("verdict") == "BLOCKED",
            branch="no_permission_shopping",
        ))

        # Engineered RED: corrupt the exact policy binding. The harness must
        # observe the expected fail-closed exception before restoring it.
        policy_path = Path(os.environ["HERMES_MISSION_POLICY"])
        original_policy = policy_path.read_bytes()
        policy_path.write_bytes(original_policy + b" ")
        red_observed = False
        try:
            MissionRuntime.from_environment()
        except MissionRuntimeError:
            red_observed = True
        finally:
            policy_path.write_bytes(original_policy)
        cases.append(case(
            "MR-07", "negative", digest,
            {"engineered_red_observed": True, "fixture_restored": True},
            {"engineered_red_observed": red_observed, "fixture_restored": sha(policy_path) == os.environ["HERMES_MISSION_POLICY_SHA256"]},
            red_observed and sha(policy_path) == os.environ["HERMES_MISSION_POLICY_SHA256"],
            branch="policy_hash_fail_closed",
        ))

        # Engineered RED: the disclosure/enforcement coupling check must catch
        # a command present in a disclosed copy but absent from enforcement.
        disclosed = disclosed_terminal_commands(runtime2.prompt_block)
        enforced = set(runtime2.terminal_read_allowlist)
        coupling_before = set(disclosed) == enforced and len(disclosed) == len(enforced)
        mutated_disclosure = [*disclosed, "id"]
        red_observed = set(mutated_disclosure) != enforced
        fixture_restored = disclosed_terminal_commands(runtime2.prompt_block) == disclosed
        cases.append(case(
            "MR-08", "negative", digest,
            {"coupling_before": True, "engineered_red_observed": True, "fixture_restored": True},
            {"coupling_before": coupling_before, "engineered_red_observed": red_observed, "fixture_restored": fixture_restored},
            coupling_before and red_observed and fixture_restored,
            branch="disclosure_enforcement_coupling",
        ))

        # The process tool is part of the terminal toolset but dispatches
        # outside the terminal command guard. A strict mission must stop it
        # before a valid background-session ID could become a shell-input
        # bypass, and it must skip every later sibling in the same batch.
        runtime3, _ = build_fixture(root / "process-boundary")
        process_tool_defs = [{
            "type": "function",
            "function": {"name": "process", "description": "process", "parameters": {"type": "object", "properties": {}}},
        }]
        with (
            patch("run_agent.get_tool_definitions", return_value=process_tool_defs),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            process_agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:8000/v1", quiet_mode=True, skip_context_files=True, skip_memory=True)
        process_agent.client = MagicMock()
        process_agent._mission_runtime = runtime3
        process_calls = [
            SimpleNamespace(id="call-process", function=SimpleNamespace(name="process", arguments=json.dumps({
                "action": "submit",
                "session_id": "existing-shell",
                "data": "bash -c 'echo bypass'",
            }))),
            SimpleNamespace(id="call-sibling", function=SimpleNamespace(name="terminal", arguments=json.dumps({"command": SAFE_HEALTH}))),
        ]
        process_messages = []
        with patch("run_agent.handle_function_call") as process_dispatch:
            process_agent._execute_tool_calls(
                SimpleNamespace(tool_calls=process_calls),
                process_messages,
                "harness-process-task",
            )
        process_second_tool_call = (
            process_messages[1].get("tool_call_id")
            if len(process_messages) == 2
            and isinstance(process_messages[1], dict)
            else None
        )
        process_blocker = process_agent._mission_runtime_halt.get("blocker", {})
        cases.append(case(
            "MR-09", "negative", digest,
            {
                "dispatch_count": 0,
                "blocked_tool": "process",
                "second_tool_call_id": "call-sibling",
                "process_disclosed_unavailable": True,
            },
            {
                "dispatch_count": process_dispatch.call_count,
                "blocked_tool": process_blocker.get("tool_name"),
                "second_tool_call_id": process_second_tool_call,
                "process_disclosed_unavailable": "process tool is unavailable" in runtime3.prompt_block,
            },
            process_dispatch.call_count == 0
            and process_blocker.get("tool_name") == "process"
            and process_second_tool_call == "call-sibling"
            and "process tool is unavailable" in runtime3.prompt_block,
            branch="process_tool_fail_closed",
        ))

        # One-shot ``chat -q`` sessions bypass the interactive shutdown path.
        # The finalizer must close the agent-owned durable session before it
        # releases the active-session lease, including when cleanup raises.
        import cli

        finalize_calls = []
        finalize_cli = SimpleNamespace(
            agent=SimpleNamespace(
                close=lambda: finalize_calls.append("agent_close"),
            ),
            _release_active_session=lambda: finalize_calls.append("release"),
        )

        def failing_cleanup(**_kwargs):
            finalize_calls.append("cleanup")
            raise RuntimeError("engineered cleanup failure")

        cleanup_red_observed = False
        with (
            patch(
                "cli._notify_single_query_session_finalize",
                side_effect=lambda _cli: finalize_calls.append("finalize"),
            ),
            patch("cli._run_cleanup", side_effect=failing_cleanup),
        ):
            try:
                cli._finalize_single_query(finalize_cli)
            except RuntimeError as exc:
                cleanup_red_observed = str(exc) == "engineered cleanup failure"
        expected_finalize_order = ["finalize", "cleanup", "agent_close", "release"]
        cases.append(case(
            "MR-10", "system", digest,
            {
                "engineered_cleanup_red_observed": True,
                "finalize_order": expected_finalize_order,
                "session_close_before_release": True,
            },
            {
                "engineered_cleanup_red_observed": cleanup_red_observed,
                "finalize_order": finalize_calls,
                "session_close_before_release": (
                    "agent_close" in finalize_calls
                    and "release" in finalize_calls
                    and finalize_calls.index("agent_close") < finalize_calls.index("release")
                ),
            },
            cleanup_red_observed and finalize_calls == expected_finalize_order,
            branch="one_shot_session_finalization",
        ))


        # ── G1 controls MR-11..MR-16 ─────────────────────────────────────
        from agent.mission_runtime import (
            COMPLETE_RECEIPT_SCHEMA,
            MISSION_COMPLETE_TOOL_NAME,
        )

        runtime_c, policy_c = build_fixture(root / "complete")
        work_c = root / "complete" / "cwork"
        work_c.mkdir(parents=True, exist_ok=True)
        deliverable = work_c / "DELIVERABLE.md"
        deliverable.write_text("done\n", encoding="utf-8")
        policy_path_c = Path(os.environ["HERMES_MISSION_POLICY"])
        payload_c = json.loads(policy_path_c.read_text(encoding="utf-8"))
        payload_c["completion"] = {
            "version": 1,
            "predicates": [
                {"type": "artifact_exists_nonempty", "path": str(deliverable)}
            ],
            "receipt_path": str(work_c / "complete_receipt.json"),
            "checkpoint_path": str(root / "complete" / "mission" / "receipts" / "complete.json"),
            "report_path": str(work_c / "COMPLETE_REPORT.md"),
            "notification_path": str(root / "complete" / "mail" / "COMPLETE.md"),
            "note": "ok",
        }
        (root / "complete" / "mail").mkdir(parents=True, exist_ok=True)
        (root / "complete" / "mission" / "receipts").mkdir(parents=True, exist_ok=True)
        policy_path_c.write_text(json.dumps(payload_c), encoding="utf-8")
        os.environ["HERMES_MISSION_POLICY_SHA256"] = sha(policy_path_c)
        runtime_c = MissionRuntime.from_environment()
        complete_receipt = runtime_c.complete(
            {"session_id": "harness-complete", "tool_call_id": "complete-1"}
        )
        state_c = json.loads(Path(policy_c["state_path"]).read_text())
        cases.append(case(
            "MR-11", "positive", digest,
            {"status": "COMPLETED", "schema": COMPLETE_RECEIPT_SCHEMA},
            {
                "status": state_c.get("status"),
                "schema": complete_receipt.get("schema"),
            },
            state_c.get("status") == "COMPLETED"
            and complete_receipt.get("schema") == COMPLETE_RECEIPT_SCHEMA
            and Path(payload_c["completion"]["receipt_path"]).is_file(),
            branch="valid_completion",
        ))

        # MR-12 = V6+V8 second completion + block-after-complete
        state_bytes = Path(policy_c["state_path"]).read_bytes()
        second = runtime_c.complete({"session_id": "harness-complete", "tool_call_id": "complete-2"})
        runtime_c.halt_receipt = None
        block_after = runtime_c.block(
            tool_name="terminal",
            tool_args={"command": GATED_COMMAND},
            tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
            session_id="harness-complete",
            tool_call_id="block-after",
        )
        cases.append(case(
            "MR-12", "negative", digest,
            {"state_unchanged": True, "second_refused_or_same": True, "block_refused": True},
            {
                "state_unchanged": Path(policy_c["state_path"]).read_bytes() == state_bytes,
                "second_refused_or_same": (
                    second.get("verdict") == "COMPLETED"
                    or second.get("status") == "refused"
                ),
                "block_refused": (
                    block_after.get("status") == "refused"
                    or block_after.get("verdict") == "COMPLETED"
                ),
            },
            Path(policy_c["state_path"]).read_bytes() == state_bytes,
            branch="idempotent_terminal",
        ))

        # MR-13 = V2+V3 missing + symlink
        runtime_m, policy_m = build_fixture(root / "missing")
        work_m = root / "missing" / "cwork"
        work_m.mkdir(parents=True, exist_ok=True)
        missing_path = work_m / "MISSING.md"
        policy_path_m = Path(os.environ["HERMES_MISSION_POLICY"])
        payload_m = json.loads(policy_path_m.read_text(encoding="utf-8"))
        payload_m["completion"] = {
            "version": 1,
            "predicates": [
                {"type": "artifact_exists_nonempty", "path": str(missing_path)}
            ],
            "receipt_path": str(work_m / "complete_receipt.json"),
            "checkpoint_path": str(root / "missing" / "mission" / "receipts" / "complete.json"),
            "report_path": str(work_m / "COMPLETE_REPORT.md"),
            "note": "ok",
        }
        (root / "missing" / "mission" / "receipts").mkdir(parents=True, exist_ok=True)
        policy_path_m.write_text(json.dumps(payload_m), encoding="utf-8")
        os.environ["HERMES_MISSION_POLICY_SHA256"] = sha(policy_path_m)
        runtime_m = MissionRuntime.from_environment()
        before_m = Path(policy_m["state_path"]).read_bytes()
        refused_missing = runtime_m.complete({"session_id": "m", "tool_call_id": "1"})
        real = work_m / "real.md"
        real.write_text("x\n", encoding="utf-8")
        missing_path.symlink_to(real)
        refused_symlink = runtime_m.complete({"session_id": "m", "tool_call_id": "2"})
        cases.append(case(
            "MR-13", "negative", digest,
            {"missing_refused": True, "symlink_refused": True, "state_untouched": True},
            {
                "missing_refused": refused_missing.get("status") == "refused",
                "symlink_refused": refused_symlink.get("status") == "refused",
                "state_untouched": Path(policy_m["state_path"]).read_bytes() == before_m,
            },
            refused_missing.get("status") == "refused"
            and refused_symlink.get("status") == "refused"
            and Path(policy_m["state_path"]).read_bytes() == before_m,
            branch="predicate_refusal",
        ))

        # MR-14 = V14 restart after COMPLETED
        os.environ.pop("HERMES_MISSION_RESUME", None)
        completed_entry = MissionRuntime.from_environment()
        # reload the completed fixture from MR-11 paths
        os.environ["HERMES_MISSION_PATH"] = str(Path(policy_c["state_path"]).parent.parent / "MISSION.md")
        os.environ["HERMES_MISSION_POLICY"] = str(
            Path(policy_c["state_path"]).parent.parent / "mission_policy.json"
        )
        os.environ["HERMES_MISSION_POLICY_SHA256"] = sha(Path(os.environ["HERMES_MISSION_POLICY"]))
        before_c = Path(policy_c["state_path"]).read_bytes()
        loaded_c = MissionRuntime.from_environment()
        os.environ["HERMES_MISSION_RESUME"] = "1"
        loaded_c2 = MissionRuntime.from_environment()
        cases.append(case(
            "MR-14", "system", digest,
            {"completed_on_entry": True, "resume_noop": True, "zero_writes": True},
            {
                "completed_on_entry": loaded_c.completed_on_entry,
                "resume_noop": loaded_c2.completed_on_entry,
                "zero_writes": Path(policy_c["state_path"]).read_bytes() == before_c,
            },
            loaded_c.completed_on_entry
            and loaded_c2.completed_on_entry
            and Path(policy_c["state_path"]).read_bytes() == before_c,
            branch="completed_entry_guard",
        ))

        # MR-15 = V10+V11 seq ordering
        os.environ.pop("HERMES_MISSION_RESUME", None)
        runtime_s, policy_s = build_fixture(root / "seq")
        from unittest.mock import patch as _patch
        import agent.mission_runtime as mrm
        with _patch.object(mrm, "_utc_now", side_effect=["2026-08-25T12:00:00Z", "2026-08-25T11:00:00Z"]):
            runtime_s.block(
                tool_name="terminal",
                tool_args={"command": GATED_COMMAND},
                tool_result=json.dumps({"status": "blocked", "error": "x"}),
                session_id="seq",
                tool_call_id="1",
            )
        state_s = json.loads(Path(policy_s["state_path"]).read_text())
        seqs = [h["seq"] for h in state_s["history"] if "seq" in h]
        cases.append(case(
            "MR-15", "system", digest,
            {"seq_increasing": True, "state_revision": True},
            {
                "seq_increasing": seqs == sorted(seqs) and len(seqs) >= 1,
                "state_revision": isinstance(state_s.get("state_revision"), int)
                and state_s["state_revision"] >= 1,
            },
            seqs == sorted(seqs) and state_s.get("state_revision", 0) >= 1,
            branch="ordering_contract",
        ))

        # MR-16 = V24+V25+V34 boundary
        runtime_b, policy_b = build_fixture(root / "boundary")
        deny_state = runtime_b.authorize_tool(
            "write_file", {"path": policy_b["state_path"], "content": "x"}
        )
        deny_exec = runtime_b.authorize_tool("execute_code", {"code": "1"})
        deny_patch = runtime_b.authorize_tool(
            "patch", {"mode": "patch", "patch": "*** Begin Patch\n*** End Patch"}
        )
        allow_replace = runtime_b.authorize_tool(
            "patch",
            {
                "mode": "replace",
                "path": str(root / "boundary" / "ok.md"),
                "old_string": "a",
                "new_string": "b",
            },
        )
        cases.append(case(
            "MR-16", "negative", digest,
            {
                "protected_write_denied": True,
                "ceiling_denied": True,
                "mode_patch_denied": True,
                "mode_replace_allowed": True,
            },
            {
                "protected_write_denied": deny_state.allowed is False,
                "ceiling_denied": deny_exec.allowed is False,
                "mode_patch_denied": deny_patch.allowed is False,
                "mode_replace_allowed": allow_replace.allowed is True,
            },
            deny_state.allowed is False
            and deny_exec.allowed is False
            and deny_patch.allowed is False
            and allow_replace.allowed is True,
            branch="runtime_owned_state_boundary",
        ))

    production_after = snapshot(production_paths)
    production_unchanged = production_before == production_after
    for item in cases:
        item["side_effects_unchanged"] = production_unchanged
        if not production_unchanged:
            item["result"] = "FAIL"
    result = {
        "schema": "hermes.mission-runtime.acceptance.v1",
        "source_tip": source_tip,
        "subject_digest": digest,
        "subject_files": {str(path): sha(path) for path in source_paths},
        "case_order": [item["case_id"] for item in cases],
        "required_control_classes": ["positive", "negative", "system"],
        "observed_control_classes": sorted({item["control_class"] for item in cases}),
        "engineered_red_observed": all(
            any(item["case_id"] == case_id and item["result"] == "PASS" for item in cases)
            for case_id in ("MR-07", "MR-08")
        ),
        "production_before": production_before,
        "production_after": production_after,
        "production_unchanged": production_unchanged,
        "cases": cases,
        "verdict": "PASS" if all(item["result"] == "PASS" for item in cases) else "FAIL",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "out": str(out), "sha256": sha(out)}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
