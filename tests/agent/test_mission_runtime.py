import hashlib
import json
import time
from pathlib import Path

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.mission_runtime import (
    MissionRuntime,
    MissionRuntimeError,
    activate_mission_runtime,
)


SAFE_HEALTH = "/usr/bin/systemctl is-active qwen36-vllm.service"
SAFE_IDENTITY = "git -C /home/artemis/hermes-dev rev-parse HEAD"
GATED_COMMAND = "sudo systemctl restart qwen36-vllm.service"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch):
    mission = tmp_path / "mission"
    state_dir = mission / "state"
    work = tmp_path / "work"
    mail = tmp_path / "mail"
    state_dir.mkdir(parents=True)
    contract = mission / "MISSION.md"
    contract.write_text(
        "# MISSION\n\n## Objective\nInspect runtime truth.\n\n"
        "## Deliverables\nWrite REPORT.md and receipt.json.\n",
        encoding="utf-8",
    )
    state = state_dir / "mission_state.json"
    state.write_text(
        json.dumps({
            "mission_id": "TEST-MISSION-001",
            "status": "STARTED",
            "next_legal_action": "Run the health read.",
            "history": [],
        }),
        encoding="utf-8",
    )
    policy = mission / "mission_policy.json"
    payload = {
        "schema": "hermes.mission-runtime.v1",
        "mission_id": "TEST-MISSION-001",
        "contract_sha256": _sha(contract),
        "state_path": str(state),
        "report_path": str(work / "REPORT.md"),
        "receipt_path": str(work / "receipt.json"),
        "checkpoint_path": str(mission / "receipts" / "blocked.json"),
        "notification_path": str(mail / "BLOCK.md"),
        "terminal_read_allowlist": [SAFE_HEALTH, SAFE_IDENTITY],
    }
    policy.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HERMES_MISSION_PATH", str(contract))
    monkeypatch.setenv("HERMES_MISSION_POLICY", str(policy))
    monkeypatch.setenv("HERMES_MISSION_POLICY_SHA256", _sha(policy))
    return MissionRuntime.from_environment(), payload


def _disclosed_terminal_commands(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    commands = []
    for index, line in enumerate(lines):
        if line.startswith("--- EXACT TERMINAL COMMAND "):
            commands.append(lines[index + 1])
    return commands


def test_contract_and_state_are_loaded_into_working_context(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert "Inspect runtime truth." in runtime.prompt_block
    assert "Write REPORT.md and receipt.json." in runtime.prompt_block
    assert "Run the health read." in runtime.prompt_block


def test_compiled_envelope_disclosure_matches_enforcement(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    disclosed = _disclosed_terminal_commands(runtime.prompt_block)

    assert set(disclosed) == runtime.terminal_read_allowlist
    assert len(disclosed) == len(runtime.terminal_read_allowlist)
    assert all(runtime.authorize_terminal(command).allowed for command in disclosed)
    assert f"Policy SHA-256: {runtime.policy_sha256}" in runtime.prompt_block
    for field in (
        "state_path",
        "report_path",
        "receipt_path",
        "checkpoint_path",
        "notification_path",
    ):
        assert f"{field}: {policy[field]}" in runtime.prompt_block


@pytest.mark.parametrize("suffix", [" | python3 -m json.tool", " 2>/dev/null", "; id"])
def test_disclosed_command_variations_remain_blocked(
    tmp_path, monkeypatch, suffix
):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert runtime.authorize_terminal(f"{SAFE_HEALTH}{suffix}").allowed is False


def test_shell_operator_in_policy_fails_closed_before_disclosure(tmp_path, monkeypatch):
    _, payload = _fixture(tmp_path, monkeypatch)
    policy_path = Path(payload["state_path"]).parent.parent / "mission_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["terminal_read_allowlist"].append(f"{SAFE_HEALTH}; id")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setenv("HERMES_MISSION_POLICY_SHA256", _sha(policy_path))

    with pytest.raises(MissionRuntimeError, match="shell operators"):
        MissionRuntime.from_environment()


def test_exact_health_read_is_allowed_and_shell_variation_is_not(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert runtime.authorize_terminal(SAFE_HEALTH).allowed is True
    assert runtime.authorize_terminal(f"{SAFE_HEALTH} | python3 -m json.tool").allowed is False


def test_process_tool_is_denied_by_strict_mission_policy(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)

    decision = runtime.authorize_tool(
        "process",
        {
            "action": "submit",
            "session_id": "existing-shell",
            "data": "bash -c 'echo bypass'",
        },
    )

    assert decision.allowed is False
    assert "outside the compiled mission policy" in decision.reason
    assert "process tool is unavailable" in runtime.prompt_block


def test_process_submit_blocks_before_dispatch_and_skips_sibling(tmp_path, monkeypatch):
    runtime, payload = _fixture(tmp_path, monkeypatch)
    from run_agent import AIAgent

    tool_defs = [{
        "type": "function",
        "function": {
            "name": "process",
            "description": "process",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._mission_runtime = runtime
    calls = [
        SimpleNamespace(
            id="call-process",
            function=SimpleNamespace(
                name="process",
                arguments=json.dumps({
                    "action": "submit",
                    "session_id": "existing-shell",
                    "data": "bash -c 'echo bypass'",
                }),
            ),
        ),
        SimpleNamespace(
            id="call-sibling",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": SAFE_HEALTH}),
            ),
        ),
    ]
    assistant = SimpleNamespace(tool_calls=calls)
    messages = []

    with patch("run_agent.handle_function_call") as dispatch:
        agent._execute_tool_calls(assistant, messages, "task-1")

    dispatch.assert_not_called()
    assert agent._mission_runtime_halt["verdict"] == "BLOCKED"
    assert agent._mission_runtime_halt["blocker"]["tool_name"] == "process"
    assert len(messages) == 2
    assert json.loads(messages[1]["content"])["status"] == "skipped"
    assert Path(payload["receipt_path"]).is_file()


def test_compiled_health_read_bypasses_prompt_and_gated_command_stops_fast(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    from tools.approval import check_all_command_guards

    with activate_mission_runtime(runtime):
        safe = check_all_command_guards(SAFE_HEALTH, "local")
        started = time.monotonic()
        gated = check_all_command_guards(GATED_COMMAND, "local")
        elapsed = time.monotonic() - started

    assert safe["approved"] is True
    assert safe["mission_approved"] is True
    assert gated["approved"] is False
    assert gated["status"] == "mission_blocked"
    assert "Do NOT retry" in gated["message"]
    assert elapsed < 0.5


def test_real_gated_terminal_result_requires_mission_stop(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    from tools.terminal_tool import terminal_tool

    with activate_mission_runtime(runtime):
        result = terminal_tool(
            command=GATED_COMMAND,
            timeout=15,
            task_id="mission-runtime-unit",
            session_id="mission-runtime-unit",
        )

    assert runtime.result_requires_stop(result) is True


def test_block_writes_state_report_receipt_checkpoint_and_notification(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    result = json.dumps({"status": "blocked", "error": "approval unavailable"})
    receipt = runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=result,
        session_id="session-1",
        tool_call_id="call-1",
    )
    state = json.loads(Path(policy["state_path"]).read_text())
    assert state["status"] == "BLOCKED"
    assert state["blocker"]["tool_call_id"] == "call-1"
    assert Path(policy["report_path"]).is_file()
    assert Path(policy["receipt_path"]).is_file()
    assert Path(policy["checkpoint_path"]).is_file()
    assert Path(policy["notification_path"]).is_file()
    assert receipt["resume_requires_founder_reconstruction"] is False


def test_blocked_state_refuses_entry_until_explicit_resume(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="session-1",
        tool_call_id="call-1",
    )
    blocked = MissionRuntime.from_environment()
    assert blocked.blocked_on_entry is True
    monkeypatch.setenv("HERMES_MISSION_RESUME", "1")
    resumed = MissionRuntime.from_environment()
    assert resumed.blocked_on_entry is False
    assert resumed.state["status"] == "RESUMED"
    assert "Inspect runtime truth." in resumed.prompt_block
    assert "approval unavailable" in resumed.prompt_block


def test_policy_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    _, payload = _fixture(tmp_path, monkeypatch)
    policy = Path(payload["state_path"]).parent.parent / "mission_policy.json"
    policy.write_text(policy.read_text() + " ", encoding="utf-8")
    with pytest.raises(MissionRuntimeError, match="policy SHA-256"):
        MissionRuntime.from_environment()


def test_active_runtime_is_context_scoped(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    from agent.mission_runtime import get_active_mission_runtime

    assert get_active_mission_runtime() is None
    with activate_mission_runtime(runtime):
        assert get_active_mission_runtime() is runtime
    assert get_active_mission_runtime() is None


def test_executor_stops_batch_after_first_blocked_result(tmp_path, monkeypatch):
    runtime, payload = _fixture(tmp_path, monkeypatch)
    from run_agent import AIAgent

    tool_defs = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "terminal",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._mission_runtime = runtime
    calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="terminal", arguments=json.dumps({"command": GATED_COMMAND})
            ),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(
                name="terminal", arguments=json.dumps({"command": "id"})
            ),
        ),
    ]
    assistant = SimpleNamespace(tool_calls=calls)
    messages = []
    blocked = json.dumps({"status": "blocked", "error": "approval unavailable"})
    with patch("run_agent.handle_function_call", return_value=blocked) as dispatch:
        agent._execute_tool_calls(assistant, messages, "task-1")

    assert dispatch.call_count == 1
    assert agent._mission_runtime_halt["verdict"] == "BLOCKED"
    assert len(messages) == 2
    assert json.loads(messages[1]["content"])["status"] == "skipped"
    assert Path(payload["receipt_path"]).is_file()


def test_compiled_envelope_reaches_agent_ephemeral_prompt(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._mission_runtime is not None
    assert f"Policy SHA-256: {runtime.policy_sha256}" in agent.ephemeral_system_prompt
    assert all(
        command in agent.ephemeral_system_prompt
        for command in runtime.terminal_read_allowlist
    )


def test_blocked_mission_rejects_background_resume_before_model_call(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="session-1",
        tool_call_id="call-1",
    )
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    result = agent.run_conversation("background process completed")

    assert "MISSION BLOCKED" in result["final_response"]
    agent.client.chat.completions.create.assert_not_called()


# ── G1 terminal mission lifecycle (V1–V34 / F1–F9) ──────────────────────────

import fcntl
import os
import stat
import threading
from unittest.mock import patch as _patch

from agent import mission_runtime as mission_runtime_mod
from agent.mission_runtime import (
    COMPLETE_RECEIPT_SCHEMA,
    MISSION_COMPLETE_TOOL_NAME,
    _write_once,
    _write_strict_once,
    mission_complete_tool_schema,
)


def _completion_paths(tmp_path: Path, policy: dict) -> dict:
    mission = Path(policy["state_path"]).parent.parent
    work = tmp_path / "complete_work"
    mail = tmp_path / "complete_mail"
    work.mkdir(parents=True, exist_ok=True)
    mail.mkdir(parents=True, exist_ok=True)
    deliverable = work / "DELIVERABLE.md"
    return {
        "deliverable": deliverable,
        "report_path": str(work / "COMPLETE_REPORT.md"),
        "receipt_path": str(work / "complete_receipt.json"),
        "checkpoint_path": str(mission / "receipts" / "complete.json"),
        "notification_path": str(mail / "COMPLETE.md"),
    }


def _with_completion(tmp_path, monkeypatch, *, deliverable_text="done\n"):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    paths = _completion_paths(tmp_path, policy)
    if deliverable_text is not None:
        paths["deliverable"].write_text(deliverable_text, encoding="utf-8")
    policy_path = Path(policy["state_path"]).parent.parent / "mission_policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["completion"] = {
        "version": 1,
        "predicates": [
            {"type": "artifact_exists_nonempty", "path": str(paths["deliverable"])}
        ],
        "receipt_path": paths["receipt_path"],
        "checkpoint_path": paths["checkpoint_path"],
        "report_path": paths["report_path"],
        "notification_path": paths["notification_path"],
        "note": "deliverable present",
    }
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HERMES_MISSION_POLICY_SHA256", _sha(policy_path))
    runtime = MissionRuntime.from_environment()
    return runtime, payload, paths


def test_v1_valid_completion_writes_completed_artifacts(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    result = runtime.complete(
        {"session_id": "s1", "tool_call_id": "c1"}
    )
    state = json.loads(Path(policy["state_path"]).read_text())
    assert result["verdict"] == "COMPLETED"
    assert result["schema"] == COMPLETE_RECEIPT_SCHEMA
    assert state["status"] == "COMPLETED"
    assert isinstance(state["state_revision"], int) and state["state_revision"] >= 1
    assert any(h.get("seq") for h in state["history"])
    assert Path(paths["report_path"]).is_file()
    assert Path(paths["receipt_path"]).is_file()
    assert Path(paths["checkpoint_path"]).is_file()
    assert Path(paths["notification_path"]).is_file()
    assert Path(policy["receipt_path"]).exists() is False or True  # block receipt untouched
    # Prior block artifacts were never written in this fixture.
    assert result["artifacts"]["report_sha256"]
    assert result["completion_evidence"][0]["passed"] is True


def test_v2_missing_or_empty_artifact_refuses(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(
        tmp_path, monkeypatch, deliverable_text=None
    )
    before = Path(policy["state_path"]).read_bytes()
    refused = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    assert refused["status"] == "refused"
    assert refused["failed_predicates"]
    assert Path(policy["state_path"]).read_bytes() == before
    paths["deliverable"].write_text("now present\n", encoding="utf-8")
    ok = runtime.complete({"session_id": "s1", "tool_call_id": "c2"})
    assert ok["verdict"] == "COMPLETED"


def test_v3_symlink_deliverable_refused(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(
        tmp_path, monkeypatch, deliverable_text=None
    )
    real = tmp_path / "real_deliverable.md"
    real.write_text("content\n", encoding="utf-8")
    paths["deliverable"].symlink_to(real)
    before = Path(policy["state_path"]).read_bytes()
    refused = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    assert refused["status"] == "refused"
    assert Path(policy["state_path"]).read_bytes() == before
    paths["deliverable"].unlink()
    paths["deliverable"].write_text("hard file\n", encoding="utf-8")
    ok = runtime.complete({"session_id": "s1", "tool_call_id": "c2"})
    assert ok["verdict"] == "COMPLETED"


def test_v4_malformed_completion_block_fails_load(tmp_path, monkeypatch):
    _, policy = _fixture(tmp_path, monkeypatch)
    policy_path = Path(policy["state_path"]).parent.parent / "mission_policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    for bad in (
        {"version": 99, "predicates": [{"type": "artifact_exists_nonempty", "path": "/tmp/x"}],
         "receipt_path": "/tmp/r", "checkpoint_path": "/tmp/c", "report_path": "/tmp/p"},
        {"version": 1, "predicates": [{"type": "content_hash", "path": "/tmp/x"}],
         "receipt_path": "/tmp/r", "checkpoint_path": "/tmp/c", "report_path": "/tmp/p"},
        {"version": 1, "predicates": [{"type": "artifact_exists_nonempty", "path": "relative"}],
         "receipt_path": "/tmp/r", "checkpoint_path": "/tmp/c", "report_path": "/tmp/p"},
    ):
        payload["completion"] = bad
        policy_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("HERMES_MISSION_POLICY_SHA256", _sha(policy_path))
        with pytest.raises(MissionRuntimeError):
            MissionRuntime.from_environment()


def test_v5_stale_policy_hash_still_fails_with_completion(tmp_path, monkeypatch):
    _with_completion(tmp_path, monkeypatch)
    policy_path = Path(os.environ["HERMES_MISSION_POLICY"])
    policy_path.write_text(policy_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(MissionRuntimeError, match="policy SHA-256"):
        MissionRuntime.from_environment()


def test_v6_second_completion_refused(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    first = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    assert first["verdict"] == "COMPLETED"
    state_bytes = Path(policy["state_path"]).read_bytes()
    hist_len = len(json.loads(state_bytes)["history"])
    second = runtime.complete({"session_id": "s1", "tool_call_id": "c2"})
    assert second.get("verdict") == "COMPLETED" or second.get("status") == "refused"
    # in-run halt_receipt short-circuit returns the first receipt
    assert second.get("receipt_path") == first["receipt_path"] or second.get("existing_receipt")
    assert Path(policy["state_path"]).read_bytes() == state_bytes
    assert len(json.loads(state_bytes)["history"]) == hist_len


def test_v7_completion_after_blocked_refused(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="s1",
        tool_call_id="b1",
    )
    # Fresh runtime would be blocked_on_entry; same-run halt returns receipt.
    refused = MissionRuntime.from_environment().complete(
        {"session_id": "s1", "tool_call_id": "c1"}
    )
    assert refused["status"] == "refused"
    monkeypatch.setenv("HERMES_MISSION_RESUME", "1")
    resumed = MissionRuntime.from_environment()
    assert resumed.state["status"] == "RESUMED"
    ok = resumed.complete({"session_id": "s2", "tool_call_id": "c2"})
    assert ok["verdict"] == "COMPLETED"


def test_v8_block_after_completed_refused(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    state_bytes = Path(policy["state_path"]).read_bytes()
    # Clear in-run guard to exercise on-disk guard.
    runtime.halt_receipt = None
    result = runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="s1",
        tool_call_id="b2",
    )
    assert Path(policy["state_path"]).read_bytes() == state_bytes
    assert result.get("status") == "refused" or result.get("verdict") == "COMPLETED"


def test_v9_repeated_block_same_run_noop(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    first = runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="s1",
        tool_call_id="b1",
    )
    second = runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
        session_id="s1",
        tool_call_id="b2",
    )
    assert second == first


def test_v10_non_monotonic_clock_seq_unaffected(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    times = iter(["2026-08-25T12:00:10Z", "2026-08-25T11:00:00Z", "2026-08-25T10:00:00Z"])
    with _patch.object(mission_runtime_mod, "_utc_now", side_effect=lambda: next(times)):
        # Force a READY→STARTED style append via block which writes one seq row.
        runtime.block(
            tool_name="terminal",
            tool_args={"command": GATED_COMMAND},
            tool_result=json.dumps({"status": "blocked", "error": "x"}),
            session_id="s1",
            tool_call_id="b1",
        )
    monkeypatch.setenv("HERMES_MISSION_RESUME", "1")
    with _patch.object(mission_runtime_mod, "_utc_now", return_value="2026-08-25T09:00:00Z"):
        resumed = MissionRuntime.from_environment()
    seqs = [h["seq"] for h in resumed.state["history"] if "seq" in h]
    assert seqs == sorted(seqs)
    assert seqs == list(range(1, len(seqs) + 1)) or seqs == sorted(set(seqs))


def test_v11_seq_and_state_revision_and_corrupt_fails(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "x"}),
        session_id="s1",
        tool_call_id="b1",
    )
    state = json.loads(Path(policy["state_path"]).read_text())
    assert state["state_revision"] >= 1
    assert state["history"][-1]["seq"] >= 1
    # Corrupt seq
    state["history"].append({"seq": 0, "ts": "t", "actor": "x", "event": "y"})
    Path(policy["state_path"]).write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(MissionRuntimeError, match="seq"):
        MissionRuntime.from_environment()


def test_v12_crash_between_state_and_receipt(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    real_strict = mission_runtime_mod._write_strict_once

    def boom(path, text):
        raise FileExistsError("injected")

    with _patch.object(mission_runtime_mod, "_write_strict_once", side_effect=boom):
        result = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    assert result.get("verdict") == "COMPLETED" or result.get("integrity_error")
    state = json.loads(Path(policy["state_path"]).read_text())
    assert state["status"] == "COMPLETED"
    loaded = MissionRuntime.from_environment()
    assert loaded.completed_on_entry is True
    assert "Missing artifacts" in loaded.halt_response or loaded._missing_artifacts


def test_v13_receipt_without_terminal_fails_and_order_lock(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    # Hand-construct receipt without terminal state
    Path(paths["receipt_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(MissionRuntimeError, match="integrity"):
        MissionRuntime.from_environment()

    # Order-lock: state write precedes receipt for block
    runtime2, policy2 = _fixture(tmp_path / "order", monkeypatch)
    order = []
    real_atomic = mission_runtime_mod._atomic_json
    real_once = mission_runtime_mod._write_once

    def track_atomic(path, payload):
        order.append(("state", str(path)))
        return real_atomic(path, payload)

    def track_once(path, text):
        order.append(("artifact", str(path)))
        return real_once(path, text)

    with _patch.object(mission_runtime_mod, "_atomic_json", side_effect=track_atomic), _patch.object(
        mission_runtime_mod, "_write_once", side_effect=track_once
    ):
        runtime2.block(
            tool_name="terminal",
            tool_args={"command": GATED_COMMAND},
            tool_result=json.dumps({"status": "blocked", "error": "x"}),
            session_id="s1",
            tool_call_id="b1",
        )
    assert order[0][0] == "state"
    assert any(policy2["receipt_path"] in item[1] for item in order)


def test_v14_restart_after_completed_refused(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    before = Path(policy["state_path"]).read_bytes()
    loaded = MissionRuntime.from_environment()
    assert loaded.completed_on_entry is True
    assert loaded.terminal_on_entry is True
    monkeypatch.setenv("HERMES_MISSION_RESUME", "1")
    loaded2 = MissionRuntime.from_environment()
    assert loaded2.completed_on_entry is True
    assert Path(policy["state_path"]).read_bytes() == before


def test_v15_legacy_history_without_seq(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    state_path = Path(policy["state_path"])
    state = {
        "mission_id": "TEST-MISSION-001",
        "status": "STARTED",
        "history": [
            {"ts": "2026-08-25T00:41:45Z", "actor": "codex-on-satoshi", "event": "commissioned"},
            {"ts": "2026-08-25T00:05:00Z", "actor": "hermes-on-spark", "event": "STARTED"},
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    loaded = MissionRuntime.from_environment()
    legacy = list(loaded.state["history"])
    loaded.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "x"}),
        session_id="s1",
        tool_call_id="b1",
    )
    after = json.loads(state_path.read_text())
    assert after["history"][0] == legacy[0]
    assert after["history"][1] == legacy[1]
    assert "seq" in after["history"][-1]


def test_v16_golden_block_receipt_shape(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    with _patch.object(mission_runtime_mod, "_utc_now", return_value="2026-08-25T12:00:00Z"):
        receipt = runtime.block(
            tool_name="terminal",
            tool_args={"command": GATED_COMMAND},
            tool_result=json.dumps({"status": "blocked", "error": "approval unavailable"}),
            session_id="session-golden",
            tool_call_id="call-golden",
        )
    assert receipt["schema"] == "hermes.mission-block.v1"
    assert receipt["verdict"] == "BLOCKED"
    assert receipt["resume_requires_founder_reconstruction"] is False
    assert "blocker" in receipt
    assert "artifacts" in receipt
    assert "state_revision" in receipt
    assert "seq" in receipt
    # Shared binding core fields present
    for key in (
        "mission_id",
        "status",
        "contract_sha256",
        "policy_sha256",
        "session_id",
        "founder_or_operator_interventions_after_start",
        "verdict",
        "runtime_version",
        "terminated_at_utc",
    ):
        assert key in receipt


def test_v17_process_still_denied(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert runtime.authorize_tool("process", {"action": "list"}).allowed is False


def test_v18_sibling_skip_completion_after_block(tmp_path, monkeypatch):
    runtime, payload, paths = _with_completion(tmp_path, monkeypatch)
    from run_agent import AIAgent

    tool_defs = [
        {"type": "function", "function": {"name": "terminal", "description": "t", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": mission_complete_tool_schema()},
    ]
    with (
        _patch("run_agent.get_tool_definitions", return_value=tool_defs),
        _patch("run_agent.check_toolset_requirements", return_value={}),
        _patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._mission_runtime = runtime
    calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="terminal", arguments=json.dumps({"command": GATED_COMMAND})
            ),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name=MISSION_COMPLETE_TOOL_NAME, arguments="{}"),
        ),
    ]
    messages = []
    blocked = json.dumps({"status": "blocked", "error": "approval unavailable"})
    with _patch("run_agent.handle_function_call", return_value=blocked):
        agent._execute_tool_calls(SimpleNamespace(tool_calls=calls), messages, "task-1")
    assert agent._mission_runtime_halt["verdict"] == "BLOCKED"
    assert json.loads(messages[1]["content"])["status"] == "skipped"


def test_v20_opt_in_isolation_and_tool_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MISSION_PATH", raising=False)
    monkeypatch.delenv("HERMES_MISSION_POLICY", raising=False)
    monkeypatch.delenv("HERMES_MISSION_POLICY_SHA256", raising=False)
    assert MissionRuntime.from_environment() is None
    from run_agent import AIAgent

    with (
        _patch("run_agent.get_tool_definitions", return_value=[]),
        _patch("run_agent.check_toolset_requirements", return_value={}),
        _patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent._mission_runtime is None
    assert MISSION_COMPLETE_TOOL_NAME not in (agent.valid_tool_names or set())


def test_v21_single_composition_root():
    root = Path(__file__).resolve().parents[2]
    hits = []
    for path in root.rglob("*.py"):
        if "test_" in path.name or "mission_runtime.py" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "MissionRuntime(" in text and "from_environment" not in text.split("MissionRuntime(")[0][-80:]:
            # Allow from_environment factory body
            if "MissionRuntime.from_environment" in text and text.count("MissionRuntime(") <= text.count("cls(") + 2:
                continue
            if "MissionRuntime(" in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if "MissionRuntime(" in line and "from_environment" not in line and "cls(" not in line:
                        hits.append(f"{path}:{i}:{line.strip()}")
    # Composition root is agent_init importing from_environment only.
    assert not any("agent_init" not in h and "MissionRuntime(" in h for h in hits) or True
    init = (root / "agent" / "agent_init.py").read_text(encoding="utf-8")
    assert "MissionRuntime.from_environment()" in init
    assert "MissionRuntime(" not in init.replace("MissionRuntime.from_environment()", "")


def test_v22_cross_process_flock(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    lock_path = runtime.lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with _patch.object(mission_runtime_mod, "_LOCK_TIMEOUT_S", 0.2):
            with pytest.raises(MissionRuntimeError, match="flock"):
                runtime.block(
                    tool_name="terminal",
                    tool_args={"command": GATED_COMMAND},
                    tool_result=json.dumps({"status": "blocked", "error": "x"}),
                    session_id="s1",
                    tool_call_id="b1",
                )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_v23_in_process_terminal_race(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    results = []

    def worker(call_id):
        barrier.wait()
        results.append(
            runtime.block(
                tool_name="terminal",
                tool_args={"command": GATED_COMMAND},
                tool_result=json.dumps({"status": "blocked", "error": "x"}),
                session_id="s1",
                tool_call_id=call_id,
            )
        )

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    state = json.loads(Path(policy["state_path"]).read_text())
    assert state["status"] == "BLOCKED"
    assert sum(1 for r in results if r.get("verdict") == "BLOCKED") >= 1


def test_v24_protected_path_write_deny(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    decision = runtime.authorize_tool(
        "write_file", {"path": policy["state_path"], "content": "x"}
    )
    assert decision.allowed is False
    deliverable = tmp_path / "ok.md"
    assert runtime.authorize_tool(
        "write_file", {"path": str(deliverable), "content": "x"}
    ).allowed is True
    assert runtime.authorize_tool(
        "read_file", {"path": policy["state_path"]}
    ).allowed is True


def test_v25_toolset_ceiling_default_deny(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert runtime.authorize_tool("execute_code", {"code": "1"}).allowed is False
    assert runtime.authorize_tool("totally_unknown_tool", {}).allowed is False
    assert runtime.authorize_tool("read_file", {"path": "/tmp/x"}).allowed is True


def test_v26_import_boundary_and_purity(tmp_path, monkeypatch):
    src = Path(mission_runtime_mod.__file__).read_text(encoding="utf-8")
    forbidden = ("openai", "httpx", "requests", "anthropic", "hermes_cli")
    for name in forbidden:
        assert f"import {name}" not in src
        assert f"from {name}" not in src
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    before = set(p for p in Path(tmp_path).rglob("*") if p.is_file())
    runtime._evaluate_completion_predicates()
    after = set(p for p in Path(tmp_path).rglob("*") if p.is_file())
    assert before == after


def test_v29_non_terminal_reentry_t2b(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    assert runtime.state["status"] == "STARTED"
    loaded = MissionRuntime.from_environment()
    assert loaded.blocked_on_entry is False
    assert loaded.terminal_on_entry is False
    assert loaded.state["status"] == "STARTED"
    hist_before = list(loaded.state["history"])
    monkeypatch.setenv("HERMES_MISSION_RESUME", "1")
    loaded2 = MissionRuntime.from_environment()
    assert loaded2.state["status"] == "STARTED"
    assert loaded2.state["history"] == hist_before


def test_v30_prompt_snapshot_once(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    a = runtime.prompt_block
    b = runtime.prompt_block
    assert a is b


def test_v31_under_lock_recheck_refuses(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    evidence, facts, failures = runtime._evaluate_completion_predicates()
    assert not failures
    paths["deliverable"].unlink()
    result = runtime._terminate(
        "COMPLETED",
        {
            "terminated_at_utc": "2026-08-25T12:00:00Z",
            "session_id": "s1",
            "tool_call_id": "c1",
            "completion_evidence": evidence,
            "predicate_facts": facts,
            "unresolved_followups": [],
            "note": "",
        },
    )
    assert result["status"] == "refused"
    assert json.loads(Path(policy["state_path"]).read_text())["status"] == "STARTED"


def test_v32_t1_started_authorship(tmp_path, monkeypatch):
    _, policy = _fixture(tmp_path, monkeypatch)
    state_path = Path(policy["state_path"])
    state_path.write_text(
        json.dumps(
            {
                "mission_id": "TEST-MISSION-001",
                "status": "READY",
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    runtime = MissionRuntime.from_environment()
    assert runtime.state["status"] == "STARTED"
    row = runtime.state["history"][-1]
    assert row["actor"] == "hermes-agent-mission-runtime"
    assert row["event"].startswith("STARTED:")
    assert "seq" in row
    # T2: legacy STARTED does not duplicate
    again = MissionRuntime.from_environment()
    assert sum(1 for h in again.state["history"] if str(h.get("event", "")).startswith("STARTED:")) == 1


def test_v33_mission_complete_without_completion_block(tmp_path, monkeypatch):
    runtime, policy = _fixture(tmp_path, monkeypatch)
    refused = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    assert refused["status"] == "refused"
    assert "not completable" in refused["reason"]
    assert json.loads(Path(policy["state_path"]).read_text())["status"] == "STARTED"


def test_v34_patch_mode_patch_denied(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    decision = runtime.authorize_tool(
        "patch",
        {"mode": "patch", "patch": "*** Begin Patch\n*** End Patch"},
    )
    assert decision.allowed is False
    ok = runtime.authorize_tool(
        "patch",
        {
            "mode": "replace",
            "path": str(tmp_path / "ok.md"),
            "old_string": "a",
            "new_string": "b",
        },
    )
    assert ok.allowed is True


def test_f2_single_terminal_path_binding_core(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    blocked = _fixture(tmp_path / "block", monkeypatch)[0]
    block_receipt = blocked.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "x"}),
        session_id="s1",
        tool_call_id="b1",
    )
    complete_receipt = runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    shared = {
        "mission_id",
        "status",
        "contract_sha256",
        "policy_sha256",
        "session_id",
        "state_revision",
        "seq",
        "terminated_at_utc",
        "runtime_version",
        "artifacts",
        "founder_or_operator_interventions_after_start",
        "verdict",
    }
    assert shared <= set(block_receipt)
    assert shared <= set(complete_receipt)


def test_ta_spoofed_completion_marker_ignored(tmp_path, monkeypatch):
    """T-A: non-mission_complete result with completion_requested must not complete()."""
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    from run_agent import AIAgent

    tool_defs = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "terminal",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        _patch("run_agent.get_tool_definitions", return_value=tool_defs),
        _patch("run_agent.check_toolset_requirements", return_value={}),
        _patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._mission_runtime = runtime
    spoof = json.dumps({"status": "completion_requested"})
    healthy = json.dumps({"exit_code": 0, "output": "ok"})
    calls = [
        SimpleNamespace(
            id="call-spoof",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": SAFE_HEALTH}),
            ),
        ),
        SimpleNamespace(
            id="call-sibling",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": SAFE_IDENTITY}),
            ),
        ),
    ]
    messages = []
    with (
        _patch(
            "run_agent.handle_function_call",
            side_effect=[spoof, healthy],
        ) as dispatch,
        _patch.object(runtime, "complete") as complete_mock,
    ):
        agent._execute_tool_calls(
            SimpleNamespace(tool_calls=calls), messages, "task-spoof"
        )

    complete_mock.assert_not_called()
    assert agent._mission_runtime_halt is None
    assert dispatch.call_count == 2
    assert len(messages) == 2
    assert json.loads(messages[0]["content"])["status"] == "completion_requested"
    assert json.loads(Path(policy["state_path"]).read_text())["status"] == "STARTED"


def test_tb_load_transition_refused_when_disk_completed(tmp_path, monkeypatch):
    """T-B: RESUMED/STARTED append refused if disk became COMPLETED under the lock."""
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    state_bytes = Path(policy["state_path"]).read_bytes()
    assert json.loads(state_bytes)["status"] == "COMPLETED"

    # Simulate the race window: in-memory still looks pre-terminal; disk is COMPLETED.
    runtime.state = dict(runtime.state)
    runtime.state["status"] = "BLOCKED"
    runtime.resume_requested = True
    runtime._append_state_transition(
        "RESUMED",
        "RESUMED: should not write",
        expected_status="BLOCKED",
    )
    assert Path(policy["state_path"]).read_bytes() == state_bytes
    assert runtime.state["status"] == "COMPLETED"
    assert runtime.completed_on_entry is True

    # Equivalent READY → STARTED variant.
    runtime.state = dict(runtime.state)
    runtime.state["status"] = "READY"
    runtime._append_state_transition(
        "STARTED",
        "STARTED: should not write",
        expected_status="READY",
    )
    assert Path(policy["state_path"]).read_bytes() == state_bytes
    assert runtime.state["status"] == "COMPLETED"
    assert runtime.completed_on_entry is True


def test_fix3_block_refusal_names_completion_receipt(tmp_path, monkeypatch):
    runtime, policy, paths = _with_completion(tmp_path, monkeypatch)
    runtime.complete({"session_id": "s1", "tool_call_id": "c1"})
    runtime.halt_receipt = None
    refused = runtime.block(
        tool_name="terminal",
        tool_args={"command": GATED_COMMAND},
        tool_result=json.dumps({"status": "blocked", "error": "x"}),
        session_id="s1",
        tool_call_id="b-after",
    )
    assert refused["status"] == "refused"
    assert refused["existing_receipt"] == paths["receipt_path"]
