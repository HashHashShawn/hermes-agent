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


SAFE_HEALTH = "curl -s http://127.0.0.1:8000/v1/models"
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
        "terminal_read_allowlist": [SAFE_HEALTH],
    }
    policy.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HERMES_MISSION_PATH", str(contract))
    monkeypatch.setenv("HERMES_MISSION_POLICY", str(policy))
    monkeypatch.setenv("HERMES_MISSION_POLICY_SHA256", _sha(policy))
    return MissionRuntime.from_environment(), payload


def test_contract_and_state_are_loaded_into_working_context(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert "Inspect runtime truth." in runtime.prompt_block
    assert "Write REPORT.md and receipt.json." in runtime.prompt_block
    assert "Run the health read." in runtime.prompt_block


def test_exact_health_read_is_allowed_and_shell_variation_is_not(tmp_path, monkeypatch):
    runtime, _ = _fixture(tmp_path, monkeypatch)
    assert runtime.authorize_terminal(SAFE_HEALTH).allowed is True
    assert runtime.authorize_terminal(f"{SAFE_HEALTH} | python3 -m json.tool").allowed is False


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
