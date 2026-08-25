"""Strict, opt-in runtime for hash-bound autonomous missions.

The normal Hermes chat runtime remains unchanged unless both
``HERMES_MISSION_PATH`` and ``HERMES_MISSION_POLICY`` are set.  A mission
runtime loads the full durable contract into working context, enforces the
compiled terminal allowlist, and turns any blocked tool result into a durable
mission stop.  It never asks the model to interpret whether a denial is final.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


MISSION_SCHEMA = "hermes.mission-runtime.v1"
BLOCK_RECEIPT_SCHEMA = "hermes.mission-block.v1"
_MISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHELL_OPERATOR_RE = re.compile(r"(?:\n|&&|\|\||[;&|<>`]|\$\()")
_ACTIVE_RUNTIME: contextvars.ContextVar["MissionRuntime | None"] = (
    contextvars.ContextVar("hermes_active_mission_runtime", default=None)
)


class MissionRuntimeError(RuntimeError):
    """The mission envelope is missing, invalid, or cannot be persisted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _write_once(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    suffix = 1
    while True:
        try:
            with candidate.open("x", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.stem}.v{suffix}{path.suffix}")
            suffix += 1


def _resolve_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MissionRuntimeError(f"mission policy field {field!r} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise MissionRuntimeError(f"mission policy field {field!r} must be absolute")
    return path.resolve(strict=False)


def _blocked_payload(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "").lower()
    if status in {"blocked", "pending_approval", "approval_required"}:
        return payload
    if payload.get("approval_pending") is True:
        return payload
    return None


@dataclass(frozen=True)
class MissionToolDecision:
    allowed: bool
    reason: str


class MissionRuntime:
    """One loaded, hash-bound mission and its compiled execution envelope."""

    def __init__(
        self,
        *,
        contract_path: Path,
        policy_path: Path,
        contract_text: str,
        policy: dict[str, Any],
        state: dict[str, Any],
        resume_requested: bool,
    ) -> None:
        self.contract_path = contract_path
        self.policy_path = policy_path
        self.contract_text = contract_text
        self.policy = policy
        self.state = state
        self.resume_requested = resume_requested
        self.mission_id = policy["mission_id"]
        self.contract_sha256 = policy["contract_sha256"]
        self.policy_sha256 = _sha256_file(policy_path)
        self.state_path = _resolve_path(policy["state_path"], field="state_path")
        self.report_path = _resolve_path(policy["report_path"], field="report_path")
        self.receipt_path = _resolve_path(policy["receipt_path"], field="receipt_path")
        self.checkpoint_path = _resolve_path(
            policy["checkpoint_path"], field="checkpoint_path"
        )
        notification = policy.get("notification_path")
        self.notification_path = (
            _resolve_path(notification, field="notification_path")
            if notification
            else None
        )
        self.terminal_read_allowlist = frozenset(policy["terminal_read_allowlist"])
        self.halt_receipt: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls) -> "MissionRuntime | None":
        contract_raw = os.getenv("HERMES_MISSION_PATH", "").strip()
        policy_raw = os.getenv("HERMES_MISSION_POLICY", "").strip()
        if not contract_raw and not policy_raw:
            return None
        if not contract_raw or not policy_raw:
            raise MissionRuntimeError(
                "HERMES_MISSION_PATH and HERMES_MISSION_POLICY must be set together"
            )

        contract_path = Path(contract_raw).expanduser().resolve(strict=True)
        policy_path = Path(policy_raw).expanduser().resolve(strict=True)
        contract_text = contract_path.read_text(encoding="utf-8")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(policy, dict) or policy.get("schema") != MISSION_SCHEMA:
            raise MissionRuntimeError(f"mission policy schema must be {MISSION_SCHEMA}")

        mission_id = policy.get("mission_id")
        if not isinstance(mission_id, str) or not _MISSION_ID_RE.fullmatch(mission_id):
            raise MissionRuntimeError("mission_id is missing or invalid")

        expected_contract = str(policy.get("contract_sha256") or "")
        if _sha256_file(contract_path) != expected_contract:
            raise MissionRuntimeError("MISSION.md SHA-256 does not match mission policy")
        expected_policy = os.getenv("HERMES_MISSION_POLICY_SHA256", "").strip()
        if not expected_policy:
            raise MissionRuntimeError("HERMES_MISSION_POLICY_SHA256 is required")
        if _sha256_file(policy_path) != expected_policy:
            raise MissionRuntimeError("mission policy SHA-256 does not match environment binding")

        state_path = _resolve_path(policy.get("state_path"), field="state_path")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("mission_id") != mission_id:
            raise MissionRuntimeError("mission state does not match mission_id")

        allowlist = policy.get("terminal_read_allowlist")
        if not isinstance(allowlist, list) or not allowlist:
            raise MissionRuntimeError("terminal_read_allowlist must be a non-empty list")
        for command in allowlist:
            if not isinstance(command, str) or not command.strip():
                raise MissionRuntimeError("terminal allowlist entries must be non-empty strings")
            if _SHELL_OPERATOR_RE.search(command):
                raise MissionRuntimeError(
                    "terminal allowlist entries cannot contain shell operators"
                )

        runtime = cls(
            contract_path=contract_path,
            policy_path=policy_path,
            contract_text=contract_text,
            policy=policy,
            state=state,
            resume_requested=os.getenv("HERMES_MISSION_RESUME", "").strip() == "1",
        )
        if runtime.resume_requested and runtime.state.get("status") == "BLOCKED":
            runtime._append_state_transition(
                "RESUMED",
                "Runtime loaded the durable blocker, contract, and next legal action; no founder restatement was supplied.",
            )
        return runtime

    @property
    def prompt_block(self) -> str:
        terminal_commands = "\n".join(
            f"--- EXACT TERMINAL COMMAND {index} ---\n{command}\n"
            f"--- END EXACT TERMINAL COMMAND {index} ---"
            for index, command in enumerate(
                sorted(self.terminal_read_allowlist), start=1
            )
        )
        notification_path = (
            str(self.notification_path)
            if self.notification_path is not None
            else "UNCONFIGURED"
        )
        return (
            "STRICT HASH-BOUND MISSION RUNTIME\n"
            f"Mission: {self.mission_id}\n"
            f"Contract SHA-256: {self.contract_sha256}\n"
            "The complete durable contract and current mission state follow. "
            "Consume them directly. Never ask the founder to restate objectives, "
            "deliverables, stop rules, or the next legal action. A blocked tool "
            "result is terminal for this run. Do not retry, rephrase, substitute "
            "another command, or shop for permission.\n\n"
            "--- COMPILED EXECUTION ENVELOPE ---\n"
            f"Policy SHA-256: {self.policy_sha256}\n"
            "A terminal command is admitted only when, after leading and trailing "
            "whitespace is stripped, it byte-matches one complete command line "
            "listed below. Copy an admitted command exactly. Chaining, redirection, "
            "alternate paths, network access, or any other terminal command will "
            "durably BLOCK the mission. After a blocked result, do not retry, "
            "rephrase, or substitute another command or tool path. Use the file "
            "tool for permitted reads that are not listed terminal commands.\n"
            f"{terminal_commands}\n"
            "--- RESOLVED MISSION ARTIFACT PATHS ---\n"
            f"state_path: {self.state_path}\n"
            f"report_path: {self.report_path}\n"
            f"receipt_path: {self.receipt_path}\n"
            f"checkpoint_path: {self.checkpoint_path}\n"
            f"notification_path: {notification_path}\n"
            "--- END COMPILED EXECUTION ENVELOPE ---\n\n"
            "--- MISSION.md ---\n"
            f"{self.contract_text.rstrip()}\n"
            "--- CURRENT MISSION STATE ---\n"
            f"{json.dumps(self.state, indent=2, sort_keys=True)}"
        )

    @property
    def blocked_on_entry(self) -> bool:
        return self.state.get("status") == "BLOCKED" and not self.resume_requested

    def authorize_terminal(self, command: str) -> MissionToolDecision:
        normalized = (command or "").strip()
        if normalized in self.terminal_read_allowlist:
            return MissionToolDecision(True, "exact compiled read-only command")
        return MissionToolDecision(
            False,
            "terminal command is outside the exact compiled mission allowlist",
        )

    @staticmethod
    def result_requires_stop(result: Any) -> bool:
        return _blocked_payload(result) is not None

    def _append_state_transition(self, status: str, event: str) -> None:
        current = json.loads(self.state_path.read_text(encoding="utf-8"))
        history = list(current.get("history") or [])
        history.append(
            {
                "ts": _utc_now(),
                "actor": "hermes-agent-mission-runtime",
                "event": event,
            }
        )
        current["status"] = status
        current["history"] = history
        self.state = current
        _atomic_json(self.state_path, current)

    def block(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        session_id: str,
        tool_call_id: str,
    ) -> dict[str, Any]:
        if self.halt_receipt is not None:
            return self.halt_receipt

        payload = _blocked_payload(tool_result) or {
            "status": "blocked",
            "error": str(tool_result),
        }
        blocked_at = _utc_now()
        safe_args = dict(tool_args or {})
        command = safe_args.get("command")
        command_sha256 = (
            _sha256_bytes(str(command).encode("utf-8")) if command is not None else None
        )
        blocker = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "status": payload.get("status", "blocked"),
            "error": payload.get("error") or payload.get("message") or "tool blocked",
            "command_sha256": command_sha256,
        }
        next_action = (
            "Resolve the compiled mission permission or tool-policy mismatch, then "
            "resume from this durable BLOCKED checkpoint. Do not restate the mission."
        )

        current = json.loads(self.state_path.read_text(encoding="utf-8"))
        history = list(current.get("history") or [])
        history.append(
            {
                "ts": blocked_at,
                "actor": "hermes-agent-mission-runtime",
                "event": f"BLOCKED: {blocker['error']}",
            }
        )
        current.update(
            {
                "status": "BLOCKED",
                "current_milestone": current.get("current_milestone"),
                "next_legal_action": next_action,
                "blocker": blocker,
                "history": history,
            }
        )
        _atomic_json(self.state_path, current)
        self.state = current

        report = (
            f"# Mission BLOCKED — {self.mission_id}\n\n"
            f"- Contract SHA-256: `{self.contract_sha256}`\n"
            f"- Blocked at UTC: `{blocked_at}`\n"
            f"- Tool: `{tool_name}`\n"
            f"- Tool call: `{tool_call_id}`\n"
            f"- Blocker: {blocker['error']}\n"
            "- Alternate-command retry: **not attempted**\n"
            f"- Next legal action: {next_action}\n"
        )
        report_path = _write_once(self.report_path, report)

        checkpoint = {
            "schema": BLOCK_RECEIPT_SCHEMA,
            "mission_id": self.mission_id,
            "status": "BLOCKED",
            "blocked_at_utc": blocked_at,
            "contract_sha256": self.contract_sha256,
            "policy_sha256": self.policy_sha256,
            "blocker": blocker,
            "next_legal_action": next_action,
            "resume_requires_founder_reconstruction": False,
        }
        checkpoint_path = _write_once(
            self.checkpoint_path,
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        )
        receipt = {
            **checkpoint,
            "session_id": session_id,
            "artifacts": {
                "report": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "state": str(self.state_path),
                "state_sha256": _sha256_file(self.state_path),
            },
            "founder_or_operator_interventions_after_start": [],
            "verdict": "BLOCKED",
        }
        if self.notification_path is not None:
            notification = (
                f"# BLOCK — {self.mission_id}\n\n"
                "**From:** hermes-agent-mission-runtime\n\n"
                f"Mission stopped at `{blocked_at}` without an alternate-command retry.\n\n"
                f"Blocker: {blocker['error']}\n\n"
                f"Receipt target: `{self.receipt_path}`\n\n"
                f"Next legal action: {next_action}\n"
            )
            notification_path = _write_once(self.notification_path, notification)
            receipt["notification_path"] = str(notification_path)
            receipt["notification_sha256"] = _sha256_file(notification_path)

        receipt_path = _write_once(
            self.receipt_path,
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        )
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = _sha256_file(receipt_path)

        self.halt_receipt = receipt
        return receipt

    @property
    def halt_response(self) -> str:
        if self.halt_receipt is None:
            return f"MISSION BLOCKED: {self.mission_id} remains durably blocked."
        blocker = self.halt_receipt["blocker"]["error"]
        return (
            f"MISSION BLOCKED: {self.mission_id}. {blocker} "
            f"Receipt: {self.halt_receipt['receipt_path']}"
        )


def get_active_mission_runtime() -> MissionRuntime | None:
    return _ACTIVE_RUNTIME.get()


@contextlib.contextmanager
def activate_mission_runtime(runtime: MissionRuntime | None) -> Iterator[None]:
    token = _ACTIVE_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME.reset(token)
