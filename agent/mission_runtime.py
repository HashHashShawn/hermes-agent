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
import fcntl
import hashlib
import json
import logging
import os
import re
import stat as _stat_mod
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Iterator


MISSION_SCHEMA = "hermes.mission-runtime.v1"
BLOCK_RECEIPT_SCHEMA = "hermes.mission-block.v1"
COMPLETE_RECEIPT_SCHEMA = "hermes.mission-complete.v1"
MISSION_COMPLETE_TOOL_NAME = "mission_complete"
_MISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHELL_OPERATOR_RE = re.compile(r"(?:\n|&&|\|\||[;&|<>`]|\$\()")
_ADMITTED_TOOLS = frozenset(
    {
        "terminal",
        "read_file",
        "write_file",
        "patch",
        "search_files",
        MISSION_COMPLETE_TOOL_NAME,
    }
)
_LOCK_TIMEOUT_S = 10.0
_LOCK_RETRY_S = 0.05
_RUNTIME_ACTOR = "hermes-agent-mission-runtime"
_ACTIVE_RUNTIME: contextvars.ContextVar["MissionRuntime | None"] = (
    contextvars.ContextVar("hermes_active_mission_runtime", default=None)
)
_logger = logging.getLogger(__name__)


class MissionRuntimeError(RuntimeError):
    """The mission envelope is missing, invalid, or cannot be persisted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _runtime_version() -> str:
    try:
        return _pkg_version("hermes-agent")
    except PackageNotFoundError:
        return "unknown"


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


def _write_strict_once(path: Path, text: str) -> Path:
    """Create ``path`` with O_EXCL. No .vN retry — racer/interference detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            fd = -1  # ownership transferred to fdopen
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def _resolve_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MissionRuntimeError(f"mission policy field {field!r} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise MissionRuntimeError(f"mission policy field {field!r} must be absolute")
    return path.resolve(strict=False)


def _resolve_write_target(path_value: str) -> Path:
    """§K.2 frozen resolution: expanduser → realpath(parent) / final component."""
    path = Path(path_value).expanduser()
    parent = Path(os.path.realpath(str(path.parent)))
    return parent / path.name


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


def _completion_requested_payload(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status") or "") == "completion_requested":
        return payload
    return None


def _validate_history_seq(history: list[Any]) -> None:
    last: int | None = None
    for row in history:
        if not isinstance(row, dict) or "seq" not in row:
            continue
        seq = row["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise MissionRuntimeError("mission history seq values must be integers")
        if last is not None and seq <= last:
            raise MissionRuntimeError(
                "mission history seq values must be strictly increasing"
            )
        last = seq


def _next_seq(history: list[Any]) -> int:
    max_seq = 0
    for row in history:
        if isinstance(row, dict) and isinstance(row.get("seq"), int) and not isinstance(
            row.get("seq"), bool
        ):
            max_seq = max(max_seq, row["seq"])
    return max_seq + 1


def _lstat_facts(path: Path) -> dict[str, Any] | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return {
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)),
        "is_reg": _stat_mod.S_ISREG(st.st_mode),
    }


@dataclass(frozen=True)
class MissionToolDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class _PredicateFact:
    path: Path
    inode: int
    size: int
    mtime_ns: int
    sha256: str


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
        completion_block: dict[str, Any] | None,
        protected_paths: frozenset[Path],
        state_dir: Path,
        lock_path: Path,
        missing_artifacts: tuple[str, ...],
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
        self._completion_block = completion_block
        self._protected_paths = protected_paths
        self._state_dir = state_dir
        self.lock_path = lock_path
        self._missing_artifacts = missing_artifacts
        self._thread_lock = threading.Lock()
        self._prompt_block = self._build_prompt_block()

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
            raise MissionRuntimeError(
                "mission policy SHA-256 does not match environment binding"
            )

        state_path = _resolve_path(policy.get("state_path"), field="state_path")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("mission_id") != mission_id:
            raise MissionRuntimeError("mission state does not match mission_id")

        history = list(state.get("history") or [])
        _validate_history_seq(history)

        allowlist = policy.get("terminal_read_allowlist")
        if not isinstance(allowlist, list) or not allowlist:
            raise MissionRuntimeError("terminal_read_allowlist must be a non-empty list")
        for command in allowlist:
            if not isinstance(command, str) or not command.strip():
                raise MissionRuntimeError(
                    "terminal allowlist entries must be non-empty strings"
                )
            if _SHELL_OPERATOR_RE.search(command):
                raise MissionRuntimeError(
                    "terminal allowlist entries cannot contain shell operators"
                )

        report_path = _resolve_path(policy.get("report_path"), field="report_path")
        receipt_path = _resolve_path(policy.get("receipt_path"), field="receipt_path")
        checkpoint_path = _resolve_path(
            policy.get("checkpoint_path"), field="checkpoint_path"
        )
        notification_raw = policy.get("notification_path")
        notification_path = (
            _resolve_path(notification_raw, field="notification_path")
            if notification_raw
            else None
        )
        lock_path = state_path.with_name(state_path.name + ".lock")
        state_dir = state_path.parent.resolve(strict=False)

        completion_block = cls._validate_completion_block(
            policy.get("completion"),
            block_paths={
                report_path,
                receipt_path,
                checkpoint_path,
                *([notification_path] if notification_path is not None else []),
            },
            state_path=state_path,
            state_dir=state_dir,
            lock_path=lock_path,
            policy_path=policy_path,
            contract_path=contract_path,
        )

        protected: set[Path] = {
            state_path.resolve(strict=False),
            lock_path.resolve(strict=False),
            report_path.resolve(strict=False),
            receipt_path.resolve(strict=False),
            checkpoint_path.resolve(strict=False),
            policy_path.resolve(strict=False),
            contract_path.resolve(strict=False),
        }
        if notification_path is not None:
            protected.add(notification_path.resolve(strict=False))
        if completion_block is not None:
            for key in (
                "receipt_path",
                "checkpoint_path",
                "report_path",
                "notification_path",
            ):
                value = completion_block.get(key)
                if value is not None:
                    protected.add(Path(value).resolve(strict=False))
            # Predicate deliverable paths are validated against the protected
            # set at load time but are NOT themselves write-denied.

        status = state.get("status")
        # Receipt present + non-terminal state ⇒ tamper/corruption (§C.6).
        if status not in {"BLOCKED", "COMPLETED"}:
            if receipt_path.is_file():
                raise MissionRuntimeError(
                    "integrity error: block receipt present without terminal state"
                )
            if completion_block is not None:
                c_receipt = Path(completion_block["receipt_path"])
                if c_receipt.is_file():
                    raise MissionRuntimeError(
                        "integrity error: completion receipt present without terminal state"
                    )

        missing_artifacts: list[str] = []
        if status == "BLOCKED":
            for label, path in (
                ("report", report_path),
                ("checkpoint", checkpoint_path),
                ("receipt", receipt_path),
            ):
                if not path.is_file():
                    missing_artifacts.append(label)
            if notification_path is not None and not notification_path.is_file():
                missing_artifacts.append("notification")
        elif status == "COMPLETED" and completion_block is not None:
            for label, key in (
                ("report", "report_path"),
                ("checkpoint", "checkpoint_path"),
                ("receipt", "receipt_path"),
            ):
                if not Path(completion_block[key]).is_file():
                    missing_artifacts.append(label)
            notif = completion_block.get("notification_path")
            if notif and not Path(notif).is_file():
                missing_artifacts.append("notification")
        if missing_artifacts:
            _logger.error(
                "terminal_incomplete_artifacts mission_id=%s status=%s missing=%s",
                mission_id,
                status,
                ",".join(missing_artifacts),
            )

        if "state_revision" not in state:
            state = dict(state)
            state["state_revision"] = 0

        runtime = cls(
            contract_path=contract_path,
            policy_path=policy_path,
            contract_text=contract_text,
            policy=policy,
            state=state,
            resume_requested=os.getenv("HERMES_MISSION_RESUME", "").strip() == "1",
            completion_block=completion_block,
            protected_paths=frozenset(protected),
            state_dir=state_dir,
            lock_path=lock_path,
            missing_artifacts=tuple(missing_artifacts),
        )

        # Load-time transitions (T1 / T4) under the single lock scope.
        if runtime.state.get("status") == "READY":
            runtime._append_state_transition(
                "STARTED",
                "STARTED: Runtime activated the hash-bound mission envelope.",
            )
            runtime._prompt_block = runtime._build_prompt_block()
        elif runtime.resume_requested and runtime.state.get("status") == "BLOCKED":
            runtime._append_state_transition(
                "RESUMED",
                "RESUMED: Runtime loaded the durable blocker, contract, and next "
                "legal action; no founder restatement was supplied.",
            )
            runtime._prompt_block = runtime._build_prompt_block()
        return runtime

    @staticmethod
    def _validate_completion_block(
        raw: Any,
        *,
        block_paths: set[Path],
        state_path: Path,
        state_dir: Path,
        lock_path: Path,
        policy_path: Path,
        contract_path: Path,
    ) -> dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise MissionRuntimeError("completion block must be an object")
        version = raw.get("version")
        if version != 1:
            raise MissionRuntimeError("completion block version must be 1")
        predicates = raw.get("predicates")
        if not isinstance(predicates, list) or not predicates:
            raise MissionRuntimeError("completion.predicates must be a non-empty list")
        compiled_preds: list[dict[str, str]] = []
        pred_paths: set[Path] = set()
        for index, pred in enumerate(predicates):
            if not isinstance(pred, dict):
                raise MissionRuntimeError(f"completion.predicates[{index}] must be an object")
            ptype = pred.get("type")
            if ptype != "artifact_exists_nonempty":
                raise MissionRuntimeError(
                    f"unsupported completion predicate type: {ptype!r}"
                )
            path = _resolve_path(pred.get("path"), field=f"completion.predicates[{index}].path")
            if path in pred_paths:
                raise MissionRuntimeError("completion predicate paths must be unique")
            pred_paths.add(path)
            compiled_preds.append({"type": "artifact_exists_nonempty", "path": str(path)})

        def _req(field: str) -> Path:
            return _resolve_path(raw.get(field), field=f"completion.{field}")

        c_report = _req("report_path")
        c_receipt = _req("receipt_path")
        c_checkpoint = _req("checkpoint_path")
        c_notification = None
        if raw.get("notification_path") is not None:
            c_notification = _req("notification_path")

        artifact_paths = {c_report, c_receipt, c_checkpoint}
        if c_notification is not None:
            artifact_paths.add(c_notification)
        if len(artifact_paths) != (4 if c_notification is not None else 3):
            raise MissionRuntimeError("completion artifact paths must not collide")

        protected_base = {
            state_path.resolve(strict=False),
            lock_path.resolve(strict=False),
            policy_path.resolve(strict=False),
            contract_path.resolve(strict=False),
            state_dir.resolve(strict=False),
        }
        protected_base |= {p.resolve(strict=False) for p in block_paths}

        def _check_not_protected(path: Path, label: str) -> None:
            resolved = path.resolve(strict=False)
            if resolved in protected_base or resolved == state_dir or _is_under(
                resolved, state_dir
            ):
                raise MissionRuntimeError(
                    f"{label} overlaps the protected runtime path set"
                )
            for blocked in block_paths:
                if resolved == blocked.resolve(strict=False):
                    raise MissionRuntimeError(
                        f"{label} collides with a block artifact path"
                    )

        for pred_path in pred_paths:
            _check_not_protected(pred_path, "completion predicate path")
            if pred_path.resolve(strict=False) in {
                p.resolve(strict=False) for p in artifact_paths
            }:
                raise MissionRuntimeError(
                    "completion predicate paths must not collide with completion artifacts"
                )
        for path, label in (
            (c_report, "completion.report_path"),
            (c_receipt, "completion.receipt_path"),
            (c_checkpoint, "completion.checkpoint_path"),
        ):
            _check_not_protected(path, label)
        if c_notification is not None:
            _check_not_protected(c_notification, "completion.notification_path")

        note = raw.get("note")
        if note is not None and not isinstance(note, str):
            raise MissionRuntimeError("completion.note must be a string when present")

        compiled: dict[str, Any] = {
            "version": 1,
            "predicates": compiled_preds,
            "receipt_path": str(c_receipt),
            "checkpoint_path": str(c_checkpoint),
            "report_path": str(c_report),
            "note": note if isinstance(note, str) else "",
        }
        if c_notification is not None:
            compiled["notification_path"] = str(c_notification)
        return compiled

    def _build_prompt_block(self) -> str:
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
        admitted = ", ".join(sorted(_ADMITTED_TOOLS))
        completion_disclosure = (
            f"Admitted tools in this strict mission: {admitted}. "
            "The patch tool admits only mode=replace (or omitted mode) with a path "
            "argument; mode=patch (V4A multi-target) is unavailable. "
            f"Request mechanical completion only via the {MISSION_COMPLETE_TOOL_NAME} "
            "tool when the compiled completion predicates are satisfied. Model text "
            "alone never completes the mission.\n"
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
            "tool for permitted reads that are not listed terminal commands. "
            "The process tool is unavailable in this strict mission; do not use "
            "process list, poll, log, wait, kill, write, submit, or close. If a "
            "required input moved, use the permitted file search and read tools. "
            "If an optional receipt is absent, record UNKNOWN; do not invent a "
            "terminal filter or discovery command.\n"
            f"{completion_disclosure}"
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
    def prompt_block(self) -> str:
        return self._prompt_block

    @property
    def protected_paths(self) -> frozenset[Path]:
        return self._protected_paths

    @property
    def blocked_on_entry(self) -> bool:
        return self.state.get("status") == "BLOCKED" and not self.resume_requested

    @property
    def completed_on_entry(self) -> bool:
        return self.state.get("status") == "COMPLETED"

    @property
    def terminal_on_entry(self) -> bool:
        return self.blocked_on_entry or self.completed_on_entry

    def authorize_tool(
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> MissionToolDecision:
        """Apply the compiled mission policy before any tool is dispatched.

        G1: DEFAULT-DENY admitted set plus §K.2 protected-path write-deny and
        the V4A ``mode=patch`` freeze.
        """
        args = tool_args or {}
        # Preserve the proven process-denial reason (existing tests pin the text).
        if tool_name == "process":
            return MissionToolDecision(
                False,
                "process tool is outside the compiled mission policy",
            )
        if tool_name not in _ADMITTED_TOOLS:
            return MissionToolDecision(
                False,
                f"tool {tool_name!r} is outside the admitted strict-mission tool set",
            )
        if tool_name == "terminal":
            return MissionToolDecision(
                True,
                "terminal command is checked by the strict command guard",
            )
        if tool_name == MISSION_COMPLETE_TOOL_NAME:
            return MissionToolDecision(True, "mission completion tool is admitted")
        if tool_name in {"read_file", "search_files"}:
            return MissionToolDecision(True, "read/search tools are admitted")
        if tool_name == "patch":
            mode = args.get("mode", "replace")
            if mode == "patch":
                return MissionToolDecision(
                    False,
                    "patch mode=patch is unavailable in strict missions",
                )
            if mode not in (None, "replace"):
                return MissionToolDecision(
                    False,
                    f"patch mode {mode!r} is unavailable in strict missions",
                )
            if "path" not in args or not isinstance(args.get("path"), str) or not str(
                args.get("path")
            ).strip():
                return MissionToolDecision(
                    False,
                    "patch requires a resolvable path argument in strict missions",
                )
            return self._authorize_write_path(str(args["path"]))
        if tool_name == "write_file":
            if "path" not in args or not isinstance(args.get("path"), str) or not str(
                args.get("path")
            ).strip():
                return MissionToolDecision(
                    False,
                    "write_file requires a resolvable path argument in strict missions",
                )
            return self._authorize_write_path(str(args["path"]))
        return MissionToolDecision(
            False,
            f"tool {tool_name!r} is outside the admitted strict-mission tool set",
        )

    def _authorize_write_path(self, path_value: str) -> MissionToolDecision:
        try:
            target = _resolve_write_target(path_value)
        except (OSError, ValueError) as exc:
            return MissionToolDecision(
                False, f"write target could not be resolved: {exc}"
            )
        if target in self._protected_paths or _is_under(target, self._state_dir):
            return MissionToolDecision(
                False,
                f"write to protected runtime path is denied: {target}",
            )
        return MissionToolDecision(True, "write target is outside the protected set")

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

    @staticmethod
    def result_requests_completion(result: Any) -> bool:
        return _completion_requested_payload(result) is not None

    @contextlib.contextmanager
    def _lock_scope(self) -> Iterator[None]:
        """Acquire in-process lock then flock (§C.2). Release in reverse."""
        self._thread_lock.acquire()
        lock_fd: int | None = None
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(
                str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644
            )
            deadline = time.monotonic() + _LOCK_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise MissionRuntimeError(
                            "mission flock acquisition timed out"
                        )
                    time.sleep(_LOCK_RETRY_S)
                except InterruptedError:
                    continue
                except OSError as exc:
                    raise MissionRuntimeError(
                        f"mission flock acquisition failed: {exc}"
                    ) from exc
            try:
                yield
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            self._thread_lock.release()

    def _append_state_transition(self, status: str, event: str) -> None:
        with self._lock_scope():
            current = json.loads(self.state_path.read_text(encoding="utf-8"))
            history = list(current.get("history") or [])
            seq = _next_seq(history)
            history.append(
                {
                    "seq": seq,
                    "ts": _utc_now(),
                    "actor": _RUNTIME_ACTOR,
                    "event": event,
                }
            )
            revision = int(current.get("state_revision") or 0) + 1
            current["status"] = status
            current["history"] = history
            current["state_revision"] = revision
            self.state = current
            _atomic_json(self.state_path, current)

    def _evaluate_completion_predicates(
        self,
    ) -> tuple[list[dict[str, Any]], list[_PredicateFact], list[dict[str, Any]]]:
        """Pure predicate evaluation outside the lock (§E.2 / F9)."""
        assert self._completion_block is not None
        evidence: list[dict[str, Any]] = []
        facts: list[_PredicateFact] = []
        failures: list[dict[str, Any]] = []
        for pred in self._completion_block["predicates"]:
            path = Path(pred["path"])
            st = _lstat_facts(path)
            if st is None:
                item = {
                    "type": pred["type"],
                    "path": str(path),
                    "sha256": None,
                    "size_bytes": 0,
                    "passed": False,
                    "condition": "path does not exist",
                }
                evidence.append(item)
                failures.append(item)
                continue
            if not st["is_reg"]:
                item = {
                    "type": pred["type"],
                    "path": str(path),
                    "sha256": None,
                    "size_bytes": st["size"],
                    "passed": False,
                    "condition": "path is not a regular file (symlinks rejected)",
                }
                evidence.append(item)
                failures.append(item)
                continue
            if st["size"] <= 0:
                item = {
                    "type": pred["type"],
                    "path": str(path),
                    "sha256": None,
                    "size_bytes": 0,
                    "passed": False,
                    "condition": "file size is not above zero",
                }
                evidence.append(item)
                failures.append(item)
                continue
            digest = _sha256_file(path)
            item = {
                "type": pred["type"],
                "path": str(path),
                "sha256": digest,
                "size_bytes": st["size"],
                "passed": True,
            }
            evidence.append(item)
            facts.append(
                _PredicateFact(
                    path=path,
                    inode=st["inode"],
                    size=st["size"],
                    mtime_ns=st["mtime_ns"],
                    sha256=digest,
                )
            )
        return evidence, facts, failures

    def _recheck_predicate_facts(self, facts: list[_PredicateFact]) -> str | None:
        for fact in facts:
            st = _lstat_facts(fact.path)
            if st is None:
                return f"predicate path missing at commit: {fact.path}"
            if not st["is_reg"]:
                return f"predicate path is not a regular file at commit: {fact.path}"
            if st["size"] <= 0:
                return f"predicate file empty at commit: {fact.path}"
            if (
                st["inode"] != fact.inode
                or st["size"] != fact.size
                or st["mtime_ns"] != fact.mtime_ns
            ):
                return (
                    f"predicate file changed between evaluation and commit: {fact.path}"
                )
        return None

    def complete(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sole full-evaluation site for COMPLETED (§E.2)."""
        request = request or {}
        if self.halt_receipt is not None:
            return {
                "status": "refused",
                "reason": "mission already terminal in this run",
                "existing_receipt": self.halt_receipt.get("receipt_path"),
                "halt_receipt": self.halt_receipt,
            }

        status = self.state.get("status")
        if status == "BLOCKED":
            return {
                "status": "refused",
                "reason": (
                    "blocked mission cannot complete without the explicit resume "
                    "contract first"
                ),
            }
        if status == "COMPLETED":
            return {
                "status": "refused",
                "reason": "mission already COMPLETED",
                "existing_receipt": str(
                    (self._completion_block or {}).get("receipt_path")
                    or self.receipt_path
                ),
            }
        if self._completion_block is None:
            return {
                "status": "refused",
                "reason": "mission not completable by contract",
            }

        evidence, facts, failures = self._evaluate_completion_predicates()
        if failures:
            return {
                "status": "refused",
                "reason": "completion predicates failed",
                "failed_predicates": failures,
            }

        terminated_at = _utc_now()
        payload = {
            "terminated_at_utc": terminated_at,
            "session_id": str(request.get("session_id") or ""),
            "tool_call_id": str(request.get("tool_call_id") or ""),
            "completion_evidence": evidence,
            "predicate_facts": facts,
            "unresolved_followups": list(request.get("unresolved_followups") or []),
            "note": self._completion_block.get("note") or "",
        }
        return self._terminate("COMPLETED", payload)

    def handle_mission_complete_tool(
        self, tool_args: dict[str, Any] | None = None
    ) -> str:
        """Validate-only handler; resolution happens on the serial results path."""
        args = tool_args or {}
        return json.dumps(
            {
                "status": "completion_requested",
                "unresolved_followups": list(args.get("unresolved_followups") or []),
            },
            ensure_ascii=False,
        )

    def _terminate(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind not in {"BLOCKED", "COMPLETED"}:
            raise MissionRuntimeError(f"unsupported terminal kind: {kind}")
        if self.halt_receipt is not None:
            return self.halt_receipt

        if kind == "BLOCKED":
            return self._terminate_blocked(payload)
        return self._terminate_completed(payload)

    def _terminate_blocked(self, payload: dict[str, Any]) -> dict[str, Any]:
        blocked_at = payload["terminated_at_utc"]
        blocker = payload["blocker"]
        next_action = payload["next_legal_action"]
        session_id = payload["session_id"]
        tool_name = blocker["tool_name"]
        tool_call_id = blocker["tool_call_id"]

        with self._lock_scope():
            current = json.loads(self.state_path.read_text(encoding="utf-8"))
            status = current.get("status")
            if status in {"BLOCKED", "COMPLETED"}:
                existing = self.halt_receipt or {
                    "receipt_path": str(self.receipt_path),
                    "status": status,
                    "verdict": status,
                }
                _logger.info(
                    "second terminal attempt refused mission_id=%s status=%s",
                    self.mission_id,
                    status,
                )
                return {
                    "status": "refused",
                    "reason": f"mission already terminal ({status})",
                    "existing_receipt": existing.get("receipt_path"),
                    "halt_receipt": existing if self.halt_receipt is not None else None,
                    **(
                        {
                            k: existing[k]
                            for k in ("schema", "verdict", "receipt_path")
                            if k in existing
                        }
                    ),
                }

            history = list(current.get("history") or [])
            seq = _next_seq(history)
            history.append(
                {
                    "seq": seq,
                    "ts": blocked_at,
                    "actor": _RUNTIME_ACTOR,
                    "event": f"BLOCKED: {blocker['error']}",
                }
            )
            revision = int(current.get("state_revision") or 0) + 1
            current.update(
                {
                    "status": "BLOCKED",
                    "current_milestone": current.get("current_milestone"),
                    "next_legal_action": next_action,
                    "blocker": blocker,
                    "history": history,
                    "state_revision": revision,
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
                "state_revision": revision,
                "seq": seq,
                "terminated_at_utc": blocked_at,
                "runtime_version": _runtime_version(),
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

    def _terminate_completed(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._completion_block is not None
        facts: list[_PredicateFact] = list(payload["predicate_facts"])
        terminated_at = payload["terminated_at_utc"]
        session_id = payload["session_id"]
        tool_call_id = payload["tool_call_id"]
        evidence = payload["completion_evidence"]
        c_report = Path(self._completion_block["report_path"])
        c_checkpoint = Path(self._completion_block["checkpoint_path"])
        c_receipt = Path(self._completion_block["receipt_path"])
        c_notification = (
            Path(self._completion_block["notification_path"])
            if self._completion_block.get("notification_path")
            else None
        )

        with self._lock_scope():
            current = json.loads(self.state_path.read_text(encoding="utf-8"))
            status = current.get("status")
            if status in {"BLOCKED", "COMPLETED"}:
                _logger.info(
                    "second terminal attempt refused mission_id=%s status=%s",
                    self.mission_id,
                    status,
                )
                return {
                    "status": "refused",
                    "reason": f"mission already terminal ({status})",
                    "existing_receipt": str(c_receipt),
                }

            mismatch = self._recheck_predicate_facts(facts)
            if mismatch is not None:
                return {
                    "status": "refused",
                    "reason": "completion predicates failed under-lock re-check",
                    "failed_predicates": [
                        {
                            "type": "artifact_exists_nonempty",
                            "condition": mismatch,
                            "passed": False,
                        }
                    ],
                }

            history = list(current.get("history") or [])
            seq = _next_seq(history)
            history.append(
                {
                    "seq": seq,
                    "ts": terminated_at,
                    "actor": _RUNTIME_ACTOR,
                    "event": "COMPLETED: verified completion predicates passed",
                }
            )
            revision = int(current.get("state_revision") or 0) + 1
            current.update(
                {
                    "status": "COMPLETED",
                    "next_legal_action": "Mission completed; no further runtime action.",
                    "history": history,
                    "state_revision": revision,
                }
            )
            # Drop blocker if present from a prior resumed cycle.
            current.pop("blocker", None)
            _atomic_json(self.state_path, current)
            self.state = current

            note = payload.get("note") or ""
            report = (
                f"# Mission COMPLETED — {self.mission_id}\n\n"
                f"- Contract SHA-256: `{self.contract_sha256}`\n"
                f"- Completed at UTC: `{terminated_at}`\n"
                f"- Tool call: `{tool_call_id}`\n"
                f"- Note: {note}\n"
                f"- Predicates verified: {len(evidence)}\n"
            )
            report_path = _write_once(c_report, report)

            checkpoint = {
                "schema": COMPLETE_RECEIPT_SCHEMA,
                "mission_id": self.mission_id,
                "status": "COMPLETED",
                "terminated_at_utc": terminated_at,
                "contract_sha256": self.contract_sha256,
                "policy_sha256": self.policy_sha256,
                "state_revision": revision,
                "seq": seq,
                "runtime_version": _runtime_version(),
                "completion_evidence": evidence,
                "note": note,
            }
            checkpoint_path = _write_once(
                c_checkpoint,
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            )
            receipt = {
                **checkpoint,
                "session_id": session_id,
                "completion_requested_by": tool_call_id,
                "unresolved_followups": list(payload.get("unresolved_followups") or []),
                "artifacts": {
                    "report": str(report_path),
                    "report_sha256": _sha256_file(report_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "state": str(self.state_path),
                    "state_sha256": _sha256_file(self.state_path),
                },
                "founder_or_operator_interventions_after_start": [],
                "verdict": "COMPLETED",
            }
            if c_notification is not None:
                notification = (
                    f"# COMPLETE — {self.mission_id}\n\n"
                    "**From:** hermes-agent-mission-runtime\n\n"
                    f"Mission completed at `{terminated_at}`.\n\n"
                    f"Receipt target: `{c_receipt}`\n\n"
                    f"Note: {note}\n"
                )
                notification_path = _write_once(c_notification, notification)
                receipt["notification_path"] = str(notification_path)
                receipt["notification_sha256"] = _sha256_file(notification_path)

            try:
                receipt_path = _write_strict_once(
                    c_receipt,
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                )
                receipt["receipt_path"] = str(receipt_path)
                receipt["receipt_sha256"] = _sha256_file(receipt_path)
            except FileExistsError:
                _logger.error(
                    "terminal_incomplete_artifacts mission_id=%s status=COMPLETED "
                    "missing=receipt (O_EXCL integrity window; state stands)",
                    self.mission_id,
                )
                receipt["receipt_path"] = str(c_receipt)
                receipt["receipt_sha256"] = None
                receipt["integrity_error"] = "completion_receipt_oexcl_failed"

            self.halt_receipt = receipt
            return receipt

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
        result = self._terminate(
            "BLOCKED",
            {
                "terminated_at_utc": blocked_at,
                "blocker": blocker,
                "next_legal_action": next_action,
                "session_id": session_id,
            },
        )
        # Preserve prior block() return shape for in-run callers: on a refused
        # second terminal, return the existing halt receipt when available.
        if result.get("status") == "refused" and self.halt_receipt is not None:
            return self.halt_receipt
        if result.get("status") == "refused" and result.get("halt_receipt"):
            return result["halt_receipt"]
        return result

    @property
    def halt_response(self) -> str:
        if self.halt_receipt is not None:
            verdict = self.halt_receipt.get("verdict") or self.halt_receipt.get("status")
            receipt = self.halt_receipt.get("receipt_path")
            if verdict == "COMPLETED":
                return (
                    f"MISSION COMPLETED: {self.mission_id}. "
                    f"Receipt: {receipt}"
                )
            blocker = (self.halt_receipt.get("blocker") or {}).get("error")
            if blocker:
                return (
                    f"MISSION BLOCKED: {self.mission_id}. {blocker} "
                    f"Receipt: {receipt}"
                )
            return f"MISSION BLOCKED: {self.mission_id}. Receipt: {receipt}"

        status = self.state.get("status")
        if status == "COMPLETED":
            missing = (
                f" Missing artifacts: {', '.join(self._missing_artifacts)}."
                if self._missing_artifacts
                else ""
            )
            path = (
                (self._completion_block or {}).get("receipt_path")
                if self._completion_block
                else self.receipt_path
            )
            return (
                f"MISSION COMPLETED: {self.mission_id} is already complete."
                f"{missing} Receipt: {path}"
            )
        missing = (
            f" Missing artifacts: {', '.join(self._missing_artifacts)}."
            if self._missing_artifacts
            else ""
        )
        return (
            f"MISSION BLOCKED: {self.mission_id} remains durably blocked.{missing}"
        )


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
        return True
    except ValueError:
        return False


def get_active_mission_runtime() -> MissionRuntime | None:
    return _ACTIVE_RUNTIME.get()


@contextlib.contextmanager
def activate_mission_runtime(runtime: MissionRuntime | None) -> Iterator[None]:
    token = _ACTIVE_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME.reset(token)


def mission_complete_tool_schema() -> dict[str, Any]:
    return {
        "name": MISSION_COMPLETE_TOOL_NAME,
        "description": (
            "Request mechanical completion of the active strict mission after "
            "compiled completion predicates are satisfied. Model text alone "
            "never completes the mission."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "unresolved_followups": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional follow-ups that remain after completion.",
                },
            },
            "additionalProperties": False,
        },
    }
