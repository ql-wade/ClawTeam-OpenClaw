"""PhaseController — background process that spawns agents when gating tasks complete.

Launched by _start_phase_controller() in commands.py as a tmux pane or subprocess.
Reads config from ~/.clawteam/teams/{team}/phase-controller.json.

Usage:
    python -m clawteam.team.phase_controller <team_name>
    python clawteam/team/phase_controller.py <team_name>
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [phase-ctrl] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(team_name: str) -> dict | None:
    """Load the phase controller config written by _start_phase_controller."""
    from clawteam.team.models import get_data_dir

    config_path = get_data_dir() / "teams" / team_name / "phase-controller.json"
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def _task_is_completed(team_name: str, task_id: str) -> bool:
    """Check if a task has status == completed."""
    from clawteam.store.file import FileTaskStore

    ts = FileTaskStore(team_name)
    task = ts.get(task_id)
    if task is None:
        return False
    return task.status.value == "completed"


def _spawn_agent(cfg: dict, agent_cfg: dict) -> str:
    """Spawn a single delayed agent now that its gate is open."""
    from clawteam.spawn import get_backend, spawn_with_retry
    from clawteam.spawn.prompt import build_agent_prompt
    from clawteam.model_resolution import resolve_model
    from clawteam.config import load_config
    from clawteam.templates import render_task

    team_name = cfg["team_name"]
    be = get_backend(cfg["backend_name"])

    # Variable substitution in task
    rendered = render_task(
        agent_cfg["task"],
        goal=cfg["goal"],
        team_name=team_name,
        agent_name=agent_cfg["name"],
    )

    # Build prompt
    prompt = build_agent_prompt(
        agent_name=agent_cfg["name"],
        agent_id=agent_cfg["agent_id"],
        agent_type=agent_cfg["type"],
        team_name=team_name,
        leader_name=cfg["leader_name"],
        task=rendered,
        memory_scope=f"custom:team-{team_name}",
        intent=agent_cfg.get("intent") or "",
        end_state=agent_cfg.get("end_state") or "",
        constraints=agent_cfg.get("constraints"),
        team_size=len(cfg["agents"]) + 1,  # +1 for leader
    )

    # Resolve model
    _mc = load_config()
    model = resolve_model(
        cli_model=cfg.get("model_override"),
        agent_model=agent_cfg.get("model"),
        agent_model_tier=agent_cfg.get("model_tier"),
        template_model_strategy=cfg.get("model_strategy_override"),
        template_model=None,
        config_default_model=_mc.default_model,
        agent_type=agent_cfg["type"],
        tier_overrides=_mc.model_tiers or None,
    )

    # Command
    cmd = agent_cfg.get("command") or cfg.get("command_override") or ["openclaw"]

    # Spawn
    kwargs = dict(
        command=cmd,
        agent_name=agent_cfg["name"],
        agent_id=agent_cfg["agent_id"],
        agent_type=agent_cfg["type"],
        team_name=team_name,
        prompt=prompt,
        skip_permissions=True,
        model=model,
    )
    retry_max = agent_cfg.get("retry_max", 0)
    if retry_max > 0:
        return spawn_with_retry(
            be,
            max_retries=retry_max,
            backoff_base=agent_cfg.get("retry_backoff_base", 1.0),
            backoff_max=agent_cfg.get("retry_backoff_max", 30.0),
            **kwargs,
        )
    return be.spawn(**kwargs)


def _report(team_name: str, leader_name: str, message: str) -> None:
    """Send a message to the team leader inbox (for IM relay via Kev)."""
    try:
        from clawteam.team.mailbox import MailboxManager
        mb = MailboxManager(team_name)
        mb.send(from_agent="phase-controller", to=leader_name, content=message)
    except Exception as exc:
        logger.warning("Failed to report: %s", exc)


def run(team_name: str) -> None:
    """Main loop: poll for gating task completions and spawn agents."""
    cfg = _load_config(team_name)
    if cfg is None:
        sys.exit(1)

    leader_name = cfg["leader_name"]
    name_to_id = cfg.get("name_to_id", {})
    poll_interval = cfg.get("poll_interval_seconds", 5)
    gate_timeout = cfg.get("gate_timeout_seconds", 3600)

    pending = list(cfg["agents"])  # agents still waiting to be spawned
    logger.info(
        "PhaseController started for team '%s' — %d delayed agents, poll=%ds, timeout=%ds",
        team_name, len(pending), poll_interval, gate_timeout,
    )
    _report(team_name, leader_name, f"PhaseController started: {len(pending)} agents queued")

    elapsed = 0.0
    while pending:
        still_pending = []
        for agent_cfg in pending:
            gate_task_name = agent_cfg["spawn_after_task_name"]
            gate_task_id = name_to_id.get(gate_task_name, "")

            if not gate_task_id:
                logger.warning("No task ID for gate '%s' — skipping agent '%s'",
                               gate_task_name, agent_cfg["name"])
                continue

            if _task_is_completed(team_name, gate_task_id):
                logger.info("Gate '%s' completed — spawning agent '%s'",
                            gate_task_name, agent_cfg["name"])
                _report(
                    team_name, leader_name,
                    f"Gate '{gate_task_name}' passed — spawning {agent_cfg['name']}",
                )
                result = _spawn_agent(cfg, agent_cfg)
                logger.info("Spawn result for '%s': %s", agent_cfg["name"], result)
            else:
                still_pending.append(agent_cfg)

        pending = still_pending

        if pending:
            if elapsed >= gate_timeout:
                logger.error("Timeout (%ds) — %d agents still gated", gate_timeout, len(pending))
                _report(
                    team_name, leader_name,
                    f"PhaseController TIMEOUT after {gate_timeout}s — "
                    f"{len(pending)} agents never spawned: "
                    + ", ".join(a["name"] for a in pending),
                )
                sys.exit(1)
            time.sleep(poll_interval)
            elapsed += poll_interval

    logger.info("All delayed agents spawned — PhaseController exiting")
    _report(team_name, leader_name, "All phases complete — pipeline finished")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <team_name>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
