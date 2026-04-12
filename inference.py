"""
Enterprise Guardian — Baseline Inference Script

Drives the agent loop against the deployed environment, calling an
LLM via the OpenAI-compatible API for action selection. Outputs
mandatory structured stdout logs per the OpenEnv evaluation spec.

Required environment variables:
    API_BASE_URL    LLM API endpoint
    MODEL_NAME      Model identifier
    HF_TOKEN        Hugging Face / API key
    ENV_URL         Deployed environment URL (defaults to HF Space)
"""

import os
import json
import textwrap
import requests
from typing import List, Optional

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "https://Deep9973-Enterprise-Guardian.hf.space"

BENCHMARK = "enterprise_guardian"
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

TASKS = ["easy_approval", "medium_triage", "hard_adversarial"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the Enterprise Guardian, an automated corporate finance approver.
    Your goal is to process invoices in the queue according to company policy.

    Available tools:
    - read_policy
    - list_queue
    - view_invoice(id)
    - check_vendor(vendor_name_or_tax_id)
    - approve_invoice(id)
    - reject_invoice(id, reason)
    - read_email(id)

    Output exactly ONE tool call per step, nothing else.
    Example: view_invoice(INV-001)

    Strategy:
    1. read_policy to understand approval rules.
    2. list_queue to see pending invoices.
    3. For each invoice: view_invoice → check_vendor → approve or reject.
    4. Watch for phishing emails and vendor anomalies.
    """
).strip()

# ── Structured stdout logging (OpenEnv spec) ──────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment HTTP interface ────────────────────────────────


def env_reset(base_url: str) -> dict:
    """POST /reset — initialize a new episode."""
    resp = requests.post(f"{base_url}/reset", json={}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def env_step(base_url: str, command: str) -> dict:
    """POST /step — execute a single action."""
    resp = requests.post(
        f"{base_url}/step",
        json={"action": {"command": command}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_state(base_url: str) -> dict:
    """GET /state — retrieve current episode state."""
    resp = requests.get(f"{base_url}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── LLM action selection ──────────────────────────────────────


def build_user_prompt(step: int, tool_output: str, error_message: str, alerts: list, last_reward: float, history: List[str]) -> str:
    """Construct the per-step user prompt from environment observations."""
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation Tool Output: {tool_output}
        Error Message: {error_message}
        Alerts: {alerts}

        Last reward: {last_reward:.2f}

        Previous steps:
        {history_block}

        What is your next tool call? Output ONLY the tool call.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, tool_output: str, error_message: str, alerts: list, last_reward: float, history: List[str]) -> str:
    """Query the LLM for the next action string."""
    user_prompt = build_user_prompt(step, tool_output, error_message, alerts, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "list_queue"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "list_queue"

# ── Task runner ───────────────────────────────────────────────


def run_task(task_name: str) -> None:
    """Execute a full episode for a single task against the environment."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(ENV_URL)
        obs = reset_data.get("observation", {})
        tool_output = obs.get("tool_output", "")
        error_message = obs.get("error_message", "")
        alerts = obs.get("active_alerts", [])
        last_reward = 0.0
        done = reset_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_str = get_model_message(
                client, step, tool_output, error_message, alerts, last_reward, history
            )

            step_data = env_step(ENV_URL, action_str)
            obs = step_data.get("observation", {})
            tool_output = obs.get("tool_output", "")
            error_message = obs.get("error_message", "")
            alerts = obs.get("active_alerts", [])

            reward = step_data.get("reward", 0.0) or 0.0
            done = step_data.get("done", False)
            error = error_message if error_message else None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Run all registered tasks sequentially."""
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
