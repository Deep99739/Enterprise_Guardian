"""
Enterprise Guardian — Environment Client

WebSocket-based client for interacting with the deployed
Enterprise Guardian environment via the OpenEnv EnvClient protocol.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from .models import (
    EnterpriseGuardianAction,
    EnterpriseGuardianObservation,
    EnterpriseGuardianState,
)


class EnterpriseGuardianEnv(EnvClient[EnterpriseGuardianAction, EnterpriseGuardianObservation, EnterpriseGuardianState]):
    """Client interface for the Enterprise Guardian environment.

    Provides deserialization of raw API payloads into typed domain models
    for use in training loops and evaluation scripts.
    """

    def _step_payload(self, action: EnterpriseGuardianAction) -> dict:
        return {"command": action.command}

    def _parse_result(self, payload: dict) -> StepResult[EnterpriseGuardianObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=EnterpriseGuardianObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                tool_output=obs_data.get("tool_output", ""),
                inbox_summary=obs_data.get("inbox_summary", ""),
                queue_summary=obs_data.get("queue_summary", ""),
                policy_snippet=obs_data.get("policy_snippet", ""),
                steps_taken=obs_data.get("steps_taken", 0),
                max_steps=obs_data.get("max_steps", 15),
                active_alerts=obs_data.get("active_alerts", []),
                error_message=obs_data.get("error_message", "")
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> EnterpriseGuardianState:
        return EnterpriseGuardianState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            difficulty=payload.get("difficulty", "easy"),
            invoices_total=payload.get("invoices_total", 0),
            invoices_processed=payload.get("invoices_processed", 0),
            correct_decisions=payload.get("correct_decisions", 0),
            policy_read=payload.get("policy_read", False),
            vendor_checks_made=payload.get("vendor_checks_made", 0),
            adversarial_traps_resisted=payload.get("adversarial_traps_resisted", 0),
            adversarial_traps_total=payload.get("adversarial_traps_total", 0),
            schema_drift_active=payload.get("schema_drift_active", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0)
        )
