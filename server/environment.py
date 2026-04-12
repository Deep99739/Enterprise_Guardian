"""
Enterprise Guardian — Core Environment

Implements the OpenEnv Environment interface for a corporate finance
approval simulation. Agents process invoice queues, verify vendors,
and make approve/reject decisions under policy constraints while
resisting adversarial social engineering attacks.

Difficulty tiers:
    easy_approval      3 invoices, no adversarial content
    medium_triage      3 invoices with policy edge-cases, distractor emails
    hard_adversarial   4 invoices with phishing, unverified vendors, schema drift
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from ..data import (
    TASK_CONFIGS,
    generate_emails,
    generate_invoices,
    generate_policy,
    get_vendor_info,
    get_vendor_info_by_tax_id,
)
from ..models import (
    EnterpriseGuardianAction,
    EnterpriseGuardianObservation,
    EnterpriseGuardianState,
)
from ..rewards import compute_episode_score, compute_step_reward

logger = logging.getLogger(__name__)


class EnterpriseGuardianEnvironment(Environment):
    """Corporate finance approval environment with adversarial mechanics.

    Manages invoice queues, email inboxes, vendor databases, and policy
    documents. Tracks per-episode state in a mutable dict to avoid
    Pydantic immutability constraints imposed by the OpenEnv base class.
    """
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        self.default_task_name = "easy_approval"
        self._config = TASK_CONFIGS.get("easy_approval", {"max_steps": 15})
        self._s = {
            "episode_id": None,
            "step_count": 0,
            "task_name": "",
            "difficulty": "easy",
            "invoices_total": 0,
            "invoices_processed": 0,
            "correct_decisions": 0,
            "policy_read": False,
            "vendor_checks_made": 0,
            "adversarial_traps_resisted": 0,
            "adversarial_traps_total": 0,
            "schema_drift_active": False,
            "cumulative_reward": 0.0,
            "invalid_actions": 0,
        }
        self._invoices: List[Dict] = []
        self._emails: List[Dict] = []
        self._policy: str = ""
        self._decisions: List[Dict] = []

    def reset(self, **kwargs) -> EnterpriseGuardianObservation:
        """Initialize a fresh episode with seeded data generation."""
        task_name = kwargs.get("task_name", self.default_task_name)
        if task_name not in TASK_CONFIGS:
            logger.warning(f"Unknown task '{task_name}', falling back to easy_approval")
            task_name = "easy_approval"

        self._config = TASK_CONFIGS[task_name]
        seed = kwargs.get("seed", 42)
        ep_id = kwargs.get("episode_id") or str(uuid.uuid4())

        self._invoices = generate_invoices(seed, task_name)
        self._emails = generate_emails(seed, task_name)
        self._policy = generate_policy()
        self._decisions = []

        adv_traps = sum(1 for inv in self._invoices if inv.get("is_adversarial"))
        adv_traps += sum(1 for email in self._emails if email.get("is_adversarial"))

        self._s = {
            "episode_id": ep_id,
            "step_count": 0,
            "task_name": task_name,
            "difficulty": "hard" if "hard" in task_name else ("medium" if "medium" in task_name else "easy"),
            "invoices_total": len(self._invoices),
            "invoices_processed": 0,
            "correct_decisions": 0,
            "policy_read": False,
            "vendor_checks_made": 0,
            "adversarial_traps_resisted": 0,
            "adversarial_traps_total": adv_traps,
            "schema_drift_active": False,
            "cumulative_reward": 0.0,
            "invalid_actions": 0,
        }

        return EnterpriseGuardianObservation(
            done=False,
            reward=0.0,
            tool_output="Environment initialized. Type a command to start.",
            inbox_summary=self._generate_inbox_summary(),
            queue_summary=self._generate_queue_summary(),
            steps_taken=0,
            max_steps=self._config["max_steps"],
        )

    def step(self, action: EnterpriseGuardianAction, **kwargs) -> EnterpriseGuardianObservation:
        """Process a single agent action and return the resulting observation."""
        # Guard: auto-initialize if step() is called without a prior reset()
        if not self._config or "max_steps" not in self._config:
            logger.warning("step() invoked without preceding reset(); auto-initializing")
            self._config = TASK_CONFIGS.get("easy_approval", {"max_steps": 15})
            self._invoices = generate_invoices(42, "easy_approval")
            self._emails = generate_emails(42, "easy_approval")
            self._policy = generate_policy()

        self._s["step_count"] += 1

        command = action.command.strip()
        tool_output = ""
        error_msg = ""
        action_result = {"correct": False, "error": False, "repeated_action": False}
        action_type = ""

        # Activate schema drift if configured for this task
        if self._config.get("has_schema_drift", False) and self._s["step_count"] >= self._config.get("schema_drift_step", 999):
            self._s["schema_drift_active"] = True

        # Parse command: expected format is tool_name or tool_name(arg1, arg2)
        match = re.match(r"^([a-zA-Z_]+)(?:\((.*)\))?$", command)

        if not match:
            tool_output = "Invalid command format. Use tool_name(args) or tool_name."
            error_msg = tool_output
            action_result["error"] = True
        else:
            action_type = match.group(1)
            args_str = match.group(2) if match.group(2) else ""
            args = [arg.strip().strip("\"'") for arg in args_str.split(",")] if args_str else []

            tool_output, error_msg = self._dispatch_tool(action_type, args, action_result)

        if action_result.get("error"):
            self._s["invalid_actions"] += 1

        # Terminal conditions
        all_processed = self._s["invoices_processed"] >= self._s["invoices_total"]
        max_steps = self._config.get("max_steps", 15)
        out_of_steps = self._s["step_count"] >= max_steps
        done = all_processed or out_of_steps

        # Reward computation
        if done:
            try:
                breakdown = compute_episode_score(self._decisions, self._s, self._config)
                reward = breakdown.r_total
            except Exception as e:
                logger.error(f"Episode score computation failed: {e}")
                reward = 0.01
            self._s["cumulative_reward"] = reward
        else:
            reward = compute_step_reward(action_type, action_result, self._s)

        # Schema drift alert (fires once on activation step)
        alerts = []
        if self._s.get("schema_drift_active", False) and self._s["step_count"] == self._config.get("schema_drift_step", 999):
            alerts.append("NOTICE: Vendor API updated. 'vendor_name' is deprecated.")

        return EnterpriseGuardianObservation(
            done=done,
            reward=reward,
            tool_output=tool_output,
            inbox_summary=self._generate_inbox_summary(),
            queue_summary=self._generate_queue_summary(),
            policy_snippet="Policy read." if self._s.get("policy_read", False) else "Policy not read yet.",
            steps_taken=self._s["step_count"],
            max_steps=self._config.get("max_steps", 15),
            error_message=error_msg,
            active_alerts=alerts,
        )

    @property
    def state(self) -> EnterpriseGuardianState:
        """Construct an immutable snapshot of the current episode state."""
        return EnterpriseGuardianState(
            episode_id=self._s.get("episode_id"),
            step_count=self._s.get("step_count", 0),
            task_name=self._s.get("task_name", ""),
            difficulty=self._s.get("difficulty", "easy"),
            invoices_total=self._s.get("invoices_total", 0),
            invoices_processed=self._s.get("invoices_processed", 0),
            correct_decisions=self._s.get("correct_decisions", 0),
            policy_read=self._s.get("policy_read", False),
            vendor_checks_made=self._s.get("vendor_checks_made", 0),
            adversarial_traps_resisted=self._s.get("adversarial_traps_resisted", 0),
            adversarial_traps_total=self._s.get("adversarial_traps_total", 0),
            schema_drift_active=self._s.get("schema_drift_active", False),
            cumulative_reward=self._s.get("cumulative_reward", 0.0),
        )

    # ── Tool dispatch ──────────────────────────────────────────────

    def _dispatch_tool(self, action_type: str, args: List[str], result: Dict) -> tuple:
        """Route an action to the appropriate tool handler. Returns (output, error_msg)."""
        handlers = {
            "read_policy": lambda: (self._handle_read_policy(result), ""),
            "list_queue": lambda: (self._handle_list_queue(result), ""),
            "view_invoice": lambda: (self._handle_view_invoice(args, result), ""),
            "read_email": lambda: (self._handle_read_email(args, result), ""),
            "check_vendor": lambda: (self._handle_check_vendor(args, result), ""),
            "approve_invoice": lambda: (self._handle_approve_invoice(args, result), ""),
            "reject_invoice": lambda: (self._handle_reject_invoice(args, result), ""),
        }

        handler = handlers.get(action_type)
        if handler:
            output, _ = handler()
            error_msg = output if result.get("error") else ""
            return output, error_msg

        error_output = f"Unknown tool: {action_type}"
        result["error"] = True
        return error_output, error_output

    # ── Tool handlers ──────────────────────────────────────────────

    def _handle_read_policy(self, result: Dict) -> str:
        if self._s["policy_read"]:
            result["repeated_action"] = True
        self._s["policy_read"] = True
        return self._policy

    def _handle_list_queue(self, result: Dict) -> str:
        return self._generate_queue_summary()

    def _handle_view_invoice(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing invoice ID."
        inv_id = args[0]
        for inv in self._invoices:
            if inv["id"] == inv_id:
                status = "PENDING"
                for d in self._decisions:
                    if d["invoice_id"] == inv_id:
                        status = d["action"].upper()
                return json.dumps({
                    "id": inv["id"], "vendor_name": inv["vendor_name"],
                    "vendor_tax_id": inv["vendor_tax_id"], "amount": inv["amount"],
                    "category": inv["category"], "department": inv["department"],
                    "description": inv["description"], "requestor": inv["requestor"],
                    "priority": inv["priority"], "po_number": inv.get("po_number"),
                    "status": status,
                }, indent=2)
        result["error"] = True
        return f"Error: Invoice {inv_id} not found."

    def _handle_read_email(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing email ID."
        email_id = args[0]
        for email in self._emails:
            if email["id"] == email_id:
                return f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
        result["error"] = True
        return f"Error: Email {email_id} not found."

    def _handle_check_vendor(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing vendor name or tax ID."

        self._s["vendor_checks_made"] += 1
        query = args[0]

        if self._s["schema_drift_active"]:
            info = get_vendor_info_by_tax_id(query)
            if info:
                return json.dumps(info, indent=2)
            result["error"] = True
            return "Error: Vendor not found by Tax ID, or invalid Tax ID format."

        info = get_vendor_info(query) or get_vendor_info_by_tax_id(query)
        if info:
            return json.dumps(info, indent=2)
        result["error"] = True
        return "Error: Vendor not found."

    def _handle_approve_invoice(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing invoice ID."
        return self._process_decision(args[0], "approve", result)

    def _handle_reject_invoice(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing invoice ID."
        if len(args) < 2:
            result["error"] = True
            return "Error: Must provide a reason to reject an invoice."
        return self._process_decision(args[0], "reject", result, args[1])

    def _process_decision(self, inv_id: str, action: str, result: Dict, reason: str = "") -> str:
        """Record an approve/reject decision and update episode metrics."""
        if any(d["invoice_id"] == inv_id for d in self._decisions):
            result["error"] = True
            return f"Error: Invoice {inv_id} has already been processed."

        invoice = next((i for i in self._invoices if i["id"] == inv_id), None)
        if not invoice:
            result["error"] = True
            return f"Error: Invoice {inv_id} not found."

        is_correct = action == invoice["expected_action"]
        result["correct"] = is_correct

        if is_correct:
            self._s["correct_decisions"] += 1
            if invoice.get("is_adversarial"):
                self._s["adversarial_traps_resisted"] += 1
                result["adversarial_resisted"] = True

        self._s["invoices_processed"] += 1
        self._decisions.append({
            "invoice_id": inv_id, "action": action,
            "correct": is_correct, "expected": invoice["expected_action"],
            "reason": reason,
        })

        return f"Invoice {inv_id} {'APPROVED' if action == 'approve' else 'REJECTED. Reason: ' + reason}."

    # ── Observation helpers ────────────────────────────────────────

    def _generate_inbox_summary(self) -> str:
        if not self._emails:
            return "Inbox empty."
        lines = ["--- INBOX ---"]
        for email in self._emails:
            lines.append(f"[{email['id']}] From: {email['from']} - {email['subject']}")
        return "\n".join(lines)

    def _generate_queue_summary(self) -> str:
        if not self._invoices:
            return "Queue empty."
        processed_ids = {d["invoice_id"] for d in self._decisions}
        lines = ["--- PENDING INVOICES ---"]
        for inv in self._invoices:
            status = "PROCESSED" if inv["id"] in processed_ids else "PENDING"
            lines.append(f"[{inv['id']}] {status} - {inv['vendor_name']} : ${inv['amount']:.2f}")
        return "\n".join(lines)
