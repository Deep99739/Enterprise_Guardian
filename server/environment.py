import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from openenv_core.env_server import Environment

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
    """
    Corporate Finance Approver Environment.
    
    The agent acts as a finance approver, managing a queue of invoices and
    emails. It must read the policy, check vendors, and approve or reject
    invoices while avoiding adversarial traps like social engineering.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "easy_approval", **kwargs):
        super().__init__(**kwargs)
        self.default_task_name = task_name
        self._state = EnterpriseGuardianState()
        self._invoices = []
        self._emails = []
        self._policy = ""
        self._config = {}
        self._decisions = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> EnterpriseGuardianObservation:
        """Reset the environment to its initial state."""
        # Use provided task_name or default
        task_name = kwargs.get("task_name", self.default_task_name)
        if task_name not in TASK_CONFIGS:
            logger.warning(f"Unknown task {task_name}, defaulting to easy_approval")
            task_name = "easy_approval"
            
        self._config = TASK_CONFIGS[task_name]
        seed = seed if seed is not None else 42
        
        # Determine episode ID
        ep_id = episode_id if episode_id else str(uuid.uuid4())

        # Generate data
        self._invoices = generate_invoices(seed, task_name)
        self._emails = generate_emails(seed, task_name)
        self._policy = generate_policy()
        self._decisions = []

        # Count adversarial traps
        adv_traps = sum(1 for inv in self._invoices if inv.get("is_adversarial"))
        adv_traps += sum(1 for email in self._emails if email.get("is_adversarial"))

        # Initialize state
        self._state = EnterpriseGuardianState(
            episode_id=ep_id,
            step_count=0,
            task_name=task_name,
            difficulty="hard" if "hard" in task_name else ("medium" if "medium" in task_name else "easy"),
            invoices_total=len(self._invoices),
            invoices_processed=0,
            correct_decisions=0,
            policy_read=False,
            vendor_checks_made=0,
            adversarial_traps_resisted=0,
            adversarial_traps_total=adv_traps,
            schema_drift_active=False,
            cumulative_reward=0.0
        )
        self._state.invalid_actions = 0 # custom tracking variable

        # Create initial observation
        return EnterpriseGuardianObservation(
            done=False,
            reward=0.0,
            tool_output="Environment initialized. Type a command to start.",
            inbox_summary=self._generate_inbox_summary(),
            queue_summary=self._generate_queue_summary(),
            steps_taken=0,
            max_steps=self._config["max_steps"]
        )

    def step(self, action: EnterpriseGuardianAction, timeout_s: Optional[float] = None, **kwargs) -> EnterpriseGuardianObservation:
        """Process one step in the environment."""
        self._state.step_count += 1
        
        command = action.command.strip()
        tool_output = ""
        error_msg = ""
        action_result = {"correct": False, "error": False, "repeated_action": False}
        action_type = ""

        # Schema drift check
        if self._config.get("has_schema_drift") and self._state.step_count >= self._config.get("schema_drift_step", 999):
            self._state.schema_drift_active = True

        # Parse basic command structure: tool_name(args)
        match = re.match(r"^([a-zA-Z_]+)(?:\((.*)\))?$", command)
        
        if not match:
            tool_output = "Invalid command format. Use tool_name(args) or tool_name."
            error_msg = tool_output
            action_result["error"] = True
        else:
            action_type = match.group(1)
            args_str = match.group(2) if match.group(2) else ""
            args = [arg.strip() for arg in args_str.split(',')] if args_str else []
            # remove quotes if present around args
            args = [arg.strip("\"'") for arg in args]

            # Dispatch tool
            if action_type == "read_policy":
                tool_output = self._handle_read_policy(action_result)
            elif action_type == "list_queue":
                tool_output = self._handle_list_queue(action_result)
            elif action_type == "view_invoice":
                tool_output = self._handle_view_invoice(args, action_result)
            elif action_type == "read_email":
                tool_output = self._handle_read_email(args, action_result)
            elif action_type == "check_vendor":
                tool_output = self._handle_check_vendor(args, action_result)
            elif action_type == "approve_invoice":
                tool_output = self._handle_approve_invoice(args, action_result)
            elif action_type == "reject_invoice":
                tool_output = self._handle_reject_invoice(args, action_result)
            else:
                tool_output = f"Unknown tool: {action_type}"
                error_msg = tool_output
                action_result["error"] = True

        if action_result.get("error"):
            self._state.invalid_actions += 1

        # Check terminal conditions
        all_processed = self._state.invoices_processed >= self._state.invoices_total
        out_of_steps = self._state.step_count >= self._config["max_steps"]
        done = all_processed or out_of_steps

        # Compute rewards
        if done:
            breakdown = compute_episode_score(self._decisions, self.__dict__.get('_state').__dict__, self._config)
            reward = breakdown.r_total
            self._state.cumulative_reward = reward
        else:
            step_reward = compute_step_reward(action_type, action_result, self.__dict__.get('_state').__dict__)
            reward = step_reward

        # Alerts (e.g. schema drift warning)
        alerts = []
        if self._state.schema_drift_active and self._state.step_count == self._config.get("schema_drift_step"):
             alerts.append("NOTICE: Vendor API updated. 'vendor_name' is deprecated.")

        return EnterpriseGuardianObservation(
            done=done,
            reward=reward,
            tool_output=tool_output,
            inbox_summary=self._generate_inbox_summary(),
            queue_summary=self._generate_queue_summary(),
            policy_snippet="Policy read." if self._state.policy_read else "Policy not read yet.",
            steps_taken=self._state.step_count,
            max_steps=self._config["max_steps"],
            error_message=error_msg,
            active_alerts=alerts
        )

    @property
    def state(self) -> EnterpriseGuardianState:
        return self._state

    # --- Tool Handlers ---

    def _handle_read_policy(self, result: Dict) -> str:
        if self._state.policy_read:
            result["repeated_action"] = True
        self._state.policy_read = True
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
                    "id": inv["id"],
                    "vendor_name": inv["vendor_name"],
                    "vendor_tax_id": inv["vendor_tax_id"],
                    "amount": inv["amount"],
                    "category": inv["category"],
                    "department": inv["department"],
                    "description": inv["description"],
                    "requestor": inv["requestor"],
                    "priority": inv["priority"],
                    "po_number": inv.get("po_number"),
                    "status": status
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
        
        self._state.vendor_checks_made += 1
        query = args[0]

        if self._state.schema_drift_active:
            # Enforce tax_id lookup
            info = get_vendor_info_by_tax_id(query)
            if info:
                return json.dumps(info, indent=2)
            else:
                result["error"] = True
                return "Error: Vendor not found by Tax ID, or invalid Tax ID format."
        else:
            # Allow name lookup, but tax_id also works
            info = get_vendor_info(query)
            if not info:
                 info = get_vendor_info_by_tax_id(query)
            
            if info:
                return json.dumps(info, indent=2)
            else:
                result["error"] = True
                return "Error: Vendor not found."

    def _handle_approve_invoice(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing invoice ID."
        inv_id = args[0]
        return self._process_decision(inv_id, "approve", result)

    def _handle_reject_invoice(self, args: List[str], result: Dict) -> str:
        if not args or not args[0]:
            result["error"] = True
            return "Error: Missing invoice ID."
        if len(args) < 2:
            result["error"] = True
            return "Error: Must provide a reason to reject an invoice."
        
        inv_id = args[0]
        reason = args[1]
        return self._process_decision(inv_id, "reject", result, reason)

    def _process_decision(self, inv_id: str, action: str, result: Dict, reason: str = "") -> str:
        # Check if already processed
        if any(d["invoice_id"] == inv_id for d in self._decisions):
             result["error"] = True
             return f"Error: Invoice {inv_id} has already been processed."

        # Find invoice
        invoice = next((i for i in self._invoices if i["id"] == inv_id), None)
        if not invoice:
            result["error"] = True
            return f"Error: Invoice {inv_id} not found."

        expected = invoice["expected_action"]
        is_correct = (action == expected)
        
        result["correct"] = is_correct

        if is_correct:
            self._state.correct_decisions += 1
            if invoice.get("is_adversarial"):
                self._state.adversarial_traps_resisted += 1
                result["adversarial_resisted"] = True

        self._state.invoices_processed += 1
        
        self._decisions.append({
            "invoice_id": inv_id,
            "action": action,
            "correct": is_correct,
            "expected": expected,
            "reason": reason
        })

        if action == "approve":
            return f"Invoice {inv_id} APPROVED."
        else:
            return f"Invoice {inv_id} REJECTED. Reason: {reason}"

    # --- Helpers ---

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
        lines = ["--- PENDING INVOICES ---"]
        processed_ids = {d["invoice_id"] for d in self._decisions}
        for inv in self._invoices:
            status = "PROCESSED" if inv["id"] in processed_ids else "PENDING"
            lines.append(f"[{inv['id']}] {status} - {inv['vendor_name']} : ${inv['amount']:.2f}")
        return "\n".join(lines)
