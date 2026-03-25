"""Utility functions for ReAct pattern helpers."""

from __future__ import annotations

import re


def parse_thought_action(text: str) -> dict[str, str]:
    """Parse a thought-action string into components.

    Expected format:
        Thought: I need to calculate something
        Action: calc
        Action Input: 2 + 2
    """
    result: dict[str, str] = {}

    thought_match = re.search(r"Thought:\s*(.+?)(?=\n|Action:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text, re.DOTALL)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n|$)", text, re.DOTALL)
    if answer_match:
        result["final_answer"] = answer_match.group(1).strip()

    return result


def format_tool_descriptions(tools: list[dict[str, str]]) -> str:
    """Format tool descriptions for inclusion in prompts."""
    lines = []
    for tool in tools:
        lines.append(f"{tool['name']}: {tool['description']}")
    return "\n".join(lines)


def format_history(steps: list[dict[str, str]]) -> str:
    """Format step history for inclusion in prompts."""
    lines = []
    for step in steps:
        lines.append(f"Thought: {step.get('thought', '')}")
        if step.get("action"):
            lines.append(f"Action: {step['action']}")
            lines.append(f"Action Input: {step.get('action_input', '')}")
            lines.append(f"Observation: {step.get('observation', '')}")
    return "\n".join(lines)


def safe_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return f"Error: Invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
