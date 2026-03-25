"""Core ReAct loop implementation with tool registry and agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class ReasoningFn(Protocol):
    """Protocol for reasoning functions that produce thoughts and actions."""

    def __call__(self, task: str, history: list[Step]) -> ThoughtAction: ...


@dataclass
class Tool:
    """A tool that an agent can use to interact with the environment."""

    name: str
    description: str
    execute: Callable[[str], str]

    def run(self, input_text: str) -> str:
        """Execute the tool with the given input."""
        try:
            return self.execute(input_text)
        except Exception as e:
            return f"Error executing {self.name}: {e}"


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def describe(self) -> str:
        """Get a formatted description of all tools."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


@dataclass
class ThoughtAction:
    """A thought-action pair produced by reasoning."""

    thought: str
    action: str | None = None
    action_input: str = ""
    is_final: bool = False
    final_answer: str = ""


@dataclass
class Step:
    """A single step in the ReAct loop."""

    step_number: int
    thought: str
    action: str | None = None
    action_input: str = ""
    observation: str = ""

    def format(self) -> str:
        """Format this step as a readable string."""
        parts = [f"Step {self.step_number}:"]
        parts.append(f"  Thought: {self.thought}")
        if self.action:
            parts.append(f"  Action: {self.action}({self.action_input})")
            parts.append(f"  Observation: {self.observation}")
        return "\n".join(parts)


@dataclass
class Trace:
    """Complete execution trace of a ReAct run."""

    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_steps: int = 0

    def format(self) -> str:
        """Format the full trace as a readable string."""
        lines = [f"Task: {self.task}", ""]
        for step in self.steps:
            lines.append(step.format())
            lines.append("")
        lines.append(f"Final Answer: {self.final_answer}")
        lines.append(f"Steps: {self.total_steps}, Success: {self.success}")
        return "\n".join(lines)


class ReActAgent:
    """Agent that implements the ReAct (Reasoning + Acting) loop.

    The agent thinks step-by-step, selects tools, observes results,
    and iterates until it reaches a final answer or hits max steps.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        reasoning_fn: ReasoningFn | None = None,
        max_steps: int = 10,
    ) -> None:
        self.tools = tools
        self.max_steps = max_steps
        self._reasoning_fn = reasoning_fn or self._default_reasoning

    def run(self, task: str) -> Trace:
        """Execute the ReAct loop for a given task."""
        trace = Trace(task=task)
        history: list[Step] = []

        for i in range(1, self.max_steps + 1):
            thought_action = self._reasoning_fn(task, history)

            step = Step(
                step_number=i,
                thought=thought_action.thought,
                action=thought_action.action,
                action_input=thought_action.action_input,
            )

            if thought_action.is_final:
                trace.final_answer = thought_action.final_answer
                trace.success = True
                trace.steps.append(step)
                trace.total_steps = i
                return trace

            if thought_action.action:
                observation = self._execute_action(
                    thought_action.action, thought_action.action_input
                )
                step.observation = observation

            history.append(step)
            trace.steps.append(step)

        trace.total_steps = self.max_steps
        trace.final_answer = "Max steps reached without a final answer."
        return trace

    def _execute_action(self, action_name: str, action_input: str) -> str:
        """Execute a tool action and return the observation."""
        tool = self.tools.get(action_name)
        if tool is None:
            available = ", ".join(t.name for t in self.tools.list_tools())
            return f"Unknown tool '{action_name}'. Available: {available}"
        return tool.run(action_input)

    @staticmethod
    def _default_reasoning(task: str, history: list[Step]) -> ThoughtAction:
        """Default reasoning that tries each tool or finishes.

        This is a simple heuristic reasoner. For real use, replace with
        an LLM-based reasoning function.
        """
        if not history:
            return ThoughtAction(
                thought=f"I need to work on: {task}. Let me gather information.",
                action=None,
                is_final=False,
            )

        last = history[-1]
        if last.observation and "Error" not in last.observation:
            return ThoughtAction(
                thought=f"Based on the observation: {last.observation}, I can answer.",
                is_final=True,
                final_answer=last.observation,
            )

        return ThoughtAction(
            thought="I have enough context to provide an answer.",
            is_final=True,
            final_answer=f"Completed analysis of: {task}",
        )
