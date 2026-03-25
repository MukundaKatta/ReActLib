"""Tests for ReActLib core components."""

from reactlib.core import ReActAgent, Tool, ToolRegistry, ThoughtAction, Step
from reactlib.utils import parse_thought_action, safe_eval


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = Tool(name="test", description="A test tool", execute=lambda x: f"got: {x}")
        registry.register(tool)
        assert "test" in registry
        assert registry.get("test") is tool
        assert len(registry) == 1

    def test_list_and_describe(self) -> None:
        registry = ToolRegistry()
        registry.register(Tool(name="a", description="Tool A", execute=lambda x: x))
        registry.register(Tool(name="b", description="Tool B", execute=lambda x: x))
        assert len(registry.list_tools()) == 2
        desc = registry.describe()
        assert "a: Tool A" in desc
        assert "b: Tool B" in desc

    def test_unknown_tool(self) -> None:
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None


class TestTool:
    def test_execute(self) -> None:
        tool = Tool(name="echo", description="Echo input", execute=lambda x: f"echo: {x}")
        assert tool.run("hello") == "echo: hello"

    def test_execute_error(self) -> None:
        def bad_fn(x: str) -> str:
            raise ValueError("boom")
        tool = Tool(name="bad", description="Broken", execute=bad_fn)
        result = tool.run("input")
        assert "Error" in result


class TestReActAgent:
    def test_run_completes(self) -> None:
        registry = ToolRegistry()
        registry.register(Tool(name="calc", description="Math", execute=safe_eval))
        agent = ReActAgent(tools=registry, max_steps=5)
        trace = agent.run("Calculate 2+2")
        assert trace.total_steps > 0
        assert trace.task == "Calculate 2+2"

    def test_custom_reasoning(self) -> None:
        registry = ToolRegistry()
        registry.register(Tool(name="calc", description="Math", execute=safe_eval))

        call_count = 0

        def custom_reason(task: str, history: list[Step]) -> ThoughtAction:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ThoughtAction(thought="Let me calculate", action="calc", action_input="3*7")
            return ThoughtAction(thought="Done", is_final=True, final_answer="21")

        agent = ReActAgent(tools=registry, reasoning_fn=custom_reason)
        trace = agent.run("What is 3*7?")
        assert trace.success
        assert trace.final_answer == "21"
        assert len(trace.steps) == 2

    def test_max_steps_reached(self) -> None:
        registry = ToolRegistry()

        def never_finish(task: str, history: list[Step]) -> ThoughtAction:
            return ThoughtAction(thought="Still thinking...", action=None)

        agent = ReActAgent(tools=registry, reasoning_fn=never_finish, max_steps=3)
        trace = agent.run("Impossible task")
        assert not trace.success
        assert trace.total_steps == 3

    def test_unknown_tool_in_action(self) -> None:
        registry = ToolRegistry()

        step = 0

        def reason_with_bad_tool(task: str, history: list[Step]) -> ThoughtAction:
            nonlocal step
            step += 1
            if step == 1:
                return ThoughtAction(thought="Try bad tool", action="nonexistent", action_input="x")
            return ThoughtAction(thought="Done", is_final=True, final_answer="handled")

        agent = ReActAgent(tools=registry, reasoning_fn=reason_with_bad_tool)
        trace = agent.run("test")
        assert "Unknown tool" in trace.steps[0].observation


class TestUtils:
    def test_parse_thought_action(self) -> None:
        text = "Thought: I need to calculate\nAction: calc\nAction Input: 2+2"
        result = parse_thought_action(text)
        assert result["thought"] == "I need to calculate"
        assert result["action"] == "calc"
        assert result["action_input"] == "2+2"

    def test_safe_eval(self) -> None:
        assert safe_eval("2 + 3") == "5"
        assert safe_eval("10 * 5") == "50"
        assert "Error" in safe_eval("import os")
