# ReActLib Architecture

## Overview

ReActLib implements the Reason+Act loop pattern where agents alternate between thinking and acting.

## Components

- **ReActAgent** - Main orchestrator running the think-act-observe loop
- **Tool** - Executable capability with name, description, and execute function
- **ToolRegistry** - Collection of available tools
- **Trace** - Full execution history for debugging

## Data Flow

```
Task -> Agent.run() -> [Think -> Act -> Observe]* -> Final Answer
                            |        |        |
                        Reasoning  ToolRegistry  Tool.run()
```

## Design Principles

1. **Pluggable reasoning** - Swap reasoning functions (heuristic, LLM, rule-based)
2. **Tool extensibility** - Register any callable as a tool
3. **Full observability** - Every step is traced and inspectable
