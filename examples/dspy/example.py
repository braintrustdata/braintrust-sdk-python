#!/usr/bin/env python
"""DSPy ReAct agent traced via braintrust.auto_instrument().

Run with: OPENAI_API_KEY=<key> BRAINTRUST_API_KEY=<key> uv run python example.py
"""

import braintrust


# auto_instrument() patches LiteLLM (which DSPy uses internally) and DSPy's
# `configure()` so the Braintrust callback is attached automatically. Call it
# BEFORE importing DSPy so the patching applies to the imported module.
braintrust.auto_instrument()
braintrust.init_logger(project="dspy-example")

import dspy


def main():
    print("Braintrust logging enabled - view traces at https://braintrust.dev")

    if hasattr(dspy, "configure_cache"):
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)  # pylint: disable=no-member

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    def calculator(expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression, {"__builtins__": {}}, {})

    def get_current_year() -> int:
        """Get the current year."""
        return 2025

    react = dspy.ReAct("question -> answer", tools=[calculator, get_current_year])

    question = "If I was born in 1990, how old will I be in the current year?"
    result = react(question=question)

    print(f"Q: {question}")
    print(f"A: {result.answer}")


if __name__ == "__main__":
    main()
