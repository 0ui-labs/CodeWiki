"""Demo: Using CodeWikiModel with pydantic_ai Agents.

This script demonstrates how to use the custom CodeWikiModel wrapper
with pydantic_ai Agents. It shows:

1. Basic Agent creation with CodeWikiModel
2. Simple tool usage (currently limited - see notes)
3. Integration with retry/fallback logic

NOTE: Tool calls from the LLM are not yet supported because ResilientLLMClient
only returns text content. The Agent can define tools, but the LLM responses
won't contain tool calls. To enable tool calls, we need to:
1. Extend LLMClient._call_anthropic and _call_openai to parse tool_use blocks
2. Update LLMResponse to include optional tool_calls field
3. Update CodeWikiModel.request() to convert tool calls to ToolCallPart objects

Example usage:
    # Set environment variables
    export CODEWIKI_ANTHROPIC_API_KEY=sk-ant-...

    # Run demo
    python -m codewiki.src.be.llm_adapter_demo
"""

import asyncio
from pydantic_ai import Agent

from codewiki.core import Settings, get_logger
from codewiki.src.be.llm_adapter import CodeWikiModel


async def demo_basic_agent():
    """Demonstrate basic Agent usage with CodeWikiModel."""
    print("=" * 80)
    print("Demo 1: Basic Agent without tools")
    print("=" * 80)

    # Initialize settings and logger
    settings = Settings()
    logger = get_logger(settings)

    # Create CodeWikiModel
    model = CodeWikiModel(settings, logger)

    # Create simple Agent
    agent = Agent(
        model,
        system_prompt="You are a helpful coding assistant.",
    )

    # Run agent
    print("\nPrompt: What is Python?")
    result = await agent.run("What is Python?")
    print(f"\nResponse: {result.data}")
    print(f"Usage: {result.usage()}")

    logger.close()


async def demo_agent_with_fallback():
    """Demonstrate Agent with fallback model support."""
    print("\n" + "=" * 80)
    print("Demo 2: Agent with fallback models")
    print("=" * 80)

    # Configure with fallback
    settings = Settings(
        main_model="claude-sonnet-4-20250514",
        fallback_models=["gpt-4o"],
        retry_attempts=2,
    )
    logger = get_logger(settings)

    # Create model with fallback
    model = CodeWikiModel(settings, logger)

    # Create Agent
    agent = Agent(
        model,
        system_prompt="You are a code reviewer.",
    )

    # Run agent
    print("\nPrompt: Review this code: def add(a, b): return a + b")
    result = await agent.run("Review this code: def add(a, b): return a + b")
    print(f"\nResponse: {result.data}")

    logger.close()


async def demo_multi_turn_conversation():
    """Demonstrate multi-turn conversation."""
    print("\n" + "=" * 80)
    print("Demo 3: Multi-turn conversation")
    print("=" * 80)

    settings = Settings()
    logger = get_logger(settings)
    model = CodeWikiModel(settings, logger)

    # Create Agent
    agent = Agent(
        model,
        system_prompt="You are a Python tutor.",
    )

    # First message
    print("\nTurn 1: What is a list comprehension?")
    result1 = await agent.run("What is a list comprehension?")
    print(f"Response: {result1.data[:100]}...")

    # Continue conversation
    print("\nTurn 2: Give me an example")
    result2 = await agent.run("Give me an example", message_history=result1.all_messages())
    print(f"Response: {result2.data[:100]}...")

    logger.close()


def demo_note_about_tools():
    """Display note about tool support."""
    print("\n" + "=" * 80)
    print("Note: Tool Call Support")
    print("=" * 80)
    print("""
CodeWikiModel currently does NOT support tool calls from the LLM because
ResilientLLMClient only returns text content.

You can define tools on the Agent, but the LLM won't call them. Example:

    from pydantic_ai import RunContext

    @agent.tool
    def get_user_info(ctx: RunContext[str], user_id: str) -> str:
        return f"User {user_id} info"

    # This won't trigger tool calls from the LLM yet

To enable tool support:
1. Extend LLMClient to parse tool_use blocks from provider responses
2. Update LLMResponse to include tool_calls: list[ToolCall]
3. Update CodeWikiModel.request() to convert to ToolCallPart objects

See codewiki/src/be/llm_adapter.py for implementation details.
    """)


async def main():
    """Run all demos."""
    print("CodeWikiModel Demo")
    print("=" * 80)

    try:
        await demo_basic_agent()
        await demo_agent_with_fallback()
        await demo_multi_turn_conversation()
        demo_note_about_tools()

        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to set CODEWIKI_ANTHROPIC_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())
