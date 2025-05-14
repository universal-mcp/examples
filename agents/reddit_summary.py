# Import necessary libraries
import os  # For environment variable access
from typing import Annotated  # For type annotations

from langchain_openai import ChatOpenAI  # OpenAI's chat model interface
from langgraph.prebuilt import (
    create_react_agent,
)  # Pre-built ReAct agent implementation
from loguru import logger  # For logging
from universal_mcp.tools.adapters import ToolFormat  # Tool format definitions
from universal_mcp.tools.manager import ToolManager  # Tool management system
from universal_mcp.applications import app_from_slug
from universal_mcp.integrations import AgentRIntegration


def get_application(name: str):
    app = app_from_slug(name)
    integration = AgentRIntegration(
        name
    )
    instance  = app(integration)
    return instance

async def main():
    # Initialize the OpenAI model - defaults to "gpt-4o-mini" if not specified
    model = os.environ.get("OPEN_AI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model)

    reddit_app = get_application("reddit")
    email_app = get_application("google-mail")


    # Create a tool manager and register our calculator tool
    tool_manager = ToolManager()
    tool_manager.register_tools_from_app(reddit_app)
    tool_manager.register_tools_from_app(email_app)

    # Get the list of tools in LangChain format
    tools = tool_manager.list_tools(format=ToolFormat.LANGCHAIN)

    # Create a ReAct agent with the specified tools and system prompt
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt="You are a helpful assistant that can use tools to help the user.",
    )

    agent_prompt = """
    You are a helpful assistant. For the given subreddit, fetch the latest posts, organize them by category (such as News, Discussion, Question, Meme, etc.), and provide a concise summary for each category. Format the output as a well-structured Markdown document, including:

    - A main heading with the subreddit name
    - Subheadings for each category
    - Bullet points or short paragraphs summarizing the key posts in each category
    - For each post, include the title (as a Markdown link to the post), author, and a brief summary

    After summarizing, share the Markdown-formatted summary.

    Send the summary to the email address: {send_to_email}
    Subreddit: {subreddit}
    """

    subreddit = input("Enter the subreddit: ")
    send_to_email = input("Enter the email to send the summary to: ")
    print("Welcome to the agent!")
    result = await agent.ainvoke(
        input={"messages": [{"role": "user", "content": agent_prompt.format(subreddit=subreddit, send_to_email=send_to_email)}]}
    )
    print(result["messages"][-1].content)

    

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
