# Import necessary libraries
import os  # For environment variable access
from langchain_openai import ChatOpenAI  # OpenAI's chat model interface
from langgraph.prebuilt import (
    create_react_agent,
)  # Pre-built ReAct agent implementation
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

    twitter_app = get_application("twitter")


    # Create a tool manager and register our calculator tool
    tool_manager = ToolManager()
    tool_manager.add_tool(twitter_app.users_id_like, name="like_tweet")
    tool_manager.add_tool(twitter_app.users_id_unlike, name="unlike_tweet")
    tool_manager.add_tool(twitter_app.find_my_user, name="find_my_user_id")

    # Get the list of tools in LangChain format
    tools = tool_manager.list_tools(format=ToolFormat.LANGCHAIN)
    # Create a ReAct agent with the specified tools and system prompt
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt="You are a helpful assistant that can use tools to help the user.",
    )

    agent_prompt = """
    You are a helpful assistant. Your task is to like a specific tweet on behalf of the current user.

    Steps:
    1. Retrieve the current user's Twitter user ID.
    2. Given the tweet URL or tweet ID ({tweet_url}), extract the tweet ID if necessary.
    3. Use the appropriate Twitter API/tool to like the tweet as the current user.
    4. Confirm to the user that the tweet has been liked, or report any errors.

    Only perform the like action; do not reply, retweet, or perform any other operation.
    """

    tweet_url = input("Enter the tweet id / url: ")
    print("Welcome to the agent!")
    result = await agent.ainvoke(
        input={"messages": [{"role": "user", "content": agent_prompt.format(tweet_url=tweet_url)}]}
    )
    print(result["messages"][-1].content)

    

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
