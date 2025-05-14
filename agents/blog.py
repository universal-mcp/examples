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

    blog_app = get_application("hashnode")
    research_app = get_application("perplexity")
    image_app = get_application("falai")


    # Create a tool manager and register our calculator tool
    tool_manager = ToolManager()
    tool_manager.register_tools_from_app(blog_app)
    tool_manager.register_tools_from_app(research_app)
    tool_manager.add_tool(image_app.generate_image, name="generate_image")

    # Get the list of tools in LangChain format
    tools = tool_manager.list_tools(format=ToolFormat.LANGCHAIN)

    # Create a ReAct agent with the specified tools and system prompt
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt="You are a helpful assistant that can use tools to help the user.",
    )
    agent_prompt = """
    Generate a blog post for given topic for agentr in a formal tone. Use perplexity to research for the topic and put up to date content. Once done publish the post to hashnode blog with publication id: {publication_id}

    Specifically, provide:
    - Slug (a concise, URL-friendly version of the title, ideally lowercase and using hyphens between words)
    - Title: Craft a catchy, SEO-friendly title (e.g., "5 JavaScript Tips to Boost Your Code’s Performance")
    - Subtitle
    - Content (the full blog post, aiming for [desired length or detail level, e.g., 500 words, a comprehensive overview, etc.], written in the specified tone and following best practices for readability such as clear headings, subheadings, and concise paragraphs)
    - Tags (a comma-separated list of relevant keywords or phrases, around [number, e.g., 3-5] tags, that categorize the blog post). Always include "blog"

    ## Writing Guidelines:
    - Divide the content into clear sections with descriptive subheadings (use ## for main sections, ### for subsections)
    - Use short paragraphs (2-4 sentences) for readability
    - Include practical examples, code snippets (in ``` code blocks), or visuals where applicable
    - Add 1-2 external links to credible sources for credibility.
    - Conclusion: Summarize key points and include a call-to-action (e.g., “Try these tips and share your results!”).

    Tone: Keep it conversational, professional, and approachable. Length: Aim for 800-1,500 words.

    Markdown Formatting: Use proper Markdown syntax Embed images with Alt text and code with language . Create inline links wherever possible

    Cover Image: Generate a cover image for the blog. Use landscape format

    Topic: {topic}
    """

    publication_id = input("Enter the publication id: ")
    topic = input("Enter the topic for the blog post: ")

    result = await agent.ainvoke(
        input={"messages": [{"role": "user", "content": agent_prompt.format(publication_id=publication_id, topic=topic)}]}
    )
    print(result["messages"][-1].content)

    

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
