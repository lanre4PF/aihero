import search_tools
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
if api_key is None:
    raise ValueError("API_KEY environment variable not set")
 
provider = GoogleProvider(api_key=api_key)
google_model = GoogleModel("gemini-2.5-flash", provider=provider)


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about documentation.

Use the search tool to find relevant information from the repository files before answering questions.

Files are stored in a flat folder structure, where:
- Slashes (/) from the original repository path are replaced with hyphens (-).
- Each file is prefixed with the repository name.

For example:
- Original path: docs/tutorials/intro.md
- Saved as: ultralytics-docs-tutorials-intro.md

When citing sources:
- Always use the GitHub link format:
  [LINK TITLE](https://github.com/{repo_owner}/{repo_name}/blob/main/{relative_path})

- The {relative_path} is reconstructed by replacing hyphens (-) in the saved filename back into slashes (/), 
  and removing the repository prefix.

Example:
Saved file: ultralytics-docs-tutorials-intro.md
Citation: [Introduction Guide](https://github.com/ultralytics/ultralytics/blob/main/docs/tutorials/intro.md)

If the search doesn't return relevant results, let the user know and provide general guidance.
"""


def init_agent(index,repo_owner, repo_name,vindex = None):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)
    if vindex:
        search_tool = search_tools.SearchTool(index,vindex)
        agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.hybrid_search],
        model= google_model
        )
        return agent
    else:
        search_tool = search_tools.SearchTool(index,None)
        agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.text_search],
        model= google_model
        )
        return agent
