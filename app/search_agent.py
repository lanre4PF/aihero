import search_tools
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv
import os

# load_dotenv()
# api_key = os.getenv('API_KEY')
# if api_key is None:
#     raise ValueError("API_KEY environment variable not set")
 
provider = GoogleProvider(api_key=api_key)
google_model = GoogleModel("gemini-2.5-flash", provider=provider)


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about documentation.

Use the search tool to find relevant information from the repository files before answering.

Files are stored in a flat folder where each saved filename:
- Is prefixed with the repository name and a hyphen (e.g., "ultralytics-").
- Replaces original folder separators (/) with hyphens (-).

To build a GitHub link from a saved filename, follow these steps exactly:
1. If the saved filename begins with "{repo_name}-", remove that exact leading prefix.
2. Extract and keep the file extension (".md" or ".mdx").
3. Remove the extension to get the base name, then split the base on hyphens (-) into tokens.
4. Reconstruct the repo path by joining those tokens with slashes (/) and append the original extension.
5. Format the citation as:
   [LINK TITLE](https://github.com/{repo_owner}/{repo_name}/blob/main/<reconstructed-path>)

Example:
Saved file: ultralytics-docs-en-modes-train.md
1. remove prefix -> docs-en-modes-train.md
2. extension -> .md
3. base -> docs-en-modes-train -> tokens: ["docs","en","modes","train"]
4. reconstructed path -> docs/en/modes/train.md
Citation -> [Title](https://github.com/{repo_owner}/{repo_name}/blob/main/docs/en/modes/train.md)

Always include references by citing the filename of the source material you used.

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
