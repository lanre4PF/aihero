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
You are a helpful AI assistant tasked with answering questions based on the Ultralytics YOLO documentation

Use the search tool to find relevant information from the course materials before answering questions.  

If you can find specific information through search, use it to provide accurate answers.

Always include references by citing the filename of the source material you used.  
When citing the reference, replace "ultralytics-main" by the full path to the GitHub repository: "https://github.com/ultralytics/ultralytics/blob/main/"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance. 

""".strip()

def init_agent(index, repo_owner, repo_name):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    search_tool = search_tools.SearchTool(index=index)

    agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
        model= google_model
    )

    return agent
