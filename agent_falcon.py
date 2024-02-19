#!pip install langchain
#!pip install google-search-results

from langchain import HuggingFaceHub
repo_id = "tiiuae/falcon-7b-instruct"
huggingfacehub_api_token = "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" #Use your own API key
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.1, "max_new_tokens":1500})
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os

tools_name = ["serpapi"]
os.environ["SERPAPI_API_KEY"] = "021FBD1094DB407285855E75B2F5CDB1" #Use your own API key
tools = load_tools(tools_name)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What is the current temperature in Los Angles?")

