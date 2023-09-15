from langchain.llms import Clarifai
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import AIPluginTool


USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'GPT-4' 
MODEL_VERSION_ID = 'ad16eda6ac054796bf9f348ab6733c72'

tool = AIPluginTool.from_plugin_url("http://localhost:8000/.well-known/ai-plugin.json")


clarifai_llm = Clarifai(user_id=USER_ID,app_id=APP_ID, model_id=MODEL_ID, model_version_id=MODEL_VERSION_ID)

# Used to load the plugin
tools = load_tools(["requests_get"])

tools += [tool]

agent_chain = initialize_agent(tools, clarifai_llm, verbose=True, agent_chain=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent_chain.run("Who is leading the relief pitchers in strikeouts?"))