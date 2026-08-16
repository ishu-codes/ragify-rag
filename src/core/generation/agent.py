from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.ragify.generation.llm import llm
from src.ragify.generation.prompts import prompts


class Agent:
    def __init__(self, tools=None):
        self.tools = tools or []
        self._prompt = prompts.system_prompt

    def create(self):
        agent = create_agent(llm.client, tools=self.tools, system_prompt=self._prompt)
        return agent

def get_agent(workspace_id: str | None = None):
    from src.ragify.retrieval import get_retriever

    tools = [get_retriever(workspace_id)]
    return Agent(tools=tools).create()
