# agents.py
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub

from tools import tools  #tools.py file


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.6,
    max_output_tokens=2048
)



# Load Agent Prompt Template

prompt = hub.pull("hwchase17/openai-tools-agent")


# Create the Tool-Calling Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Agent Runner Function
def financial_advisor_agent(pdf_file: str, financial_goal: str):
    """
    Executes the financial analysis agent.

    Args:
        pdf_file (str): Path to the user's transaction PDF.
        financial_goal (str): The user's financial goal.

    Returns:
        str: The agent's financial recommendations.
    """
    query = f"Analyze my transactions in {pdf_file} and generate 7 recommendations based on my goal: {financial_goal}"
    response = agent_executor.invoke({"input": query})
    return response["output"]
