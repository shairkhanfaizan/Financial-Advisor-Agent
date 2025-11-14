# agents.py
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub

from tools import fraud_detection_tool, analyze_transactions  #tools.py file


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.6,
    max_output_tokens=2048
)



# Load Agent Prompt Template

prompt = hub.pull("hwchase17/openai-tools-agent")


# Create the Tool-Calling Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[analyze_transactions],
    prompt=prompt
)


# Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[analyze_transactions],
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



# Fraud Detection Agent

fraud_agent = create_tool_calling_agent(
    llm=llm,
    tools=[fraud_detection_tool],  # only fraud tool
    prompt=prompt
)

fraud_executor = AgentExecutor(
    agent=fraud_agent,
    tools=[fraud_detection_tool],
    verbose=True,
)

def fraud_detection_agent(pdf_file: str):
    """
    Executes the fraud detection agent.

    This agent analyzes a user's financial transaction statement to identify
    potentially fraudulent or anomalous activity using AI-based anomaly detection. 
    It leverages transaction embeddings and the Isolation Forest algorithm to flag
    transactions that deviate significantly from normal spending behavior.

    Args:
        pdf_file (str): Path to the user's transaction statement PDF.

    Returns:
        str: A summary report listing all suspicious or potentially fraudulent
            transactions with short explanations for each.
    """
    query = f"Detect potentially fraudulent or anomalous transactions in {pdf_file}."
    response = fraud_executor.invoke({"input": query})
    return response["output"]