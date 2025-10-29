# tools.py
import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@tool
def analyze_transactions(pdf_file: str, financial_goal: str) -> str:
    """
    Analyzes the user's transaction PDF using Gemini RAG and generates 7
    personalized financial recommendations based on their financial goal.
    
    Args:
        pdf_file (str): Path to the uploaded transaction statement PDF.
        financial_goal (str): The user's financial goal or objective.
    """
    #  Load transaction PDF as documents
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()  # list[Document]

    # Build embeddings + FAISS vectorstore (in-memory for simplicity)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt template (uses {context} + {financial_goals})
    prompt = PromptTemplate(
        template="""
You are an expert financial advisor. The user provided transaction data and a financial goal.
Using the retrieved transaction context and the user's goal, generate EXACTLY 7
clear, actionable, and personalized recommendations to improve budgeting, spending,
saving, and investing. Keep language motivational and practical.

Financial Goals:
{financial_goals}

Transaction Context:
{context}

Output format (numbered 1..7):
1. ...
2. ...
...
7. ...
""",
        input_variables=["context", "financial_goals"],
    )

    # Load Gemini LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.6)

    # Helper to format docs into a single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #  Building LCEL pipeline:

    parallel_chain = RunnableParallel({
        "context": (
            RunnableLambda(lambda x: retriever.get_relevant_documents(x["financial_goal"]))
            | RunnableLambda(format_docs)
        ),
        "financial_goals": RunnableLambda(lambda x: x["financial_goal"])
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()

    #  Invoke chain with the user's financial goal
    result = main_chain.invoke({"financial_goal": financial_goal})

    #  Return the text result from Gemini (numbered 1..7 list)
    return result


# Tool registry for agents
tools = [analyze_transactions]

