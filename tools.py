# tools.py
import os
from dotenv import load_dotenv

import numpy as np

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.ensemble import IsolationForest 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
import re
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



# transaction analysis tool

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
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
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
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.6)

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




# --------------------------
# LINE-BASED PDF TEXT PARSER
# --------------------------
def extract_transactions(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    lines = []
    for page in pages:
        for ln in page.page_content.split("\n"):
            ln = ln.strip()
            if ln:
                lines.append(ln)

    lines = [ln for ln in lines if not ln.lower().startswith("date")]

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})\s+(.*?)\s+([A-Za-z]+)\s+([\d,]+)\s+(Debit|Credit)"
    )

    data = []
    for ln in lines:
        m = pattern.search(ln)
        if m:
            date, desc, cat, amt, ttype = m.groups()
            amt = float(amt.replace(",", ""))
            data.append([date, desc, cat, amt, ttype])

    df = pd.DataFrame(data, columns=["date", "description", "category", "amount", "type"])
    return df


# --------------------------
# EXTENDED FRAUD DETECTION
# --------------------------
def detect_fraud(df):

    results = []

    # 1. Numeric anomaly with LOF
    df["abs_amount"] = df["amount"].abs()

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.18  # more sensitive
    )
    df["lof_flag"] = lof.fit_predict(df[["abs_amount"]])
    df["lof_score"] = -lof.negative_outlier_factor_

    for _, row in df[df["lof_flag"] == -1].iterrows():
        results.append((row, "Unusually high amount compared to normal spending"))

    # 2. High-risk merchants
    high_risk_keywords = ["crypto", "bitworld", "luxury", "watch", "overseas", "intl", "hk"]
    for _, row in df.iterrows():
        desc = row["description"].lower()
        if any(k in desc for k in high_risk_keywords):
            results.append((row, "High-risk merchant or international vendor"))

    # 3. Unknown merchants (POS terminals, unknown city)
    suspicious_patterns = ["unknown", "pos terminal", "unknown city"]
    for _, row in df.iterrows():
        desc = row["description"].lower()
        if any(k in desc for k in suspicious_patterns):
            results.append((row, "Suspicious/unknown merchant or location"))

    # 4. Large UPI or transfer amounts
    if "upi" in df["description"].str.lower().to_string().lower():
        for _, row in df[df["amount"] >= 12000].iterrows():
            if "upi" in row["description"].lower():
                results.append((row, "Unusually large UPI transfer"))

    # Remove duplicates
    seen = set()
    cleaned = []
    for row, reason in results:
        key = (row["date"], row["description"], row["amount"])
        if key not in seen:
            cleaned.append((row, reason))
            seen.add(key)

    return cleaned


# --------------------------
# LANGCHAIN TOOL
# --------------------------
@tool
def fraud_detection_tool(pdf_path: str) -> str:
    """
    Detects fraudulent or anomalous transactions using:
    - LOF anomaly scoring
    - High-risk merchant detection
    - Suspicious keyword matching
    - International transaction detection
    - Large transfer detection
    """

    df = extract_transactions(pdf_path)
    if df.empty:
        return "❌ No transactions found in the PDF."

    flagged = detect_fraud(df)
    if not flagged:
        return "✅ No suspicious transactions detected."

    report = "🚨 Suspicious Transactions Detected:\n\n"
    for row, reason in flagged:
        report += f"{row['date']}: {row['description']} — ₹{row['amount']}  ({reason})\n"

    return report



# # fraud detection tool

# @tool
# def detect_fraud(pdf_file: str) -> str:
#     """
#     Detects potentially fraudulent or anomalous transactions from a PDF file using Isolation Forest on
#     semantic embeddings of transaction descriptions.


#     Args:
#         pdf_file (str): Path to the user's transaction PDF.

#     Returns:
#         str: A report highlighting suspicious or unusual transactions.
#     """

#     # --- Step 1: Parse the PDF into text ---
#     try:
#         loader = PyPDFLoader(pdf_file)
#         pages = loader.load()
#         transactions = []
#         for page in pages:
#             lines = [line.strip() for line in page.page_content.split("\n") if line.strip()]
#             transactions.extend(lines)
#             transactions = [
#                 line for line in transactions 
#                 if not line.lower().startswith(("date", "sample", "report"))
#             ]
#     except Exception as e:
#         return f"❌ Error parsing PDF: {e}"

#     if not transactions:
#         return "No transactions found in the uploaded PDF."

#     # --- Step 2: Convert text into embeddings ---
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
#         embeddings = embeddings.embed_documents(transactions)
#         embeddings = np.array(embeddings)
#     except Exception as e:
#         return f"❌ Error generating embeddings: {e}"

#     # --- Step 3: Detect anomalies ---
#     try:
#         model = IsolationForest(contamination=0.02, random_state=42)
#         preds = model.fit_predict(embeddings)
#         anomalies = np.where(preds == -1)[0]

#         if len(anomalies) == 0:
#             return "✅ No suspicious transactions detected."
        
#         suspicious = [transactions[i] for i in anomalies[:10]]  # limit to 10
#         report = "🚨 Suspicious Transactions Detected:\n\n"
#         for t in suspicious:
#             report += f"- {t}\n"
#         return report

#     except Exception as e:
#         return f"❌ Error detecting anomalies: {e}"


# Tool registry for agents
tools = [analyze_transactions, fraud_detection_tool]

