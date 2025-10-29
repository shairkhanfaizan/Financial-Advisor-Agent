import streamlit as st
import time
import os  
from langchain.agents import initialize_agent, AgentType
from agents import financial_advisor_agent
from tools import tools

st.set_page_config(
    page_title="Smart Financial Advisor",
    page_icon="💰",
    layout="centered"
)

# Header 
st.title("💸 Smart Financial Advisor")
st.markdown("Upload your **transaction statement (PDF)** and enter your **financial goal** to get 7 personalized recommendations powered by Gemini RAG.")

#  Input Section 
uploaded_pdf = st.file_uploader("📄 Upload your transaction statement (PDF)", type=["pdf"])
financial_goal = st.text_input("🎯 Enter your financial goal (e.g., Save for a car, reduce unnecessary spending, build emergency fund, etc.)")

#  Analyze Button 
if st.button("Generate Recommendations"):
    if uploaded_pdf is None:
        st.warning("Please upload your transaction PDF first.")
    elif not financial_goal.strip():
        st.warning("Please enter a financial goal.")
    else:
        # Ensure 'uploads' folder exists before saving
        os.makedirs("uploads", exist_ok=True)

        pdf_path = f"uploads/{uploaded_pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.info("Preparing your personalized financial insights...")

        #  Progress Animation
        progress_text = "🔍 Analyzing your transactions..."
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.2)  # Simulate processing progress
            progress_bar.progress(percent, text=progress_text)

        # ---- Spinner while the model runs ----
        with st.spinner("⚙️ Generating AI recommendations..."):
            try:
                response = financial_advisor_agent(pdf_path, financial_goal)

                st.success("✅ Personalized recommendations ready!")
                st.markdown("---")
                st.markdown("### 💼 Your Financial Plan:")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ---- Footer ----
st.markdown("---")
st.caption("💡 Powered by Streamlit + LangChain + Gemini RAG")

