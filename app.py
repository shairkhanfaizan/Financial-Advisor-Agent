import streamlit as st
import time
import os  
from langchain.agents import initialize_agent, AgentType
from agents import financial_advisor_agent
from tools import tools

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Smart Financial Advisor",
    page_icon="💰",
    layout="wide"
)

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("📂 Uploaded Documents")
    uploaded_pdf = st.file_uploader("Upload your Transaction PDF", type=["pdf"])
    
    if uploaded_pdf:
        st.success(f"✅ {uploaded_pdf.name} uploaded successfully")
        st.caption("File stored temporarily (auto-deleted after use).")
    else:
        st.info("📄 Please upload a transaction statement to begin.")
    
    st.markdown("---")
    st.markdown("💡 **Note:** Uploaded files are not stored permanently — they’re only used for analysis.")

# ---------------------- MAIN UI ----------------------
st.title("💸 Smart Financial Advisor")

st.markdown(
    """
    <style>
        .subtext {color: #636e72; font-size: 1rem;}
        .recommendation-box {
            background-color: #f9f9f9; 
            border-radius: 12px; 
            padding: 1.5rem; 
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='subtext'>Get 7 personalized recommendations powered by <b>Gemini RAG</b> and LangChain Intelligence.</p>", unsafe_allow_html=True)

# ---------------------- FINANCIAL GOAL INPUT ----------------------
financial_goal = st.text_input(
    "🎯 Enter your Financial Goal",
    placeholder="e.g., Save for a car, reduce unnecessary spending, or build an emergency fund",
)

# ---------------------- ACTION BUTTON ----------------------
generate = st.button("🚀 Generate Recommendations", use_container_width=True)

if generate:
    if uploaded_pdf is None:
        st.warning("Please upload your transaction PDF first (from sidebar).")
    elif not financial_goal.strip():
        st.warning("Please enter a financial goal.")
    else:
        os.makedirs("uploads", exist_ok=True)
        pdf_path = f"uploads/{uploaded_pdf.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.info("📊 Preparing your personalized financial insights...")

        # Progress bar simulation
        progress_text = "🔍 Analyzing your transactions..."
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.15)
            progress_bar.progress(percent, text=progress_text)

        # Spinner during AI generation
        with st.spinner("⚙️ Generating AI recommendations..."):
            try:
                response = financial_advisor_agent(pdf_path, financial_goal)
                st.success("✅ Personalized recommendations ready!")
                st.markdown("### 💼 Your Financial Plan")

                # Detect theme and set colors dynamically
                theme_base = st.get_option("theme.base")

                if theme_base == "light":
                    box_bg = "#f9fafb"      # soft light gray
                    text_color = "#1e293b"  # deep slate gray
                else:
                    box_bg = "#1e293b"      # elegant dark slate
                    text_color = "#f8fafc"  # warm off-white

                st.markdown(
                    f"""
                    <div class='recommendation-box' 
                        style='background-color:{box_bg};
                                color:{text_color};
                                padding:1.5rem;
                                border-radius:12px;
                                box-shadow:0px 2px 8px rgba(0,0,0,0.08);
                                line-height:1.6;
                                font-size:1rem;'>
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("💡 Powered by Streamlit + LangChain + Gemini RAG | Built with ❤️ by Faiz")
