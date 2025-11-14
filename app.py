import streamlit as st
import time
import os  
from all_agents import financial_advisor_agent, fraud_detection_agent
from tools import tools

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="SynFi AI",
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
st.title("💸 SynFi AI - “Synchronized Financial Intelligence.”")

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

st.markdown(
    "<p class='subtext'>Get AI-powered insights powered by <b>Gemini RAG</b> and LangChain Intelligence.</p>",
    unsafe_allow_html=True
)

# ---------------------- MODE SELECTOR ----------------------
mode = st.radio(
    "🧠 Select Analysis Mode:",
    ["Financial Advisor", "Fraud Detection"],
    horizontal=True
)

# ---------------------- DYNAMIC INPUTS ----------------------
financial_goal = None
if mode == "Financial Advisor":
    financial_goal = st.text_input(
        "🎯 Enter your Financial Goal",
        placeholder="e.g., Save for a car, reduce unnecessary spending, or build an emergency fund",
    )
elif mode == "Fraud Detection":
    st.info("🚨 Detect suspicious or anomalous transactions in your uploaded statement.")

# ---------------------- ACTION BUTTON ----------------------
generate = st.button(f"🚀 Run {mode} Analysis", use_container_width=True)

if generate:
    if uploaded_pdf is None:
        st.warning("Please upload your transaction PDF first (from sidebar).")
    elif mode == "Financial Advisor" and not financial_goal.strip():
        st.warning("Please enter a financial goal.")
    else:
        os.makedirs("uploads", exist_ok=True)
        pdf_path = f"uploads/{uploaded_pdf.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.info(f"📊 Running {mode} analysis...")

        # Progress bar simulation
        progress_text = f"🔍 Processing {mode.lower()}..."
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.15)
            progress_bar.progress(percent, text=progress_text)
            
            
        #creating session state for response    
        if "response" not in st.session_state:
            st.session_state.response = ""
        
        # Spinner during AI generation
        with st.spinner(f"⚙️ {mode} in progress..."):
            try:
                # --- Mode Logic (no layout change) ---
                if mode == "Financial Advisor":
                    result = financial_advisor_agent(pdf_path, financial_goal)
                    st.session_state.response = result
                    section_title = "💼 Your Financial Plan"
                elif mode == "Fraud Detection":
                    result = fraud_detection_agent(pdf_path)
                    st.session_state.response = result
                    section_title = "🚨 Fraud Analysis Report"

                # --- Display Result (same UI as before) ---
                st.success(f"✅ {mode} completed successfully!")
                st.markdown(f"### {section_title}")

                theme_base = st.get_option("theme.base")
                if theme_base == "light":
                    box_bg = "#f9fafb"
                    text_color = "#1e293b"
                else:
                    box_bg = "#1e293b"
                    text_color = "#f8fafc"
                
                if st.session_state.response:
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
                            {st.session_state.response}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- Download Option (fixed to support both modes) ---
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.response,
                    file_name=f"{mode.replace(' ', '_').lower()}_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("💡 Powered by Streamlit + LangChain + Gemini RAG | Built with ❤️ by Faizan")
