import streamlit as st
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq

# ============================
# CONFIG
# ============================

st.set_page_config(page_title="AI Competency Platform", layout="wide")

st.title("🚀 AI Competency Intelligence Platform")

# ============================
# LOAD GROQ FROM SECRETS
# ============================

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=st.secrets["GROQ_API_KEY"]
)

# ============================
# SIDEBAR NAVIGATION
# ============================

menu = st.sidebar.radio(
    "Navigation",
    ["Self Competency Check",
     "Team Competency Check",
     "🤖 AI Chat Assistant (RAG)"]
)

# =========================================================
# SELF CHECK (Prototype)
# =========================================================

if menu == "Self Competency Check":

    st.header("👤 Self Competency Check")

    uploaded_file = st.file_uploader("Upload Individual Competency Sheet")

    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=6)
        st.success("File Loaded Successfully")
        st.dataframe(df.head())

        st.info("Radar Dashboard + Gap Analysis can be added here (already implemented in desktop version).")

# =========================================================
# TEAM CHECK
# =========================================================

elif menu == "Team Competency Check":

    st.header("👥 Team Competency Check")

    uploaded_file = st.file_uploader("Upload Team Competency Sheet")

    if uploaded_file:
        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team File Loaded")

        # Identify employee columns dynamically
        target_index = team_df.columns.get_loc("Target")
        employee_cols = team_df.columns[target_index + 1:]

        clean_employee_cols = [
            col for col in employee_cols
            if "average" not in col.lower()
            and "maximum" not in col.lower()
            and "unnamed" not in col.lower()
        ]

        st.subheader("Team Average Dashboard")

        team_df["Team Average"] = team_df[clean_employee_cols].mean(axis=1)

        grouped = team_df.groupby("Area").mean(numeric_only=True)

        st.dataframe(grouped[["Target", "Team Average"]])

# =========================================================
# 🤖 RAG CHAT ASSISTANT
# =========================================================

elif menu == "🤖 AI Chat Assistant (RAG)":

    st.header("🤖 RAG-Based AI Competency Assistant")

    uploaded_file = st.file_uploader("Upload Team Competency Sheet for AI Analysis")

    if uploaded_file:
        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team Data Loaded for AI")

        # Convert DataFrame into structured text knowledge
        def dataframe_to_text(df):
            text_data = ""

            for _, row in df.iterrows():
                text_data += f"""
                Area: {row.get('Area')}
                Competence: {row.get('Competence')}
                Target: {row.get('Target')}
                """

                for col in df.columns:
                    if col not in ["Area", "Competence", "Target"]:
                        text_data += f"{col}: {row.get(col)}\n"

                text_data += "\n"

            return text_data

        knowledge_base = dataframe_to_text(team_df)

        st.subheader("Ask About Team Skills")

        user_question = st.text_input(
            "Example: Who has highest Software Development skill?"
        )

        if st.button("Ask AI") and user_question:

            with st.spinner("AI is analyzing team data..."):

                prompt = f"""
                You are an AI Competency Analytics Assistant.

                Use ONLY the data provided below to answer accurately.

                ====================
                TEAM DATA:
                ====================
                {knowledge_base}

                ====================
                QUESTION:
                ====================
                {user_question}

                Provide clear, structured answer.
                """

                response = llm.invoke(prompt)

                st.success("AI Response:")
                st.write(response.content)
