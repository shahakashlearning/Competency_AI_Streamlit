import streamlit as st
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="AI Competency Intelligence Platform", layout="wide")

st.title("🚀 AI Competency Intelligence Platform")

# =========================================================
# LOAD GROQ SECRET
# =========================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=st.secrets["GROQ_API_KEY"]
)

# =========================================================
# SIDEBAR MENU
# =========================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "👤 Self Competency Check",
        "👥 Team Competency Check",
        "🤖 AI Chat Assistant (RAG)"
    ]
)

# =========================================================
# 👤 SELF COMPETENCY CHECK
# =========================================================

if menu == "👤 Self Competency Check":

    st.header("👤 Self Competency Check")

    uploaded_file = st.file_uploader("Upload Individual Competency Sheet")

    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=6)
        df.columns = df.columns.str.strip()

        st.success("File Loaded Successfully")

        area_col = "Area"
        target_col = "Target"
        target_index = df.columns.get_loc("Target")
        current_col = df.columns[target_index + 1]

        grouped = df.groupby(area_col).mean(numeric_only=True)

        st.subheader("Competency Overview")
        st.dataframe(grouped[[target_col, current_col]])

        total_target = grouped[target_col].sum()
        total_current = grouped[current_col].sum()
        percentage = (total_current / total_target) * 100 if total_target else 0

        st.metric("Overall Competency Score (%)", f"{percentage:.2f}%")

# =========================================================
# 👥 TEAM COMPETENCY CHECK
# =========================================================

elif menu == "👥 Team Competency Check":

    st.header("👥 Team Competency Check")

    uploaded_file = st.file_uploader("Upload Team Competency Sheet")

    if uploaded_file:
        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team File Loaded")

        area_col = "Area"
        target_col = "Target"

        target_index = team_df.columns.get_loc("Target")
        employee_cols = team_df.columns[target_index + 1:]

        # Clean unwanted columns
        employee_cols = [
            col for col in employee_cols
            if "average" not in col.lower()
            and "maximum" not in col.lower()
            and "unnamed" not in col.lower()
        ]

        tab1, tab2 = st.tabs(["📊 Team Average", "👤 Individual Members"])

        # ----------------------------
        # TEAM AVERAGE DASHBOARD
        # ----------------------------

        with tab1:

            st.subheader("Team Average Dashboard")

            team_df["Team Average"] = team_df[employee_cols].mean(axis=1)

            grouped = team_df.groupby(area_col).mean(numeric_only=True)

            st.dataframe(grouped[[target_col, "Team Average"]])

        # ----------------------------
        # INDIVIDUAL MEMBER DASHBOARD
        # ----------------------------

        with tab2:

            selected_emp = st.selectbox("Select Team Member", employee_cols)

            grouped = team_df.groupby(area_col).mean(numeric_only=True)

            st.dataframe(grouped[[target_col, selected_emp]])

# =========================================================
# 🤖 RAG CHAT ASSISTANT
# =========================================================

elif menu == "🤖 AI Chat Assistant (RAG)":

    st.header("🤖 RAG-Based AI Competency Assistant")

    uploaded_file = st.file_uploader("Upload Team Competency Sheet for AI")

    if uploaded_file:

        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team Data Loaded for AI")

        user_question = st.text_input(
            "Ask something like: Who has highest Software Development skill?"
        )

        if st.button("Ask AI") and user_question:

            # =====================================================
            # 🔎 SIMPLE RETRIEVAL LOGIC (Prevents Token Overflow)
            # =====================================================

            filtered_df = team_df.copy()

            keywords = user_question.lower().split()

            for word in keywords:
                filtered_df = filtered_df[
                    filtered_df.apply(
                        lambda row: row.astype(str).str.lower().str.contains(word).any(),
                        axis=1
                    )
                ]

            if filtered_df.empty:
                filtered_df = team_df.head(10)

            # Limit rows (VERY IMPORTANT)
            filtered_df = filtered_df.head(15)

            context_text = filtered_df.to_string(index=False)

            with st.spinner("AI analyzing team data..."):

                prompt = f"""
                You are an AI Competency Analytics Assistant.

                Use ONLY the data below to answer.

                TEAM DATA:
                {context_text}

                QUESTION:
                {user_question}

                Provide clear structured answer.
                """

                try:
                    response = llm.invoke(prompt)
                    st.success("AI Response:")
                    st.write(response.content)

                except Exception as e:
                    st.error("AI Request Failed")
                    st.write(str(e))
