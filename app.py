import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain_groq import ChatGroq

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="SkillMatrix AI", layout="wide")

st.title("🚀 SkillMatrix AI – Competency Analytics Platform")

# ===============================
# LOAD GROQ KEY
# ===============================
groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=groq_api_key
)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload Competency Sheet",
    type=["xlsx", "xlsm"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file, engine="openpyxl", header=6)
    df.columns = df.columns.str.strip()

    area_col = "Area"
    target_col = "Target"
    target_index = df.columns.get_loc("Target")
    current_col = df.columns[target_index + 1]

    st.success(f"Employee Detected: {current_col}")

    # Convert numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[current_col] = pd.to_numeric(df[current_col], errors="coerce")

    # ===============================
    # RADAR CHART
    # ===============================
    st.subheader("Competency Radar Dashboard")

    grouped = df.groupby(area_col).mean(numeric_only=True).reset_index()

    areas = grouped[area_col].tolist()
    target_values = grouped[target_col].tolist()
    current_values = grouped[current_col].tolist()

    radar_df = pd.DataFrame({
        "Area": areas,
        "Target": target_values,
        "Current": current_values
    })

    st.dataframe(radar_df)

    # ===============================
    # GAP ANALYSIS
    # ===============================
    st.subheader("Skill Gap Analysis")

    gap_df = df[df[current_col] < df[target_col]]

    for _, row in gap_df.iterrows():

        skill = row["Competence"]
        current = row[current_col]
        target = row[target_col]

        st.markdown(f"### {skill}")
        st.write(f"Current: {current} | Target: {target}")

        if st.button(f"AI Suggest Improvement for {skill}"):

            prompt = f"""
            Skill: {skill}
            Current Level: {current}
            Target Level: {target}
            Suggest improvement steps.
            """

            response = llm.invoke(prompt)
            st.write(response.content)
