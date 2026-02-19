import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="AI Competency Intelligence Platform", layout="wide")
st.title("🚀 AI Competency Intelligence Platform")

# =========================================================
# LOAD GROQ MODEL
# =========================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=st.secrets["GROQ_API_KEY"]
)

# =========================================================
# LOAD EMBEDDING MODEL (CACHED)
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "👤 Self Competency Check",
        "👥 Team Competency Check",
        "🤖 AI Chat Assistant (Advanced RAG)"
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

        employee_cols = [
            col for col in employee_cols
            if "average" not in col.lower()
            and "maximum" not in col.lower()
            and "unnamed" not in col.lower()
        ]

        tab1, tab2 = st.tabs(["📊 Team Average", "👤 Individual Members"])

        # ----------------------------
        # TEAM AVERAGE
        # ----------------------------
        with tab1:
            st.subheader("Team Average Dashboard")

            team_df["Team Average"] = team_df[employee_cols].mean(axis=1)
            grouped = team_df.groupby(area_col).mean(numeric_only=True)

            st.dataframe(grouped[[target_col, "Team Average"]])

        # ----------------------------
        # INDIVIDUAL MEMBER
        # ----------------------------
        with tab2:
            selected_emp = st.selectbox("Select Team Member", employee_cols)

            grouped = team_df.groupby(area_col).mean(numeric_only=True)
            st.dataframe(grouped[[target_col, selected_emp]])

# =========================================================
# 🤖 ADVANCED RAG CHAT ASSISTANT
# =========================================================

elif menu == "🤖 AI Chat Assistant (Advanced RAG)":

    st.header("🤖 Advanced RAG AI Competency Assistant")

    uploaded_file = st.file_uploader("Upload Team Competency Sheet for AI")

    if uploaded_file:

        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team Data Loaded for AI")

        # ===============================
        # 1️⃣ Convert each row into chunk
        # ===============================

        chunks = []

        for _, row in team_df.iterrows():
            row_text = " | ".join(
                [f"{col}: {row[col]}" for col in team_df.columns if pd.notna(row[col])]
            )
            chunks.append(row_text)

        # ===============================
        # 2️⃣ Create Embeddings + FAISS Index
        # ===============================

        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.info("Vector database created successfully.")

        # ===============================
        # 3️⃣ Ask Question
        # ===============================

        user_question = st.text_input(
            "Ask: Who has highest Software Development skill?"
        )

        if st.button("Ask AI") and user_question:

            with st.spinner("Retrieving relevant data using vector search..."):

                question_embedding = embedding_model.encode([user_question])
                question_embedding = np.array(question_embedding).astype("float32")

                distances, indices = index.search(question_embedding, k=5)

                retrieved_chunks = [chunks[i] for i in indices[0]]
                context_text = "\n".join(retrieved_chunks)

                # ===============================
                # 4️⃣ Improved Prompt
                # ===============================

                prompt = f"""
                You are an expert Competency Analytics AI.

                IMPORTANT RULES:
                1. Use ONLY the retrieved data below.
                2. Compare numeric values properly.
                3. Identify highest / lowest clearly.
                4. Mention employee name and exact value.
                5. Do NOT guess.
                6. If skill not found, say clearly "Skill not found in data."

                Retrieved Data:
                {context_text}

                User Question:
                {user_question}

                Provide response in this format:

                Answer:
                Explanation:
                """

                try:
                    response = llm.invoke(prompt)

                    st.success("AI Response:")
                    st.write(response.content)

                except Exception as e:
                    st.error("AI Request Failed")
                    st.write(str(e))
