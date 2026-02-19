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
# LOAD GROQ SECRET
# =========================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

# =========================================================
# SIDEBAR SETTINGS (MODEL + TEMPERATURE)
# =========================================================

st.sidebar.subheader("⚙️ AI Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["meta-llama/llama-4-scout-17b-16e-instruct", "mixtral-8x7b-32768"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
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
# NAVIGATION
# =========================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "👤 Self Competency Check",
        "👥 Team Competency Check",
        "🤖 AI Chat Assistant (Advanced RAG)",
        "🧠 Smart Skill Recommendations",
        "📈 Predictive Skill Forecasting",
        "📄 AI PDF Report Generator"
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

        with tab1:
            st.subheader("Team Average Dashboard")
            team_df["Team Average"] = team_df[employee_cols].mean(axis=1)
            grouped = team_df.groupby(area_col).mean(numeric_only=True)
            st.dataframe(grouped[[target_col, "Team Average"]])

        with tab2:
            selected_emp = st.selectbox("Select Team Member", employee_cols)
            grouped = team_df.groupby(area_col).mean(numeric_only=True)
            st.dataframe(grouped[[target_col, selected_emp]])

# =========================================================
# 🤖 ADVANCED RAG CHAT ASSISTANT (WITH MEMORY + PROMPT EDITOR)
# =========================================================

elif menu == "🤖 AI Chat Assistant (Advanced RAG)":

    st.header("🤖 Advanced RAG AI Competency Assistant")

    # Initialize memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload Team Competency Sheet for AI")

    if uploaded_file:

        team_df = pd.read_excel(uploaded_file, header=6)
        team_df.columns = team_df.columns.str.strip()

        st.success("Team Data Loaded")

        # ===============================
        # 1️⃣ Convert rows into chunks
        # ===============================

        chunks = []

        for _, row in team_df.iterrows():
            row_text = " | ".join(
                [f"{col}: {row[col]}" for col in team_df.columns if pd.notna(row[col])]
            )
            chunks.append(row_text)

        # ===============================
        # 2️⃣ Create FAISS Index
        # ===============================

        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.info("Vector database created successfully.")

        # ===============================
        # 3️⃣ Dynamic Prompt Editor
        # ===============================

        st.subheader("🔧 Prompt Configuration")

        default_prompt_template = """
You are an expert Competency Analytics AI.

IMPORTANT RULES:
1. Use ONLY the retrieved data.
2. Use conversation history if relevant.
3. Compare numeric values carefully.
4. Identify highest or lowest clearly.
5. Mention employee name and value.
6. Do NOT guess.

Conversation History:
{history}

Retrieved Data:
{context}

User Question:
{question}

Provide response in format:

Answer:
Explanation:
"""

        user_prompt_template = st.text_area(
            "Edit AI Prompt Template:",
            value=default_prompt_template,
            height=300
        )

        # ===============================
        # 4️⃣ Display Chat History
        # ===============================

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ===============================
        # 5️⃣ Chat Input
        # ===============================

        user_question = st.chat_input("Ask your question")

        if user_question:

            # Show user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("Retrieving relevant data..."):

                # Vector search
                question_embedding = embedding_model.encode([user_question])
                question_embedding = np.array(question_embedding).astype("float32")

                distances, indices = index.search(question_embedding, k=5)

                retrieved_chunks = [chunks[i] for i in indices[0]]
                context_text = "\n".join(retrieved_chunks)

                # Get last 5 messages as memory
                recent_history = st.session_state.chat_history[-5:]
                history_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in recent_history]
                )

                # Build prompt
                prompt = user_prompt_template.format(
                    context=context_text,
                    question=user_question,
                    history=history_text
                )

                try:
                    response = llm.invoke(prompt)
                    assistant_reply = response.content

                    with st.chat_message("assistant"):
                        st.markdown(assistant_reply)

                    # Save assistant reply
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": assistant_reply}
                    )

                except Exception as e:
                    st.error("AI Request Failed")
                    st.write(str(e))


# =========================================================
# 🧠 SMART SKILL RECOMMENDATION ENGINE
# =========================================================

elif menu == "🧠 Smart Skill Recommendations":

    st.header("🧠 Smart Skill Recommendation Engine")

    recommendation_mode = st.radio(
        "Select Mode",
        ["Individual", "Team"]
    )

    uploaded_file = st.file_uploader("Upload Competency Sheet")

    if uploaded_file:

        df = pd.read_excel(uploaded_file, header=6)
        df.columns = df.columns.str.strip()

        area_col = "Area"
        target_col = "Target"

        target_index = df.columns.get_loc("Target")

        # =====================================================
        # 👤 INDIVIDUAL MODE
        # =====================================================

        if recommendation_mode == "Individual":

            current_col = df.columns[target_index + 1]

            df = df[[area_col, target_col, current_col]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df[current_col] = pd.to_numeric(df[current_col], errors="coerce")

            df["Gap"] = df[target_col] - df[current_col]

            # Assign Risk Level
            def assign_risk(gap):
                if gap >= 2:
                    return "High"
                elif gap == 1:
                    return "Medium"
                elif gap > 0:
                    return "Low"
                else:
                    return "No Risk"

            df["Risk Level"] = df["Gap"].apply(assign_risk)

            df = df.sort_values("Gap", ascending=False)

            # 🔥 FILTER
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )

            filtered_df = df[df["Risk Level"].isin(risk_filter)]

            st.subheader("Filtered Skill Gaps")

            for _, row in filtered_df.iterrows():

                if row["Gap"] <= 0:
                    continue

                risk_icon = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢"
                }

                st.markdown(f"""
                {risk_icon[row["Risk Level"]]} **Skill Area:** {row[area_col]}  
                Target: {row[target_col]}  
                Current: {row[current_col]}  
                Gap: {row["Gap"]}  
                Risk Level: {row["Risk Level"]}
                """)
                st.divider()

        # =====================================================
        # 👥 TEAM MODE
        # =====================================================

        else:

            employee_cols = df.columns[target_index + 1:]

            employee_cols = [
                col for col in employee_cols
                if "average" not in col.lower()
                and "maximum" not in col.lower()
                and "unnamed" not in col.lower()
            ]

            df["Team Average"] = df[employee_cols].mean(axis=1)

            df = df[[area_col, target_col, "Team Average"]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df["Team Average"] = pd.to_numeric(df["Team Average"], errors="coerce")

            df["Gap"] = df[target_col] - df["Team Average"]

            def assign_risk(gap):
                if gap >= 2:
                    return "High"
                elif gap >= 1:
                    return "Medium"
                elif gap > 0:
                    return "Low"
                else:
                    return "No Risk"

            df["Risk Level"] = df["Gap"].apply(assign_risk)

            df = df.sort_values("Gap", ascending=False)

            # 🔥 FILTER
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )

            filtered_df = df[df["Risk Level"].isin(risk_filter)]

            st.subheader("Filtered Team Skill Risk Ranking")

            for _, row in filtered_df.iterrows():

                if row["Gap"] <= 0:
                    continue

                risk_icon = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢"
                }

                st.markdown(f"""
                {risk_icon[row["Risk Level"]]} **Skill Area:** {row[area_col]}  
                Target: {row[target_col]}  
                Team Average: {row["Team Average"]:.2f}  
                Gap: {row["Gap"]:.2f}  
                Risk Level: {row["Risk Level"]}
                """)
                st.divider()

# =========================================================
# 📈 PREDICTIVE SKILL GAP FORECASTING
# =========================================================

elif menu == "📈 Predictive Skill Forecasting":

    st.header("📈 Predictive Skill Gap Forecasting")

    forecasting_mode = st.radio(
        "Select Mode",
        ["Individual", "Team"]
    )

    uploaded_file = st.file_uploader("Upload Competency Sheet")

    if uploaded_file:

        df = pd.read_excel(uploaded_file, header=6)
        df.columns = df.columns.str.strip()

        area_col = "Area"
        target_col = "Target"
        target_index = df.columns.get_loc("Target")

        # 🔥 User Input: Improvement Rate
        improvement_rate = st.slider(
            "Improvement per Quarter",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

        forecast_period = st.slider(
            "Forecast Period (Quarters)",
            min_value=1,
            max_value=12,
            value=4
        )

        # =====================================================
        # 👤 INDIVIDUAL MODE
        # =====================================================

        if forecasting_mode == "Individual":

            current_col = df.columns[target_index + 1]

            df = df[[area_col, target_col, current_col]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df[current_col] = pd.to_numeric(df[current_col], errors="coerce")

            df["Future Skill"] = df[current_col] + (
                improvement_rate * forecast_period
            )

            df["Remaining Gap"] = df[target_col] - df["Future Skill"]

            df["Estimated Quarters to Target"] = (
                (df[target_col] - df[current_col]) / improvement_rate
            )

            st.subheader("Forecast Results")

            for _, row in df.iterrows():

                st.markdown(f"""
                **Skill Area:** {row[area_col]}  
                Current: {row[current_col]}  
                Target: {row[target_col]}  
                Forecast After {forecast_period} Quarters: {row["Future Skill"]:.2f}  
                Remaining Gap: {row["Remaining Gap"]:.2f}  
                Estimated Time to Target: {row["Estimated Quarters to Target"]:.1f} Quarters
                """)
                st.divider()

        # =====================================================
        # 👥 TEAM MODE
        # =====================================================

        else:

            employee_cols = df.columns[target_index + 1:]

            employee_cols = [
                col for col in employee_cols
                if "average" not in col.lower()
                and "maximum" not in col.lower()
                and "unnamed" not in col.lower()
            ]

            df["Team Average"] = df[employee_cols].mean(axis=1)

            df = df[[area_col, target_col, "Team Average"]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df["Team Average"] = pd.to_numeric(df["Team Average"], errors="coerce")

            df["Future Skill"] = df["Team Average"] + (
                improvement_rate * forecast_period
            )

            df["Remaining Gap"] = df[target_col] - df["Future Skill"]

            df["Estimated Quarters to Target"] = (
                (df[target_col] - df["Team Average"]) / improvement_rate
            )

            st.subheader("Team Forecast Results")

            for _, row in df.iterrows():

                st.markdown(f"""
                **Skill Area:** {row[area_col]}  
                Team Average: {row["Team Average"]:.2f}  
                Target: {row[target_col]}  
                Forecast After {forecast_period} Quarters: {row["Future Skill"]:.2f}  
                Remaining Gap: {row["Remaining Gap"]:.2f}  
                Estimated Time to Target: {row["Estimated Quarters to Target"]:.1f} Quarters
                """)
                st.divider()

# =========================================================
# 📄 AI GENERATED PDF REPORT
# =========================================================

elif menu == "📄 AI PDF Report Generator":

    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    st.header("📄 AI-Generated Competency Report")

    report_mode = st.radio(
        "Select Report Type",
        ["Individual", "Team"]
    )

    uploaded_file = st.file_uploader("Upload Competency Sheet")

    if uploaded_file:

        df = pd.read_excel(uploaded_file, header=6)
        df.columns = df.columns.str.strip()

        area_col = "Area"
        target_col = "Target"
        target_index = df.columns.get_loc("Target")

        # ===============================
        # INDIVIDUAL REPORT
        # ===============================

        if report_mode == "Individual":

            current_col = df.columns[target_index + 1]

            df = df[[area_col, target_col, current_col]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df[current_col] = pd.to_numeric(df[current_col], errors="coerce")

            df["Gap"] = df[target_col] - df[current_col]

            total_target = df[target_col].sum()
            total_current = df[current_col].sum()

            overall_score = (
                (total_current / total_target) * 100 if total_target else 0
            )

            top_risks = df.sort_values("Gap", ascending=False).head(3)

            # AI Insight
            prompt = f"""
            You are an HR Analytics Expert.

            Overall Score: {overall_score:.2f}%
            Top Risk Skills:
            {top_risks.to_string(index=False)}

            Provide executive summary.
            """

            ai_summary = llm.invoke(prompt).content

        # ===============================
        # TEAM REPORT
        # ===============================

        else:

            employee_cols = df.columns[target_index + 1:]

            employee_cols = [
                col for col in employee_cols
                if "average" not in col.lower()
                and "maximum" not in col.lower()
                and "unnamed" not in col.lower()
            ]

            df["Team Average"] = df[employee_cols].mean(axis=1)

            df = df[[area_col, target_col, "Team Average"]].dropna()

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df["Team Average"] = pd.to_numeric(df["Team Average"], errors="coerce")

            df["Gap"] = df[target_col] - df["Team Average"]

            total_target = df[target_col].sum()
            total_current = df["Team Average"].sum()

            overall_score = (
                (total_current / total_target) * 100 if total_target else 0
            )

            top_risks = df.sort_values("Gap", ascending=False).head(3)

            prompt = f"""
            You are a Workforce Strategy Expert.

            Team Overall Score: {overall_score:.2f}%
            Top Risk Areas:
            {top_risks.to_string(index=False)}

            Provide executive summary.
            """

            ai_summary = llm.invoke(prompt).content

        # ===============================
        # CREATE PDF
        # ===============================

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []

        styles = getSampleStyleSheet()

        elements.append(Paragraph("Competency Intelligence Report", styles["Title"]))
        elements.append(Spacer(1, 0.5 * inch))

        elements.append(Paragraph(f"Overall Score: {overall_score:.2f}%", styles["Heading2"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Top Risk Areas:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        for _, row in top_risks.iterrows():
            elements.append(
                Paragraph(
                    f"{row[area_col]} - Gap: {row['Gap']:.2f}",
                    styles["Normal"]
                )
            )
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("AI Executive Summary:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(ai_summary, styles["Normal"]))

        doc.build(elements)

        buffer.seek(0)

        st.download_button(
            label="📥 Download PDF Report",
            data=buffer,
            file_name="Competency_Report.pdf",
            mime="application/pdf"
        )
