import streamlit as st
from langchain_ollama import ChatOllama
from pydantic import BaseModel


# -----------------------------
# Data models
# -----------------------------

class ActivityAnalysis(BaseModel):
    activity: str
    category: str
    why: str
    ai_could_do: str
    human_focus: str
    risk: str


class WorkRedesignAnalysis(BaseModel):
    analyses: list[ActivityAnalysis]
    lowest_confidence_activity: str
    lowest_confidence_reason: str


# -----------------------------
# Ollama model
# -----------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    WorkRedesignAnalysis
)


# -----------------------------
# Streamlit interface
# -----------------------------

st.set_page_config(
    page_title="Work Redesign Advisor",
    page_icon="🤖",
    layout="wide"
)

st.title("Work Redesign Advisor")

st.write(
    "Redesign work around the strengths of humans and AI."
)


role = st.text_input(
    "Role",
    placeholder="e.g. HR Business Partner"
)


activities_input = st.text_area(
    "Major activities",
    placeholder=(
        "e.g. Workforce planning, Employee relations, "
        "Talent reviews, HR reporting"
    ),
    height=120
)


# -----------------------------
# Analyse button
# -----------------------------

if st.button("Analyse work", type="primary"):

    if not role.strip():
        st.warning("Please enter a role.")
        st.stop()

    if not activities_input.strip():
        st.warning("Please enter at least one activity.")
        st.stop()

    activities = [
        activity.strip()
        for activity in activities_input.split(",")
        if activity.strip()
    ]

    activity_list = "\n".join(
        f"{i + 1}. {activity}"
        for i, activity in enumerate(activities)
    )

    prompt = f"""
You are a Work Redesign Advisor helping organisations redesign work
as AI agents become part of the workforce.

Role being analysed:
{role}

Major activities:
{activity_list}

Analyse every activity exactly once.

Do not rename, combine, omit, reinterpret, or invent activities.

Classify each activity into exactly ONE category:

1. Human-led
2. Human + AI
3. Potentially agent-led

Do NOT assume that an activity should become agent-led simply because
AI is technically capable of performing it.

Evaluate each activity using:

1. Repeatability
2. Rule-based nature
3. Data availability
4. Human judgement
5. Human interaction
6. Consequence of error
7. Accountability

Human-led means the activity depends heavily on human judgement,
relationships, empathy, negotiation, accountability or sensitive decisions.

Human + AI means AI can substantially support, analyse, recommend,
prepare or augment the work, while meaningful human judgement remains.

Potentially agent-led means the activity is highly repeatable,
rule-based and data-driven, has relatively low consequences of error,
and can reasonably be executed by an AI agent with appropriate controls.

For each activity provide:

- Activity
- Category
- Why
- What AI could do
- What the human should focus on
- Key risk or consideration

Finally identify the activity with the lowest confidence and explain why.
"""

    # -----------------------------
    # Call Ollama
    # -----------------------------

    with st.spinner("Analysing work with Ollama..."):

        try:
            response = structured_llm.invoke(prompt)

        except Exception as e:
            st.error("The analysis could not be completed.")
            st.exception(e)
            st.stop()


    # -----------------------------
    # Display results
    # -----------------------------

    st.divider()

    st.header("Work Redesign Analysis")

    st.write(f"**Role:** {role}")

    for analysis in response.analyses:

        st.subheader(analysis.activity)

        st.write(
            f"**Recommendation:** {analysis.category}"
        )

        st.write(
            f"**Why:** {analysis.why}"
        )

        st.write(
            f"**AI could do:** {analysis.ai_could_do}"
        )

        st.write(
            f"**Human focus:** {analysis.human_focus}"
        )

        st.write(
            f"**Risk:** {analysis.risk}"
        )

        st.divider()


    # -----------------------------
    # Lowest confidence
    # -----------------------------

    st.header("Lowest Confidence")

    st.write(
        f"**Activity:** "
        f"{response.lowest_confidence_activity}"
    )

    st.write(
        f"**Reason:** "
        f"{response.lowest_confidence_reason}"
    )