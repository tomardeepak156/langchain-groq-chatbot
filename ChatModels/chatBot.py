import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0e1117;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Cards */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(
            content="You are a funny AI Agent."
        )
    ]

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚙️ Chat Settings")

    st.markdown("---")

    behaviour = st.selectbox(
        "🎭 Choose Behaviour",
        [
            "Funny AI",
            "Professional Assistant",
            "Friendly Teacher",
            "Coding Expert",
            "Data Science Mentor",
            "Interview Coach",
            "Custom"
        ]
    )

    behaviour_prompts = {

        "Funny AI":
            "You are a funny AI Agent. Be helpful, witty and entertaining.",

        "Professional Assistant":
            "You are a professional AI assistant. Give clear, concise and professional answers.",

        "Friendly Teacher":
            "You are a friendly teacher. Explain concepts simply with examples.",

        "Coding Expert":
            "You are an expert programming assistant. Provide accurate code and explain solutions clearly.",

        "Data Science Mentor":
            "You are an expert Data Science mentor. Explain machine learning, statistics, Python and data science concepts practically.",

        "Interview Coach":
            "You are an interview coach. Help the user prepare for technical and HR interviews with practical guidance."
    }

    if behaviour == "Custom":

        system_prompt = st.text_area(
            "✍️ Custom Behaviour",
            value="You are a helpful AI assistant.",
            height=120
        )

    else:

        system_prompt = behaviour_prompts[behaviour]

    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.code(
        "openai/gpt-oss-20b",
        language="text"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = [
            SystemMessage(
                content=system_prompt
            )
        ]

        st.rerun()

    st.markdown("---")

    user_messages = sum(
        isinstance(m, HumanMessage)
        for m in st.session_state.messages
    )

    ai_messages = sum(
        isinstance(m, AIMessage)
        for m in st.session_state.messages
    )

    st.markdown(
        f"""
        <div class="info-card">
            <b>💬 User Messages:</b> {user_messages}<br>
            <b>🤖 AI Responses:</b> {ai_messages}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# UPDATE SYSTEM PROMPT
# ---------------------------------------------------------

if st.session_state.messages:

    st.session_state.messages[0] = SystemMessage(
        content=system_prompt
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI ChatBot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your interactive AI assistant powered by Groq + LangChain</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)

# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

query = st.chat_input(
    "Ask your AI assistant anything..."
)

if query:

    # Add user message
    user_message = HumanMessage(
        content=query
    )

    st.session_state.messages.append(
        user_message
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.write(query)

    # Create model
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=temperature
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = model.invoke(
                    st.session_state.messages
                )

                answer = response.content

                st.write(answer)

                # Store AI response
                st.session_state.messages.append(
                    AIMessage(
                        content=answer
                    )
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )