import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from textwrap import dedent

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "max_output_tokens": 300,
        "temperature": 0.6,
        "top_p": 0.9
    }
)

# === Styling ===
st.markdown("""
    <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 12px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #f8f9fb;
            margin-bottom: 12px;
        }
        .message {
            padding: 12px 16px;
            border-radius: 16px;
            margin-bottom: 8px;
            width: fit-content;
            max-width: 80%;
            line-height: 1.4;
            position: relative;
            font-size: 0.95rem;
        }
        .user-msg {
            background: #2563eb;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .bot-msg {
            background: #ffffff;
            color: #1f2d3d;
            border: 1px solid #d1d9e6;
            margin-right: auto;
        }
        .meta {
            font-size: 0.65rem;
            color: #1f2d3d;
            font-weight: bold; /* Highlight You and Assistant */
            margin-bottom: 4px;
        }
        .context-box {
            background: #f1f5fe;
            border-left: 4px solid #2563eb;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 10px;
            white-space: pre-wrap;
        }
        .footer {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 6px;
        }
        .btn-clear {
            background-color: #ef4444;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# === Context / Prompt Components ===
CLUSTER_SUMMARIES = [
    "Regular Shoppers (Cluster 0): frequent moderate purchases. Recommend upselling and loyalty programs.",
    "Premium Shoppers (Cluster 1): high-ticket items but infrequent. Recommend premium campaigns to increase frequency.",
    "Wholesale Buyers (Cluster 2): rare high-volume low-price orders. Recommend B2B onboarding and bulk discounts.",
    "Core Shoppers (Cluster 3): frequent, high-volume buyers driving majority of revenue. Prioritize retention and upsells.",
    "Budget Shoppers (Cluster 4): frequent low-cost purchases. Use promotions and low-cost add-ons."
]
KEY_INSIGHTS = [
    "Core Shoppers contribute the most revenue and volume.",
    "Retention drops sharply after the first period – early reactivation is critical.",
    "Premium Shoppers have high margins but need more frequent engagement.",
    "Wholesale Buyers are a B2B opportunity with irregular cadence."
]
EXEC_SUMMARY = dedent("""
    The project segments customers into meaningful clusters and forecasts demand.
    Objectives: targeted marketing, retention improvement, and inventory optimization.
    Seasonal peaks are predictable, enabling optimized inventory and campaign timing.
""").strip()

def build_prompt(user_question, history, extra_context=None):
    context_parts = [
        "You are a strategic data science assistant for an online retail analytics dashboard.",
        "Use the following customer segmentation and retention insights to answer concisely with actionable recommendations.",
        "Cluster summaries:",
        *CLUSTER_SUMMARIES,
        "Key insights:",
        *KEY_INSIGHTS,
        "Executive summary:",
        EXEC_SUMMARY
    ]
    if extra_context:
        context_parts.append(f"Additional context: {extra_context}")

    context_text = "\n".join(context_parts)

    history_text = ""
    if history:
        recent = history[-6:]  # last few exchanges
        formatted = []
        for speaker, msg in recent:
            prefix = "User:" if speaker == "You" else "Assistant:"
            formatted.append(f"{prefix} {msg}")
        history_text = "\n".join(formatted)

    full_prompt = dedent(f"""
    {context_text}

    {("Conversation so far:\n" + history_text) if history_text else ""}

    User question: {user_question}

    Provide a clear, business-oriented answer referencing relevant clusters, retention patterns, and suggested tactics. If unclear, ask a clarifying follow-up.
    """).strip()
    return full_prompt

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Chatbot UI ===
def chatbot_ui():
    st.markdown("---")
    st.subheader("🤖 Strategy Assistant")

    # Initialize session history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat history container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(
                f"<div class='message user-msg'><div class='meta'>You</div>{message}</div>",
                unsafe_allow_html=True
            )
        else:  # Bot
            st.markdown(
                f"<div class='message bot-msg'><div class='meta'>Assistant</div>{message}</div>",
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Input + buttons
    col1, col2 = st.columns([8, 1])
    with col1:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your question",
                placeholder="E.g., Which cluster should I focus retention on?",
                key="user_input",
                label_visibility="collapsed"
            )
            submit = st.form_submit_button("💬 Send")
    with col2:
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Handle submission
    if submit and user_input and user_input.strip():
        question = user_input.strip()
        prompt = build_prompt(question, st.session_state.chat_history)
        with st.spinner("Thinking..."):
            answer = query_gemini(prompt)
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Bot", answer))
        st.rerun()  # Force UI update to display response immediately

    # Footer guidance
    st.markdown("<div class='footer'>Try: 'How do I retain Core Shoppers?', 'What campaign suits Premium Shoppers?', or 'Which cluster has early churn risk?'</div>", unsafe_allow_html=True)

