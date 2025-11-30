# filename: app.py
import streamlit as st

st.set_page_config(page_title="Chatbot UI", layout="wide")

# --- INIT SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"/"assistant", "content": "..."}

# --- PAGE TITLE ---
st.title("My Chatbot")

# --- CUSTOM CSS FOR BUBBLES ---
st.markdown(
    """
    <style>
    .chat-row {
        display: flex;
        margin-bottom: 0.5rem;
        width: 100%;
    }
    .chat-bubble {
        padding: 0.6rem 0.9rem;
        border-radius: 18px;
        max-width: 65%;
        word-wrap: break-word;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .chat-bubble-user {
        margin-left: auto;
        background-color: #DCF8C6; /* light green like WhatsApp */
        border-bottom-right-radius: 4px;
    }
    .chat-bubble-bot {
        margin-right: auto;
        background-color: #FFFFFF;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
    }
    .chat-container {
        padding: 0.5rem 0.5rem 5rem 0.5rem; /* bottom space for input */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CHAT HISTORY ---
#all messages are inside of this div
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

#displays the entire conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":#user messages
        # user on the right
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-bubble chat-bubble-user">
                    {msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:#bot messages
        # bot on the left
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-bubble chat-bubble-bot">
                    {msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

#input box prompting user to tyoe
prompt = st.chat_input("Type your message")

if prompt:
    # 1. save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Show spinner while generating reply
    with st.spinner("Thinking..."):
        # TODO: call backend/model here
        bot_reply = f"You said: {prompt}"

    # 3. save bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # 4. rerun to show the new messages (now the loop includes the new message)
    st.rerun()
