import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from io import BytesIO

# Load API Key
load_dotenv()
api_key = os.getenv("API_Key")

# Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🧠", layout="wide")

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key="API_Key")

# Initialize Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# System Instruction
system_message = SystemMessage(
    content="You are an AI tutor specialized in Data Science and AI. "
            "If asked about other topics, politely refuse. "
            "Provide structured explanations with examples."
)

# UI Title
st.title("🧠 AI Data Science Tutor")
st.write("Ask anything related to **Data Science & AI**!")

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.memory.clear()
    st.rerun()

# Display Chat History
for msg in st.session_state.memory.chat_memory.messages:
    role = "👤 You:" if isinstance(msg, HumanMessage) else "🤖 AI:"
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").markdown(f"**{role}** {msg.content}")

# User Input
user_input = st.chat_input("Ask me anything about Data Science or AI...")

if user_input:
    # Generate AI Response
    conversation = [system_message] + st.session_state.memory.chat_memory.messages + [HumanMessage(content=user_input)]
    ai_response = chat_model.invoke(conversation)

    # Store in Memory
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(ai_response.content)

    # Display Messages
    st.chat_message("user").markdown(f"👤 **You:** {user_input}")
    st.chat_message("assistant").markdown(f"🤖 **AI:** {ai_response.content}")
    

# Download Chat History
if st.session_state.memory.chat_memory.messages:
    chat_text = "\n".join([msg.content for msg in st.session_state.memory.chat_memory.messages])
    chat_file = BytesIO(chat_text.encode("utf-8"))
    st.sidebar.download_button("💾 Download Chat History", chat_file, file_name="chat_history.txt", mime="text/plain")
