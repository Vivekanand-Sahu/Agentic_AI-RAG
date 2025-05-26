import streamlit as st
import os
import sys
# Add src to sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
try:
    from agentic_rag.main import Agentic_RAG_Flow, run_agentic_rag_flow
except ModuleNotFoundError as e:
    st.error(f"Module import error: {e}.\n\nMake sure you have __init__.py files in both src/ and src/agentic_rag/ directories, and that all required Python packages (including crewai-tools) are installed.\nInstall crewai-tools with: pip install crewai-tools")
    st.stop()

st.set_page_config(page_title="Agentic AI RAG Chat", page_icon="🤖")
st.title("Agentic AI RAG Chat")

# Initialize session state for chat history and flow object
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'flow' not in st.session_state:
    st.session_state.flow = Agentic_RAG_Flow()
    # Create session log directory once per session
    base_dir = os.path.dirname(os.path.abspath(__file__))
    session_time = st.session_state.get('session_time')
    if not session_time:
        session_time = st.session_state['session_time'] = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(base_dir, "logs", f"session_{session_time}")
    os.makedirs(session_dir, exist_ok=True)
    st.session_state.flow.state.SESSION_LOG_DIR = session_dir

# Chat UI
user_input = st.text_input("You:", key="user_input")
if st.button("Send") or (user_input and st.session_state.get('last_input') != user_input):
    if user_input:
        st.session_state['last_input'] = user_input
        # Call backend
        answer = run_agentic_rag_flow(st.session_state.flow, user_input)
        st.session_state.chat_history.append((user_input, answer))
        st.experimental_rerun()

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Amar:** {a}")

st.markdown("---")
st.caption("Agentic AI RAG Chat - Powered by Streamlit")
