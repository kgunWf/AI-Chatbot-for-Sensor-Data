# streamlit_app.py

import streamlit as st
from agent_implementation import SensorDataAgent, CSV_PATH, MODEL_PATH
import time

st.set_page_config(
    page_title="Sensor Data AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 Sensor Data Analysis Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Statistical AI Agent for OK/KO Dataset Analysis</div>', unsafe_allow_html=True)

# Initialize agent (cached)
@st.cache_resource
def load_agent():
    """Load and cache the AI agent."""
    with st.spinner("🔄 Loading AI model... This may take up to a minute on first run."):
        try:
            agent = SensorDataAgent(
                df_path=str(CSV_PATH),
                model_path=str(MODEL_PATH),
            )
            return agent, None
        except Exception as e:
            return None, str(e)

agent, error = load_agent()

if error:
    st.error(f"❌ Failed to load agent: {error}")
    st.stop()

# Sidebar with dataset info and quick actions
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")
    
    try:
        summary = agent.get_dataset_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", summary['total_samples'])
        with col2:
            st.metric("Features", len(summary['columns']))
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("OK Samples", summary['ok_samples'], delta_color="off")
        with col4:
            st.metric("KO Samples", summary['ko_samples'], delta_color="off")
            
    except Exception as e:
        st.warning(f"Could not load dataset summary: {e}")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("📋 Show Dataset Info", use_container_width=True):
        with st.spinner("Fetching dataset information..."):
            response = agent.run("Show me the dataset structure and available sensors")
            st.session_state.messages.append({"role": "user", "content": "Show dataset info"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    if st.button("⭐ Top Features", use_container_width=True):
        with st.spinner("Analyzing discriminative features..."):
            response = agent.run("Show me the most important features that discriminate between OK and KO")
            st.session_state.messages.append({"role": "user", "content": "Show top features"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Queries")
    
    examples = [
        "Plot time series of acc for OK samples",
        "Show frequency spectrum of mic for KO",
        "What are the top discriminative features?",
        "Plot gyro KO time vel-fissa",
        "Show me frequency plot for mag OK"
    ]
    
    for example in examples:
        if st.button(f"📌 {example}", key=example, use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Main chat interface
st.markdown("### 💬 Chat with Your Data")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending query from sidebar
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            response = agent.run(query)
            elapsed = time.time() - start_time
            
            st.markdown(response)
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about your sensor data... (e.g., 'plot acc OK time')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your request..."):
            start_time = time.time()
            response = agent.run(prompt)
            elapsed = time.time() - start_time
            
            st.markdown(response)
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Help section at the bottom
with st.expander("ℹ️ How to Use This Agent"):
    st.markdown("""
    ### Plotting Commands
    Use the format: `<sensor> <OK/KO> <time/frequency> [condition]`
    
    **Available Sensors:** acc, gyro, mag, mic, press, temp, hum
    
    **Examples:**
    - `"plot acc OK time"`
    - `"show mic KO frequency"`
    - `"plot gyro OK time vel-fissa"`
    
    ### Feature Analysis
    Simply ask:
    - `"Show me the top features"`
    - `"What features discriminate OK from KO?"`
    - `"Display feature importance"`
    
    ### Dataset Information
    Ask questions like:
    - `"What sensors are available?"`
    - `"Show dataset structure"`
    - `"How many OK and KO samples?"`
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    '🎓 System and Device Programming - Project Q3 | Powered by Llama 3.1'
    '</div>',
    unsafe_allow_html=True
)