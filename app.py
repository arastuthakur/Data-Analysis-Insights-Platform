import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pages import api_config, data_upload, data_analysis, data_quality, model_selection, model_results

# Page configuration
st.set_page_config(
    page_title="Data Analysis & Insights Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #262730;
        padding: 10px;
        text-align: center;
        z-index: 998;
    }
    .nav-container {
        position: fixed;
        bottom: 40px;
        right: 20px;
        z-index: 999;
        display: flex;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "API Configuration"

# Define page order
PAGE_ORDER = [
    "API Configuration",
    "Data Upload",
    "Data Analysis",
    "Data Quality",
    "Model Selection",
    "Model Results"
]

def get_next_page(current_page):
    try:
        current_index = PAGE_ORDER.index(current_page)
        if current_index < len(PAGE_ORDER) - 1:
            return PAGE_ORDER[current_index + 1]
    except ValueError:
        pass
    return None

def get_prev_page(current_page):
    try:
        current_index = PAGE_ORDER.index(current_page)
        if current_index > 0:
            return PAGE_ORDER[current_index - 1]
    except ValueError:
        pass
    return None

def show_navigation():
    # Create a container for navigation buttons
    nav_col1, nav_col2 = st.columns([6, 1])
    
    with nav_col2:
        next_page = get_next_page(st.session_state.current_page)
        prev_page = get_prev_page(st.session_state.current_page)
        
        # Navigation buttons container
        st.markdown('<div class="nav-container">', unsafe_allow_html=True)
        
        # Previous button
        if prev_page:
            if st.button(f"⬅️ Previous: {prev_page}", key="prev_button"):
                st.session_state.current_page = prev_page
                st.rerun()
        
        # Next button
        if next_page:
            if st.button(f"Next: {next_page} ➡️", key="next_button"):
                st.session_state.current_page = next_page
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_footer():
    st.markdown(
        """
        <div class="footer">
            Made with ❤️ by Arastu Thakur | 
            <a href="https://github.com/arastuthakur" target="_blank" style="color: #FF4B4B;">GitHub</a> | 
            <a href="https://linkedin.com/in/arastu-thakur" target="_blank" style="color: #FF4B4B;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    PAGE_ORDER,
    index=PAGE_ORDER.index(st.session_state.current_page)
)

# Update current page in session state
st.session_state.current_page = page

# Page routing with progress indicator
progress = PAGE_ORDER.index(page) / (len(PAGE_ORDER) - 1)
st.progress(progress)

# Page routing
if page == "API Configuration":
    api_config.show()
elif page == "Data Upload":
    if st.session_state.api_key:
        data_upload.show()
    else:
        st.error("Please configure API key first!")
        if st.button("Go to API Configuration"):
            st.session_state.current_page = "API Configuration"
            st.rerun()
elif page == "Data Analysis":
    if st.session_state.data is not None:
        data_analysis.show()
    else:
        st.error("Please upload data first!")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "Data Upload"
            st.rerun()
elif page == "Data Quality":
    if st.session_state.data is not None:
        data_quality.show()
    else:
        st.error("Please upload data first!")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "Data Upload"
            st.rerun()
elif page == "Model Selection":
    if st.session_state.data is not None:
        model_selection.show()
    else:
        st.error("Please upload data first!")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "Data Upload"
            st.rerun()
elif page == "Model Results":
    if st.session_state.model is not None:
        model_results.show()
    else:
        st.error("Please select a model first!")
        if st.button("Go to Model Selection"):
            st.session_state.current_page = "Model Selection"
            st.rerun()

# Show navigation and footer
show_navigation()
show_footer() 