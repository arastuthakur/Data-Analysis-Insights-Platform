import streamlit as st
import google.generativeai as genai
import time

def test_api_key(api_key):
    try:
        # Configure Gemini with the provided API key
        genai.configure(api_key=api_key)
        
        # Test the API key with a simple request
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Test API connection")
        
        return True, "API key is valid and working correctly!"
    except Exception as e:
        return False, str(e)

def show():
    st.title("🔑 API Configuration")
    st.write("Configure your Gemini API key to get started with the platform.")
    
    # Check if API key is already configured
    if st.session_state.api_key:
        st.success("API key is already configured!")
        
        # Add option to test current API key
        if st.button("Test Current API Key"):
            with st.spinner("Testing API key..."):
                is_valid, message = test_api_key(st.session_state.api_key)
                if is_valid:
                    st.success(message)
                else:
                    st.error(f"Error testing API key: {message}")
        
        # Add option to update API key
        if st.checkbox("Update API Key"):
            show_api_config()
    else:
        show_api_config()

def show_api_config():
    with st.container():
        st.markdown("""
        ### How to get your API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy and paste it below
        
        ### API Key Requirements:
        - Must be a valid Google AI Studio API key
        - Should have access to the Gemini Pro model
        - Should have sufficient quota for your usage
        """)
        
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Your API key will be stored securely in the session state"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test API Key", key="test_key"):
                if api_key:
                    with st.spinner("Testing API key..."):
                        is_valid, message = test_api_key(api_key)
                        if is_valid:
                            st.success(message)
                        else:
                            st.error(f"Error testing API key: {message}")
                else:
                    st.warning("Please enter an API key to test.")
        
        with col2:
            if st.button("Save API Key", key="save_key"):
                if api_key:
                    with st.spinner("Verifying and saving API key..."):
                        is_valid, message = test_api_key(api_key)
                        if is_valid:
                            st.session_state.api_key = api_key
                            st.success("API key configured successfully! You can now proceed to Data Upload.")
                            
                            # Add a progress bar for visual feedback
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            # Refresh the page
                            st.rerun()
                        else:
                            st.error(f"Error configuring API key: {message}")
                else:
                    st.warning("Please enter an API key.")
        
        # Add API usage information
        if st.session_state.api_key:
            st.write("### 📊 API Usage Information")
            try:
                # Here you would typically make an API call to get usage information
                # For now, we'll just show a placeholder
                st.info("""
                - Model: Gemini Pro
                - Status: Active
                - Requests today: Available
                - Rate limits: Standard tier
                """)
            except Exception as e:
                st.warning("Could not fetch API usage information.") 