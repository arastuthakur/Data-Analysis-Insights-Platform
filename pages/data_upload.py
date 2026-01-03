import streamlit as st
import pandas as pd
import io

def show():
    st.title("📤 Data Upload")
    st.write("Upload your dataset for analysis and insights.")
    
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file containing your dataset"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                data = pd.read_csv(uploaded_file)
                
                # Store the data in session state
                st.session_state.data = data
                
                # Display basic information about the dataset
                st.success("File uploaded successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Dataset Info")
                    buffer = io.StringIO()
                    data.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                with col2:
                    st.write("### Dataset Preview")
                    st.dataframe(data.head(), use_container_width=True)
                
                # Display basic statistics
                st.write("### Basic Statistics")
                st.dataframe(data.describe(), use_container_width=True)
                
                # Display missing values information
                st.write("### Missing Values")
                missing_data = pd.DataFrame({
                    'Column': data.columns,
                    'Missing Values': data.isnull().sum(),
                    'Percentage': (data.isnull().sum() / len(data)) * 100
                })
                st.dataframe(missing_data, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading the file: {str(e)}")
        else:
            st.info("Please upload a CSV file to begin analysis.") 