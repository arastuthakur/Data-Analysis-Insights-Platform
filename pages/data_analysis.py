import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import google.generativeai as genai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_insights(data_description):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""Analyze this dataset and provide key insights:
    {data_description}
    Focus on:
    1. Key patterns and trends
    2. Potential correlations
    3. Anomalies or interesting findings
    4. Recommendations for further analysis
    """
    response = model.generate_content(prompt)
    return response.text

def get_numeric_correlations(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if len(numeric_data.columns) > 0:
        return numeric_data.corr()
    return pd.DataFrame()

def plot_categorical_distribution(data, column):
    value_counts = data[column].value_counts()
    
    # Create pie chart
    fig_pie = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=f'Distribution of {column} (Pie Chart)',
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Create bar chart
    fig_bar = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f'Distribution of {column} (Bar Chart)',
        template='plotly_dark',
        color=value_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title=column,
        yaxis_title='Count'
    )
    
    return fig_pie, fig_bar

def plot_numeric_distribution(data, column):
    # Create a subplot with histogram and box plot
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data[column],
        name='Distribution',
        marker_color='#00ff00',
        opacity=0.75
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {column}',
        template='plotly_dark',
        showlegend=True,
        xaxis_title=column,
        yaxis_title='Count'
    )
    
    # Create box plot
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=data[column],
        name=column,
        marker_color='#00ff00',
        boxpoints='outliers'
    ))
    
    fig_box.update_layout(
        title=f'Box Plot of {column}',
        template='plotly_dark',
        showlegend=False,
        yaxis_title=column
    )
    
    return fig, fig_box

def plot_correlation_matrix(corr_matrix):
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        template='plotly_dark',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    # Add correlation values as text
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.index)):
            fig.add_annotation(
                x=i,
                y=j,
                text=f"{corr_matrix.iloc[j, i]:.2f}",
                showarrow=False,
                font=dict(color="white")
            )
    
    fig.update_layout(
        width=800,
        height=800
    )
    
    return fig

def plot_scatter_matrix(data, numeric_cols):
    fig = px.scatter_matrix(
        data,
        dimensions=numeric_cols,
        template='plotly_dark',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Scatter Plot Matrix',
        width=1000,
        height=1000
    )
    
    return fig

def show():
    st.title("📊 Data Analysis")
    st.write("Explore your data through visualizations and AI-powered insights.")
    
    data = st.session_state.data
    
    if data is None:
        st.error("Please upload data first!")
        return
    
    # Sidebar for visualization selection
    st.sidebar.title("📈 Visualization Options")
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Overview", "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis"]
    )
    
    if viz_type == "Overview":
        st.write("### 📋 Dataset Overview")
        
        # Display basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Rows:** {len(data)}")
        with col2:
            st.info(f"**Columns:** {len(data.columns)}")
        with col3:
            st.info(f"**Missing Values:** {data.isnull().sum().sum()}")
        
        # Display column information
        st.write("### 📊 Column Information")
        col_types = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values,
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.write(col_types.to_html(index=False), unsafe_allow_html=True)
        
        # Display sample data
        st.write("### 🔍 Sample Data")
        st.dataframe(data.head(), use_container_width=True)
        
    elif viz_type == "Univariate Analysis":
        st.write("### 📊 Univariate Analysis")
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Column selection
        col_type = st.radio("Select Column Type", ["Numeric", "Categorical"])
        
        if col_type == "Numeric" and len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Column", numeric_cols)
            
            # Plot distribution
            fig_hist, fig_box = plot_numeric_distribution(data, selected_col)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Display summary statistics
            st.write("### 📈 Summary Statistics")
            stats = data[selected_col].describe()
            st.dataframe(stats, use_container_width=True)
            
        elif col_type == "Categorical" and len(categorical_cols) > 0:
            selected_col = st.selectbox("Select Column", categorical_cols)
            
            # Plot distribution
            fig_pie, fig_bar = plot_categorical_distribution(data, selected_col)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Display value counts
            st.write("### 📊 Value Counts")
            st.dataframe(data[selected_col].value_counts(), use_container_width=True)
            
    elif viz_type == "Bivariate Analysis":
        st.write("### 📈 Bivariate Analysis")
        
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Plot"]
        )
        
        if plot_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            
            color_col = None
            if len(categorical_cols) > 0:
                if st.checkbox("Color by Category"):
                    color_col = st.selectbox("Select Category", categorical_cols)
            
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{x_col} vs {y_col}",
                template='plotly_dark',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Box Plot":
            y_col = st.selectbox("Select Numeric Column", numeric_cols)
            if len(categorical_cols) > 0:
                x_col = st.selectbox("Group by Category", categorical_cols)
                fig = px.box(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"Box Plot of {y_col} by {x_col}",
                    template='plotly_dark',
                    color=x_col,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "Violin Plot":
            y_col = st.selectbox("Select Numeric Column", numeric_cols)
            if len(categorical_cols) > 0:
                x_col = st.selectbox("Group by Category", categorical_cols)
                fig = px.violin(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"Violin Plot of {y_col} by {x_col}",
                    template='plotly_dark',
                    color=x_col,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "Bar Plot":
            if len(categorical_cols) >= 2:
                x_col = st.selectbox("Select X-axis (Category)", categorical_cols)
                y_col = st.selectbox("Select Y-axis (Category)", [col for col in categorical_cols if col != x_col])
                
                # Create contingency table
                contingency = pd.crosstab(data[x_col], data[y_col])
                fig = px.bar(
                    contingency,
                    title=f"Bar Plot of {x_col} vs {y_col}",
                    template='plotly_dark',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
    elif viz_type == "Correlation Analysis":
        st.write("### 📊 Correlation Analysis")
        
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            # Correlation matrix
            corr_matrix = data[numeric_cols].corr()
            fig = plot_correlation_matrix(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter matrix
            if st.checkbox("Show Scatter Plot Matrix"):
                fig = plot_scatter_matrix(data, numeric_cols)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    # AI Insights
    st.sidebar.write("### 🤖 AI Insights")
    if st.sidebar.button("Generate Insights"):
        with st.spinner("Analyzing data..."):
            # Prepare data description
            numeric_summary = data.describe().to_string() if len(data.select_dtypes(include=['float64', 'int64']).columns) > 0 else "No numeric columns"
            correlation_info = get_numeric_correlations(data).to_string() if len(data.select_dtypes(include=['float64', 'int64']).columns) > 1 else "No correlation data available"
            
            categorical_info = ""
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                categorical_info = "\n\nCategorical Columns Summary:\n"
                for col in categorical_cols:
                    categorical_info += f"\n{col}:\n{data[col].value_counts().to_string()}\n"
            
            description = f"""
            Dataset Summary:
            {numeric_summary}
            
            Correlation Information:
            {correlation_info}
            
            {categorical_info}
            """
            insights = generate_insights(description)
            st.sidebar.write("### 📝 Insights")
            st.sidebar.write(insights) 