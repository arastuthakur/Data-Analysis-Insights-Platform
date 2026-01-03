import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
import plotly.graph_objects as go

def detect_outliers(data, column, method='zscore', threshold=3):
    """Detect outliers in a column using various methods."""
    if method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        return (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
    return None

def handle_missing_values(data, strategy='mean', n_neighbors=5):
    """Handle missing values using various imputation strategies."""
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    else:
        # Numeric imputation
        num_imputer = SimpleImputer(strategy=strategy)
        data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
        
        # Categorical imputation
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    return data

def handle_class_imbalance(X, y, method='smote', sampling_strategy='auto'):
    """Handle class imbalance using various resampling techniques."""
    if method == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'random_under':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def scale_features(data, method='standard'):
    """Scale features using various scaling methods."""
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def show():
    st.title("🧹 Data Quality Enhancement")
    st.write("Improve your data quality through various preprocessing techniques.")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("Please upload data first!")
        return
    
    data = st.session_state.data.copy()
    
    # Data Quality Overview
    st.write("### 📊 Data Quality Overview")
    
    # Missing Values Analysis
    missing_data = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isnull().sum(),
        'Percentage': (data.isnull().sum() / len(data)) * 100
    })
    
    fig_missing = px.bar(
        missing_data[missing_data['Missing Values'] > 0],
        x='Column',
        y='Percentage',
        title='Missing Values Distribution',
        template='plotly_dark'
    )
    st.plotly_chart(fig_missing)
    
    # Data Type Information
    st.write("### 📋 Data Type Information")
    dtypes_df = pd.DataFrame({
        'Column': data.columns,
        'Data Type': [str(dtype) for dtype in data.dtypes],
        'Unique Values': [data[col].nunique() for col in data.columns]
    })
    st.dataframe(dtypes_df)
    
    # Enhancement Options
    st.write("### 🛠️ Enhancement Options")
    
    # 1. Missing Values Handling
    st.write("#### 1. Missing Values Handling")
    if data.isnull().sum().sum() > 0:
        missing_strategy = st.selectbox(
            "Select Missing Values Strategy",
            ['mean', 'median', 'most_frequent', 'knn'],
            help="Choose how to handle missing values in your dataset"
        )
        
        if missing_strategy == 'knn':
            n_neighbors = st.slider("Number of Neighbors", 1, 10, 5)
            data = handle_missing_values(data, missing_strategy, n_neighbors)
        else:
            data = handle_missing_values(data, missing_strategy)
        
        st.success("Missing values handled successfully!")
    
    # 2. Outlier Detection and Handling
    st.write("#### 2. Outlier Detection and Handling")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select Column for Outlier Analysis", numeric_cols)
        outlier_method = st.selectbox("Select Outlier Detection Method", ['zscore', 'iqr'])
        
        outliers = detect_outliers(data, selected_col, outlier_method)
        if outliers is not None:
            st.write(f"Number of outliers detected: {outliers.sum()}")
            
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data[selected_col], name=selected_col))
            st.plotly_chart(fig_box)
            
            if st.checkbox("Remove Outliers"):
                data = data[~outliers]
                st.success("Outliers removed successfully!")
    
    # 3. Feature Scaling
    st.write("#### 3. Feature Scaling")
    scaling_method = st.selectbox(
        "Select Scaling Method",
        ['standard', 'minmax', 'robust'],
        help="Choose how to scale numeric features"
    )
    
    if st.checkbox("Apply Feature Scaling"):
        data = scale_features(data, scaling_method)
        st.success("Feature scaling applied successfully!")
    
    # 4. Class Imbalance Handling
    st.write("#### 4. Class Imbalance Handling")
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        target_col = st.selectbox("Select Target Column", categorical_cols)
        class_counts = data[target_col].value_counts()
        
        fig_class = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title="Class Distribution",
            template='plotly_dark'
        )
        st.plotly_chart(fig_class)
        
        if max(class_counts) / min(class_counts) > 1.5:  # Check for imbalance
            st.warning("Class imbalance detected!")
            balance_method = st.selectbox(
                "Select Balancing Method",
                ['smote', 'adasyn', 'random_under']
            )
            
            if st.checkbox("Apply Class Balancing"):
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # Handle non-numeric data
                numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) < len(X.columns):
                    st.warning("Non-numeric columns will be encoded before balancing")
                    X = pd.get_dummies(X)
                
                X_resampled, y_resampled = handle_class_imbalance(X, y, balance_method)
                data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=target_col)], axis=1)
                st.success("Class balancing applied successfully!")
    
    # Save Changes
    if st.button("Save Enhanced Dataset"):
        st.session_state.data = data
        st.success("Dataset updated successfully! You can now proceed with model selection.")
        
        # Display sample of enhanced data
        st.write("### 📊 Enhanced Dataset Preview")
        st.write(data.head())
        
        # Display summary statistics
        st.write("### 📈 Summary Statistics")
        st.write(data.describe()) 