import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)

def plot_regression_results(y_true, y_pred):
    results_df = pd.DataFrame({
        'True Values': y_true,
        'Predicted Values': y_pred
    })
    
    fig = px.scatter(
        results_df,
        x='True Values',
        y='Predicted Values',
        title='Predicted vs True Values',
        template='plotly_dark',
        color_discrete_sequence=['#00ff00']
    )
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='True',
        xaxis={'side': 'bottom'},
        template='plotly_dark',
        width=700,
        height=700
    )
    
    return fig

def plot_roc_curves(y_true, y_pred_proba, classes):
    fig = go.Figure()
    
    # Ensure y_true is 1D
    y_true = np.array(y_true).ravel()
    
    if len(classes) == 2:
        # Binary classification
        if y_pred_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            mode='lines',
            line=dict(color='#00ff00', width=2)
        ))
    else:
        # Multiclass classification
        for i, class_name in enumerate(classes):
            y_true_binary = (y_true == i).astype(int)
            if y_pred_proba.shape[1] >= len(classes):
                class_proba = y_pred_proba[:, i]
            else:
                class_proba = (y_pred_proba == i).astype(float)
            
            fpr, tpr, _ = roc_curve(y_true_binary, class_proba)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve {class_name} (AUC = {roc_auc:.2f})',
                mode='lines'
            ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        mode='lines',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_precision_recall_curves(y_true, y_pred_proba, classes):
    fig = go.Figure()
    
    # Ensure y_true is 1D
    y_true = np.array(y_true).ravel()
    
    if len(classes) == 2:
        # Binary classification
        if y_pred_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            ap = average_precision_score(y_true, y_pred_proba[:, 1])
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba.ravel())
            ap = average_precision_score(y_true, y_pred_proba.ravel())
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f'PR curve (AP = {ap:.2f})',
            mode='lines',
            line=dict(color='#00ff00', width=2)
        ))
    else:
        # Multiclass classification
        for i, class_name in enumerate(classes):
            y_true_binary = (y_true == i).astype(int)
            if y_pred_proba.shape[1] >= len(classes):
                class_proba = y_pred_proba[:, i]
            else:
                class_proba = (y_pred_proba == i).astype(float)
            
            precision, recall, _ = precision_recall_curve(y_true_binary, class_proba)
            ap = average_precision_score(y_true_binary, class_proba)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=f'PR curve {class_name} (AP = {ap:.2f})',
                mode='lines'
            ))
    
    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_feature_importance(importance_df):
    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Feature Importance',
        template='plotly_dark',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_prediction_distribution(y_true, y_pred, problem_type):
    fig = go.Figure()
    
    if problem_type == 'Regression':
        fig.add_trace(go.Histogram(
            x=y_true,
            name='True Values',
            opacity=0.75,
            marker_color='#00ff00'
        ))
        fig.add_trace(go.Histogram(
            x=y_pred,
            name='Predicted Values',
            opacity=0.75,
            marker_color='#ff0000'
        ))
    else:
        # For classification, create a bar plot of class distributions
        true_dist = pd.Series(y_true).value_counts()
        pred_dist = pd.Series(y_pred).value_counts()
        
        fig.add_trace(go.Bar(
            x=true_dist.index,
            y=true_dist.values,
            name='True Distribution',
            marker_color='#00ff00',
            opacity=0.75
        ))
        fig.add_trace(go.Bar(
            x=pred_dist.index,
            y=pred_dist.values,
            name='Predicted Distribution',
            marker_color='#ff0000',
            opacity=0.75
        ))
    
    fig.update_layout(
        title='Distribution of True vs Predicted Values',
        barmode='overlay',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_learning_curves(history):
    fig = go.Figure()
    
    for metric in history.history.keys():
        fig.add_trace(go.Scatter(
            y=history.history[metric],
            name=metric,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Epoch',
        yaxis_title='Value',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def calculate_class_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive metrics for classification, including per-class metrics."""
    metrics = {}
    
    # Ensure arrays are 1D and integer type
    y_true = np.array(y_true).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    
    # Get unique classes from both true and predicted values
    classes = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    n_classes = len(classes)
    
    # Ensure y_pred_proba has correct shape
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
            # Binary classification with single probability
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        elif y_pred_proba.shape[1] != n_classes:
            # Invalid probability shape, create one-hot encoded probabilities
            temp_proba = np.zeros((len(y_pred), n_classes))
            for i in range(n_classes):
                temp_proba[:, i] = (y_pred == i).astype(float)
            y_pred_proba = temp_proba
    
    # Calculate overall metrics with zero_division=0
    metrics['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    # Per-class metrics
    metrics['per_class'] = {}
    
    for i, class_label in enumerate(classes):
        # Calculate true positives, false positives, etc.
        tp = np.sum((y_true == class_label) & (y_pred == class_label))
        fp = np.sum((y_true != class_label) & (y_pred == class_label))
        fn = np.sum((y_true == class_label) & (y_pred != class_label))
        tn = np.sum((y_true != class_label) & (y_pred != class_label))
        
        # Calculate metrics with handling for zero division
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        
        metrics['per_class'][str(class_label)] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(y_true == class_label)),
            'predicted': int(np.sum(y_pred == class_label))  # Add count of predictions per class
        }
    
    return metrics

def show():
    st.title("📈 Model Results")
    st.write("Evaluate your model's performance.")
    
    if st.session_state.model is None:
        st.error("No model found. Please train a model first!")
        return
    
    model_info = st.session_state.model
    model = model_info['model']
    X_test = model_info['X_test']
    y_test = model_info['y_test']
    problem_type = model_info['problem_type']
    model_type = model_info['model_type']
    algorithm = model_info['algorithm']
    
    # Validate test data
    if len(X_test) == 0 or len(y_test) == 0:
        st.error("Test data is empty. Please ensure your dataset has enough samples for testing.")
        return
    
    # Set consistent batch size for predictions
    BATCH_SIZE = min(32, len(X_test))  # Ensure batch size doesn't exceed data size
    
    try:
        # Convert y_test to numpy array and ensure integer type for classification
        y_test = np.array(y_test)
        if problem_type == 'Classification':
            y_test = y_test.astype(int)
            
            # Get predictions
            if model_type == "Deep Learning" or model_type == "Neural Networks":
                # Reshape input if needed
                if algorithm in ["CNN", "LSTM", "Hybrid"]:
                    if len(X_test.shape) == 2:
                        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    else:
                        X_test_reshaped = X_test
                    
                    if algorithm == "Hybrid":
                        X_test_reshaped = [X_test_reshaped, X_test_reshaped]
                else:
                    X_test_reshaped = X_test
                
                # Get raw predictions
                y_pred_raw = model.predict(X_test_reshaped, batch_size=BATCH_SIZE)
                
                # For multiclass (3 or more classes)
                if len(y_pred_raw.shape) > 1 and y_pred_raw.shape[1] > 1:
                    y_pred_proba = y_pred_raw  # Keep probabilities as is
                    y_pred = np.argmax(y_pred_raw, axis=1)  # Get class predictions
                else:
                    # Binary classification
                    y_pred = (y_pred_raw.squeeze() > 0.5).astype(int)
                    y_pred_proba = np.column_stack([1 - y_pred_raw.squeeze(), y_pred_raw.squeeze()])
            else:
                # Traditional ML models
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    # Create one-hot encoded probabilities
                    n_classes = len(np.unique(y_test))
                    y_pred_proba = np.zeros((len(y_pred), n_classes))
                    for i in range(n_classes):
                        y_pred_proba[:, i] = (y_pred == i).astype(float)
            
            # Ensure predictions are properly shaped
            y_pred = np.array(y_pred).ravel()
            y_test = np.array(y_test).ravel()
            
            # Get unique classes
            classes = sorted(list(set(np.unique(y_test)) | set(np.unique(y_pred))))
            class_names = [str(c) for c in classes]
            
            # Calculate metrics
            metrics = calculate_class_metrics(y_test, y_pred, y_pred_proba)
            
            # Display model information
            st.write("### 🤖 Model Information")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.info(f"**Problem Type:** {problem_type}")
            with info_col2:
                st.info(f"**Model Type:** {model_type}")
            with info_col3:
                st.info(f"**Algorithm:** {algorithm}")
            with info_col4:
                st.info(f"**Number of Classes:** {len(classes)}")
            
            # Display overall metrics
            st.write("### 📊 Overall Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['overall']['accuracy']:.4f}")
            with col2:
                st.metric("Weighted Precision", f"{metrics['overall']['weighted_precision']:.4f}")
            with col3:
                st.metric("Weighted Recall", f"{metrics['overall']['weighted_recall']:.4f}")
            with col4:
                st.metric("Weighted F1", f"{metrics['overall']['weighted_f1']:.4f}")
            
            # Display macro-averaged metrics
            st.write("### 📈 Macro-Averaged Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Macro Precision", f"{metrics['overall']['macro_precision']:.4f}")
            with col2:
                st.metric("Macro Recall", f"{metrics['overall']['macro_recall']:.4f}")
            with col3:
                st.metric("Macro F1", f"{metrics['overall']['macro_f1']:.4f}")
            
            # Display per-class metrics
            st.write("### 📊 Per-Class Performance")
            per_class_df = pd.DataFrame.from_dict(metrics['per_class'], orient='index')
            st.dataframe(per_class_df.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1': '{:.4f}',
                'support': '{:d}',
                'predicted': '{:d}'
            }))
            
            # Display detailed classification report
            st.write("### 📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=class_names)
            st.code(report)
            
            # Plot confusion matrix
            st.write("### 🎯 Confusion Matrix")
            fig = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot ROC curves
            st.write("### 📈 ROC Curves")
            fig = plot_roc_curves(y_test, y_pred_proba, classes)
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot Precision-Recall curves
            st.write("### 📈 Precision-Recall Curves")
            fig = plot_precision_recall_curves(y_test, y_pred_proba, classes)
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot prediction distribution
            st.write("### 📊 Prediction Distribution")
            fig = plot_prediction_distribution(y_test, y_pred, problem_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.write("### 🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': X_test.columns if isinstance(X_test, pd.DataFrame) else [f'Feature_{i}' for i in range(X_test.shape[1])],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = plot_feature_importance(importance_df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Learning curves (if available)
            if 'model_history' in st.session_state:
                st.write("### 📈 Learning Curves")
                history = st.session_state.model_history
                
                # Create figure for learning curves
                fig = go.Figure()
                
                # Plot training metrics
                for metric in history.keys():
                    if not metric.startswith('val_'):
                        fig.add_trace(go.Scatter(
                            y=history[metric],
                            name=metric,
                            mode='lines'
                        ))
                        # Plot validation metrics if available
                        val_metric = f'val_{metric}'
                        if val_metric in history:
                            fig.add_trace(go.Scatter(
                                y=history[val_metric],
                                name=val_metric,
                                mode='lines',
                                line=dict(dash='dash')
                            ))
                
                fig.update_layout(
                    title='Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Value',
                    template='plotly_dark',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download predictions
            st.write("### 💾 Download Results")
            predictions_df = pd.DataFrame({
                'True Values': y_test,
                'Predicted Values': y_pred
            })
            
            if problem_type == 'Classification' and y_pred_proba is not None:
                for i, class_name in enumerate(class_names):
                    predictions_df[f'Probability_{class_name}'] = y_pred_proba[:, i]
            
            st.download_button(
                label="📥 Download Predictions",
                data=predictions_df.to_csv(index=False).encode('utf-8'),
                file_name="predictions.csv",
                mime="text/csv"
            )
        
        else:  # Regression
            # Handle regression predictions
            if model_type == "Deep Learning":
                if isinstance(X_test, list):  # Hybrid model
                    y_pred = model.predict(X_test, batch_size=BATCH_SIZE).squeeze()
                else:  # CNN or LSTM
                    if len(X_test.shape) == 2:
                        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    else:
                        X_test_reshaped = X_test
                    y_pred = model.predict(X_test_reshaped, batch_size=BATCH_SIZE).squeeze()
            else:
                y_pred = model.predict(X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            st.write("### 📊 Regression Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Plot regression results
            st.write("### 📈 Regression Plot")
            fig = plot_regression_results(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        st.error("Please check your model and data.")
        st.error("Detailed error information:")
        st.code(str(e)) 