import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    LSTM, GRU, Bidirectional, Input, Flatten, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.express as px

def analyze_data_for_modeling(data, target_column, problem_type):
    """Analyze data and provide insights for model selection and parameters."""
    insights = {}
    
    # Analyze target variable
    if problem_type == 'Classification':
        class_counts = data[target_column].value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        n_classes = len(class_counts)
        
        insights['class_balance'] = {
            'min_samples': int(min_class_count),
            'max_samples': int(max_class_count),
            'n_classes': n_classes,
            'imbalance_ratio': float(max_class_count / min_class_count)
        }
        
        # Check for severe class imbalance
        if min_class_count < 2:
            insights['warnings'] = ["Some classes have too few samples (minimum 2 required)"]
            insights['suggestions'] = [
                "Consider collecting more data for underrepresented classes",
                "Use data augmentation techniques",
                "Consider combining rare classes",
                "Use SMOTE or other oversampling techniques"
            ]
    
    # Analyze feature characteristics
    n_features = len(data.drop(columns=[target_column]).columns)
    n_samples = len(data)
    
    insights['data_dimensions'] = {
        'n_samples': n_samples,
        'n_features': n_features,
        'samples_to_features_ratio': n_samples / n_features
    }
    
    return insights

def get_recommended_parameters(insights, model_type, algorithm, problem_type):
    """Get recommended model parameters based on data insights."""
    params = {}
    
    # Base recommendations on data characteristics
    n_samples = insights['data_dimensions']['n_samples']
    n_features = insights['data_dimensions']['n_features']
    
    if model_type == "Neural Networks":
        # Adjust network size based on data size and complexity
        if n_samples < 1000:
            params['architecture'] = 'simple'
            params['batch_size'] = min(16, n_samples // 10)
            params['epochs'] = 50
        elif n_samples < 10000:
            params['architecture'] = 'medium'
            params['batch_size'] = 32
            params['epochs'] = 100
        else:
            params['architecture'] = 'complex'
            params['batch_size'] = 64
            params['epochs'] = 200
        
        params['learning_rate'] = 0.001
        
    elif model_type == "Tree-Based Models":
        # Adjust tree parameters based on data size
        if n_samples < 1000:
            params['n_estimators'] = 50
            params['max_depth'] = min(5, n_features)
        else:
            params['n_estimators'] = 100
            params['max_depth'] = min(10, n_features)
        
        params['min_samples_split'] = max(2, n_samples // 1000)
        params['min_samples_leaf'] = max(1, n_samples // 2000)
    
    elif model_type == "Ensemble Models":
        if algorithm in ["XGBoost", "LightGBM"]:
            params['n_estimators'] = min(200, max(50, n_samples // 100))
            params['max_depth'] = min(8, max(3, n_features // 5))
            params['learning_rate'] = 0.1
    
    # Handle class imbalance for classification
    if problem_type == 'Classification' and 'class_balance' in insights:
        if insights['class_balance']['imbalance_ratio'] > 3:
            params['class_weight'] = 'balanced'
            if model_type in ["Neural Networks", "Deep Learning"]:
                # Calculate class weights
                class_counts = insights['class_balance']['class_counts']
                total = sum(class_counts.values())
                params['class_weights'] = {
                    cls: total / (len(class_counts) * count)
                    for cls, count in class_counts.items()
                }
    
    return params

def prepare_data(data, target_column, problem_type, batch_size=32, sequence_length=10):
    # First, analyze data and get insights
    insights = analyze_data_for_modeling(data, target_column, problem_type)
    
    # Check for severe class imbalance or other issues
    if 'warnings' in insights:
        st.warning("⚠️ Data Quality Issues Detected:")
        for warning in insights['warnings']:
            st.warning(f"- {warning}")
        
        st.info("💡 Suggestions:")
        for suggestion in insights['suggestions']:
            st.info(f"- {suggestion}")
        
        if insights.get('class_balance', {}).get('min_samples', 2) < 2:
            st.error("Cannot proceed with training: Insufficient samples in some classes (minimum 2 required)")
            st.stop()
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Ensure we have enough samples
    if len(X) < 2:
        st.error("Not enough samples for training and testing. Please provide more data.")
        st.stop()
    
    # Handle categorical variables in features
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a copy of X to avoid modifying the original data
    X_processed = X.copy()
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        X_processed[col] = encoders[col].fit_transform(X_processed[col].astype(str))
    
    # Convert target to numeric if it's categorical and get n_classes
    n_classes = None
    if problem_type == 'Classification':
        if y.dtype == 'object':
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y.astype(str))
        n_classes = len(np.unique(y))
        
        # Store class distribution information
        insights['class_balance']['class_counts'] = pd.Series(y).value_counts()
    
    # Scale features using MinMaxScaler for neural networks
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_processed)
    X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
    
    # Split data with stratification for classification
    if problem_type == 'Classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    # Verify we have samples in both train and test sets
    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Error splitting data: Not enough samples in train or test set.")
        st.stop()
    
    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Ensure batch size is not larger than dataset size
    batch_size = min(batch_size, len(X_train))
    
    # For deep learning models, ensure consistent batch size by padding
    if len(X_train) % batch_size != 0:
        pad_size = batch_size - (len(X_train) % batch_size)
        X_train = np.pad(X_train, ((0, pad_size), (0, 0)), mode='edge')
        y_train = np.pad(y_train, (0, pad_size), mode='edge')
    
    if len(X_test) % batch_size != 0:
        pad_size = batch_size - (len(X_test) % batch_size)
        X_test = np.pad(X_test, ((0, pad_size), (0, 0)), mode='edge')
        y_test = np.pad(y_test, (0, pad_size), mode='edge')
    
    # Get input dimension
    input_dim = X_train.shape[1]
    
    return X_train, X_test, y_train, y_test, scaler, insights, n_classes, input_dim

def create_mlp_model(input_dim, problem_type, n_classes=None, architecture='simple'):
    inputs = Input(shape=(input_dim,))
    
    if architecture == 'simple':
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    elif architecture == 'medium':
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    else:  # complex
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    
    if problem_type == 'Classification':
        if n_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(n_classes, activation='softmax')(x)
    else:
        outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_cnn_model(input_dim, problem_type, n_classes=None):
    inputs = Input(shape=(input_dim, 1))
    
    # Calculate number of conv layers based on input dimension
    min_dim = input_dim
    n_conv_layers = 0
    while min_dim >= 4:  # Need at least 4 points for meaningful convolution
        min_dim = (min_dim - 2) // 2  # Effect of Conv1D and MaxPooling1D
        n_conv_layers += 1
    
    x = inputs
    filters = 32
    
    # Ensure at least one conv layer
    n_conv_layers = max(1, n_conv_layers)
    
    for i in range(n_conv_layers):
        x = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        filters = min(filters * 2, 256)  # Double filters but cap at 256
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    if problem_type == 'Classification':
        if n_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(n_classes, activation='softmax')(x)
    else:
        outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_lstm_model(input_dim, problem_type, n_classes=None):
    inputs = Input(shape=(input_dim, 1))
    
    # Calculate LSTM units based on input dimension
    lstm_units = min(128, max(32, input_dim * 2))
    
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(lstm_units // 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    if problem_type == 'Classification':
        if n_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(n_classes, activation='softmax')(x)
    else:
        outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_hybrid_model(input_dim, problem_type, n_classes=None):
    # CNN branch
    cnn_input = Input(shape=(input_dim, 1))
    
    # Calculate number of conv layers
    min_dim = input_dim
    n_conv_layers = 0
    while min_dim >= 4:
        min_dim = (min_dim - 2) // 2
        n_conv_layers += 1
    
    # Ensure at least one conv layer
    n_conv_layers = max(1, n_conv_layers)
    
    cnn = cnn_input
    filters = 32
    
    for i in range(n_conv_layers):
        cnn = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='valid')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        filters = min(filters * 2, 256)
    
    cnn = Flatten()(cnn)
    
    # LSTM branch
    lstm_input = Input(shape=(input_dim, 1))
    lstm_units = min(128, max(32, input_dim * 2))
    
    lstm = LSTM(lstm_units)(lstm_input)
    lstm = BatchNormalization()(lstm)
    
    # Merge branches
    merged = concatenate([cnn, lstm])
    x = Dense(64, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    if problem_type == 'Classification':
        if n_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(n_classes, activation='softmax')(x)
    else:
        outputs = Dense(1)(x)
    
    model = Model(inputs=[cnn_input, lstm_input], outputs=outputs)
    return model

def compile_model(model, problem_type, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    
    if problem_type == 'Classification':
        if model.output_shape[-1] == 1:  # Binary classification
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        else:  # Multiclass classification
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
    else:  # Regression
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
        )
    
    return model

def show():
    st.title("🤖 Model Selection")
    st.write("Select and configure your machine learning model.")
    
    data = st.session_state.data
    
    if data is None:
        st.error("Please upload data first!")
        return
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = st.selectbox(
            "Select Problem Type",
            ["Regression", "Classification"]
        )
        
        # Show only appropriate columns for the problem type
        if problem_type == "Regression":
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns available for regression!")
                return
            target_options = numeric_cols
        else:
            target_options = data.columns
        
        target_column = st.selectbox(
            "Select Target Column",
            target_options
        )

    # Get data insights early
    insights = analyze_data_for_modeling(data, target_column, problem_type)

    with col2:
        model_type = st.selectbox(
            "Select Model Type",
            [
                "Linear Models",
                "Tree-Based Models",
                "Ensemble Models",
                "Neural Networks",
                "Deep Learning",
                "Other Models"
            ]
        )
        
        if model_type == "Linear Models":
            if problem_type == "Regression":
                algorithm = st.selectbox(
                    "Select Algorithm",
                    ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"]
                )
            else:
                algorithm = "Logistic Regression"
        elif model_type == "Tree-Based Models":
            algorithm = st.selectbox(
                "Select Algorithm",
                ["Decision Tree", "Random Forest", "Extra Trees"]
            )
        elif model_type == "Ensemble Models":
            algorithm = st.selectbox(
                "Select Algorithm",
                ["Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM"]
            )
        elif model_type == "Neural Networks":
            algorithm = st.selectbox(
                "Select Architecture",
                ["Simple MLP", "Medium MLP", "Complex MLP"]
            )
        elif model_type == "Deep Learning":
            algorithm = st.selectbox(
                "Select Architecture",
                ["CNN", "LSTM", "Hybrid (CNN+LSTM)"]
            )
        elif model_type == "Other Models":
            if problem_type == "Regression":
                algorithm = st.selectbox(
                    "Select Algorithm",
                    ["SVR", "K-Nearest Neighbors"]
                )
            else:
                algorithm = st.selectbox(
                    "Select Algorithm",
                    ["SVC", "K-Nearest Neighbors"]
                )
    
    # Model parameters
    st.write("### Model Parameters")
    
    # Common parameters
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("Cross-Validation Folds", 2, 10, 3)
    
    if model_type == "Linear Models":
        if algorithm in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
            alpha = st.slider("Alpha (Regularization Strength)", 0.0001, 1.0, 0.1, format="%.4f")
        if algorithm == "Elastic Net":
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, format="%.2f")
    
    elif model_type in ["Tree-Based Models", "Ensemble Models"]:
        n_estimators = st.slider("Number of Estimators", 1, 500, 50)
        max_depth = st.slider("Max Depth", 1, 20, 3)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    
    elif model_type in ["Neural Networks", "Deep Learning"]:
        # Get AI recommended parameters
        recommended_params = get_recommended_parameters(insights, model_type, algorithm, problem_type)
        
        # Use recommended parameters as defaults
        epochs = st.slider("Number of Epochs", 1, 200, recommended_params.get('epochs', 50))
        batch_size = st.slider("Batch Size", 1, 128, recommended_params.get('batch_size', 32))
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, recommended_params.get('learning_rate', 0.001), format="%.4f")
        
        if model_type == "Deep Learning":
            sequence_length = st.slider("Sequence Length", 2, 50, 5)
    
    elif model_type == "Other Models":
        if algorithm in ["SVR", "SVC"]:
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, format="%.2f")
        else:  # K-Nearest Neighbors
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 3)
            weights = st.selectbox("Weight Function", ["uniform", "distance"])
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Analyzing data and preparing model..."):
            try:
                # Initialize cv_scores at the start
                cv_scores = None
                
                # Get batch size for deep learning models
                current_batch_size = batch_size if model_type in ["Neural Networks", "Deep Learning"] else 32
                
                # Prepare data and get insights
                X_train, X_test, y_train, y_test, scaler, insights, n_classes, input_dim = prepare_data(
                    data, target_column, problem_type, batch_size=current_batch_size
                )
                
                # Get recommended parameters based on insights
                recommended_params = get_recommended_parameters(insights, model_type, algorithm, problem_type)
                
                # Display insights and recommendations
                st.write("### 📊 Data Insights")
                if problem_type == 'Classification':
                    st.write("Class Distribution:")
                    class_counts = insights['class_balance']['class_counts']
                    fig = px.bar(x=class_counts.index, y=class_counts.values,
                               title="Class Distribution",
                               labels={'x': 'Class', 'y': 'Count'},
                               template='plotly_dark')
                    st.plotly_chart(fig)
                
                st.write("### 🎯 Recommended Parameters")
                st.json(recommended_params)
                
                # Use recommended parameters if available
                if model_type in ["Neural Networks", "Deep Learning"]:
                    batch_size = recommended_params.get('batch_size', batch_size)
                    epochs = recommended_params.get('epochs', epochs)
                    learning_rate = recommended_params.get('learning_rate', learning_rate)
                elif model_type in ["Tree-Based Models", "Ensemble Models"]:
                    n_estimators = recommended_params.get('n_estimators', n_estimators)
                    max_depth = recommended_params.get('max_depth', max_depth)
                    min_samples_split = recommended_params.get('min_samples_split', min_samples_split)
                    min_samples_leaf = recommended_params.get('min_samples_leaf', min_samples_leaf)
                
                # Initialize model based on type and algorithm
                if model_type == "Linear Models":
                    if problem_type == "Regression":
                        if algorithm == "Linear Regression":
                            model = LinearRegression()
                        elif algorithm == "Ridge Regression":
                            model = Ridge(alpha=alpha)
                        elif algorithm == "Lasso Regression":
                            model = Lasso(alpha=alpha)
                        else:  # Elastic Net
                            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                    else:  # Classification
                        model = LogisticRegression(multi_class='auto')
                
                elif model_type == "Tree-Based Models":
                    if algorithm == "Decision Tree":
                        if problem_type == "Regression":
                            model = DecisionTreeRegressor(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                        else:
                            model = DecisionTreeClassifier(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                    elif algorithm == "Random Forest":
                        if problem_type == "Regression":
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                        else:
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                    else:  # Extra Trees
                        if problem_type == "Regression":
                            model = ExtraTreesRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                        else:
                            model = ExtraTreesClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                
                elif model_type == "Ensemble Models":
                    if algorithm == "Gradient Boosting":
                        if problem_type == "Regression":
                            model = GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                        else:
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                    elif algorithm == "AdaBoost":
                        if problem_type == "Regression":
                            model = AdaBoostRegressor(n_estimators=n_estimators)
                        else:
                            model = AdaBoostClassifier(n_estimators=n_estimators)
                    elif algorithm == "XGBoost":
                        if problem_type == "Regression":
                            model = XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_child_weight=min_samples_leaf
                            )
                        else:
                            model = XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_child_weight=min_samples_leaf
                            )
                    else:  # LightGBM
                        if problem_type == "Regression":
                            model = LGBMRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_child_samples=min_samples_leaf
                            )
                        else:
                            model = LGBMClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_child_samples=min_samples_leaf
                            )
                
                elif model_type == "Neural Networks":
                    architecture = algorithm.split()[0].lower()
                    model = create_mlp_model(input_dim, problem_type, n_classes, architecture)
                    model = compile_model(model, problem_type, learning_rate)
                
                elif model_type == "Deep Learning":
                    if algorithm == "CNN":
                        model = create_cnn_model(input_dim, problem_type, n_classes)
                    elif algorithm == "LSTM":
                        model = create_lstm_model(input_dim, problem_type, n_classes)
                    else:  # Hybrid
                        model = create_hybrid_model(input_dim, problem_type, n_classes)
                    model = compile_model(model, problem_type, learning_rate)
                
                # Model training with progress bar
                with st.spinner("Training model..."):
                    if model_type in ["Neural Networks", "Deep Learning"]:
                        # Prepare data for deep learning
                        if algorithm in ["CNN", "LSTM"]:
                            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                            val_split_idx = int(len(X_train) * 0.8)
                            X_train_val = X_train_reshaped[:val_split_idx]
                            X_val = X_train_reshaped[val_split_idx:]
                            y_train_val = y_train[:val_split_idx]
                            y_val = y_train[val_split_idx:]
                        elif algorithm == "Hybrid":
                            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                            val_split_idx = int(len(X_train) * 0.8)
                            X_train_val = [X_train_reshaped[:val_split_idx], X_train_reshaped[:val_split_idx]]
                            X_val = [X_train_reshaped[val_split_idx:], X_train_reshaped[val_split_idx:]]
                            y_train_val = y_train[:val_split_idx]
                            y_val = y_train[val_split_idx:]
                        else:
                            val_split_idx = int(len(X_train) * 0.8)
                            X_train_val = X_train[:val_split_idx]
                            X_val = X_train[val_split_idx:]
                            y_train_val = y_train[:val_split_idx]
                            y_val = y_train[val_split_idx:]
                        
                        # Train with callbacks
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
                        ]
                        
                        history = model.fit(
                            X_train_val, y_train_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=1
                        )
                        st.session_state.model_history = history.history
                    
                    else:
                        # Traditional ML model training with cross-validation
                        if problem_type == "Regression":
                            cv_scores = cross_val_score(
                                model, X_train, y_train,
                                cv=cv_folds, scoring='neg_mean_squared_error'
                            )
                            cv_rmse = np.sqrt(-cv_scores)
                            st.write(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
                        else:
                            cv_scores = cross_val_score(
                                model, X_train, y_train,
                                cv=cv_folds, scoring='accuracy'
                            )
                            st.write(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                        
                        # Train final model
                        model.fit(X_train, y_train)
                
                # Store model and data in session state
                st.session_state.model = {
                    'model': model,
                    'scaler': scaler,
                    'X_test': X_test,
                    'y_test': y_test,
                    'problem_type': problem_type,
                    'model_type': model_type,
                    'algorithm': algorithm,
                    'cv_scores': cv_scores
                }
                
                st.success("Model trained successfully! Go to Model Results to see the performance.")
            
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error("Please check your data and model configuration.")
                st.error("Detailed error information:")
                st.code(str(e)) 