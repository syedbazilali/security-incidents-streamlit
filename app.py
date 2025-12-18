import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Security Incidents Predictor", layout="wide")

st.title("🛡️ Security Incidents Fatality Predictor")
st.write("Upload your security incidents data to predict fatality likelihood")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Data preprocessing
    with st.spinner("Processing data..."):
        # Convert Total killed to numeric
        df["Total killed"] = pd.to_numeric(
            df["Total killed"].astype(str).str.replace(".", "").str.replace(",", ""),
            errors="coerce"
        )
        
        # Create binary target
        df["Killed_binary"] = (df["Total killed"] > 0).astype(int)
        
        # Remove leakage columns
        leakage_cols = [
            "Total killed", "Nationals killed", "Internationals killed",
            "Total wounded", "Total affected", "Total nationals",
            "Total internationals", "Total kidnapped", "Nationals wounded",
            "Nationals detained", "Internationals wounded"
        ]
        df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)
        
        # Convert numeric columns
        numeric_cols = [
            "Nationals kidnapped", "Internationals kidnapped",
            "Internationals detained", "Total detained",
            "Gender Male", "Gender Female", "Gender Unknown",
            "Latitude", "Longitude"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(".", "").str.replace(",", ""),
                    errors="coerce"
                )
        
        # Handle missing values
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include="object").columns
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")
        
        # Split data
        X = df.drop(columns=["Killed_binary"])
        y = df["Killed_binary"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include="object").columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 [c for c in categorical_features if X[c].nunique() < 50])
            ]
        )
        
        # Train model
        xgb_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(n_estimators=250, max_depth=6, 
                                   learning_rate=0.1, eval_metric="logloss", 
                                   random_state=42))
        ])
        
        xgb_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_pipeline.predict(X_test)
        y_prob = xgb_pipeline.predict_proba(X_test)[:,1]
        
    st.success("Model trained successfully!")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    with col2:
        st.write("### Model Performance")
        st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin")
