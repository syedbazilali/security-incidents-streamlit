import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Security Incidents Predictor",
    page_icon="🛡️",
    layout="wide"
)

# Title and description
st.title("🛡️ Security Incidents Fatal Outcome Predictor")
st.markdown("""
This application analyzes security incidents data and predicts whether an incident will result in fatalities.
Uses multiple machine learning models including Logistic Regression, Random Forest, XGBoost, SVM, and KNN.
""")

# Sidebar
st.sidebar.header("Settings")
show_eda = st.sidebar.checkbox("Show Exploratory Data Analysis", value=True)
show_training = st.sidebar.checkbox("Show Model Training", value=True)

# Load data
@st.cache_data
def load_data():
    """Load and perform initial data processing"""
    df = pd.read_csv("security_incidents.csv")
    return df

# Preprocess data
@st.cache_data
def preprocess_data(df):
    """Complete data preprocessing pipeline"""
    # Convert Total killed to numeric
    df["Total killed"] = pd.to_numeric(
        df["Total killed"].astype(str).str.replace(".", "").str.replace(",", ""),
        errors="coerce"
    )
    
    # Create binary target
    df["Killed_binary"] = (df["Total killed"] > 0).astype(int)
    
    # Remove data leakage columns
    leakage_cols = [
        "Total killed", "Nationals killed", "Internationals killed",
        "Total wounded", "Total affected", "Total nationals", "Total internationals",
        "Total kidnapped", "Nationals wounded", "Nationals detained", "Internationals wounded"
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
    
    return df

# Train models
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    """Train all models and return results"""
    
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include="object").columns
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             [c for c in categorical_features if X_train[c].nunique() < 50])
        ]
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    # Logistic Regression
    with st.spinner("Training Logistic Regression..."):
        lr_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000))
        ])
        lr_pipeline.fit(X_train, y_train)
        y_prob = lr_pipeline.predict_proba(X_test)[:,1]
        results["Logistic Regression"] = {
            "model": lr_pipeline,
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
    
    # Random Forest
    with st.spinner("Training Random Forest..."):
        rf_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(random_state=42))
        ])
        rf_params = {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }
        rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring="roc_auc", n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        results["Random Forest"] = {
            "model": rf_grid.best_estimator_,
            "roc_auc": rf_grid.best_score_
        }
    
    # XGBoost
    with st.spinner("Training XGBoost..."):
        xgb_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(eval_metric="logloss", random_state=42))
        ])
        xgb_params = {
            "model__n_estimators": [150, 250],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1]
        }
        xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=cv, scoring="roc_auc", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        results["XGBoost"] = {
            "model": xgb_grid.best_estimator_,
            "roc_auc": xgb_grid.best_score_
        }
    
    # SVM
    with st.spinner("Training SVM..."):
        pca = PCA(n_components=0.95, random_state=42)
        svm_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("pca", pca),
            ("model", SVC(probability=True, random_state=42))
        ])
        svm_params = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }
        svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring="roc_auc", n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        results["SVM"] = {
            "model": svm_grid.best_estimator_,
            "roc_auc": svm_grid.best_score_
        }
    
    # KNN
    with st.spinner("Training KNN..."):
        knn_pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("pca", pca),
            ("model", KNeighborsClassifier())
        ])
        knn_params = {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
        knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=cv, scoring="roc_auc", n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        results["KNN"] = {
            "model": knn_grid.best_estimator_,
            "roc_auc": knn_grid.best_score_
        }
    
    return results

# Main app logic
try:
    df = load_data()
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # EDA Section
    if show_eda:
        st.header("📊 Exploratory Data Analysis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Incidents", len(df))
        col2.metric("Total Countries", df['Country'].nunique())
        col3.metric("Date Range", f"{df['Year'].min()} - {df['Year'].max()}")
        
        # Missing values
        st.subheader("Missing Values Analysis")
        missing = df.isna().mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        missing.plot(kind='barh', ax=ax)
        ax.set_xlabel("Missing Rate")
        st.pyplot(fig)
    
    # Preprocessing
    df_processed = preprocess_data(df)
    st.success("✅ Data preprocessing completed!")
    
    # Show class distribution
    st.subheader("Target Variable Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Class Distribution:")
        st.write(df_processed["Killed_binary"].value_counts())
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_processed["Killed_binary"].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Killed Binary (0=No, 1=Yes)")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        st.pyplot(fig)
    
    # Train-test split
    X = df_processed.drop(columns=["Killed_binary"])
    y = df_processed["Killed_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    st.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Model Training
    if show_training:
        st.header("🤖 Model Training & Evaluation")
        
        results = train_models(X_train, y_train, X_test, y_test)
        
        # Results comparison
        st.subheader("Model Performance Comparison")
        results_df = pd.DataFrame({
            "Model": list(results.keys()),
            "ROC-AUC": [results[m]["roc_auc"] for m in results.keys()]
        }).sort_values("ROC-AUC", ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(results_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(results_df["Model"], results_df["ROC-AUC"])
            ax.set_xlabel("ROC-AUC Score")
            ax.set_title("Model Performance Comparison")
            st.pyplot(fig)
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(results.items()):
            model = result["model"]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(cmap="Blues", ax=axes[idx], colorbar=False)
            axes[idx].set_title(name)
        
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model details
        best_model_name = results_df.iloc[0]["Model"]
        st.success(f"🏆 Best Model: {best_model_name} with ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
        
        # Classification report for best model
        st.subheader(f"Detailed Report: {best_model_name}")
        best_model = results[best_model_name]["model"]
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

except FileNotFoundError:
    st.error("❌ Error: 'security_incidents.csv' not found. Please upload the dataset.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.write("Please check your data file and try again.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Security Incidents Analysis")
