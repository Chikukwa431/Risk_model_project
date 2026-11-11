import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

# Page Configuration & Layout
st.set_page_config(
    page_title="Youth Substance Abuse Risk Assessor - Harare", 
    page_icon="🛡️", 
    layout="wide"
)

# --- CUSTOM CSS STYLING FOR COLORS & INTUITIVENESS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 16px;
        font-weight: 500;
    }
    .card-container {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #475569;
        margin-bottom: 20px;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 8px;
        background-color: rgba(30, 58, 138, 0.4);
        border-left: 6px solid #3B82F6;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECURITY / PASSWORD MECHANISM ---
st.sidebar.header("🔐 Access Security Gate")
app_password = st.sidebar.text_input("Enter Access Password", type="password")

CORRECT_PASSWORD = "tanaka2026"

if app_password != CORRECT_PASSWORD:
    st.markdown('<p class="main-header">🛡️ Predicting High-Risk Drug and Substance Use Among Youths in Harare Province Using a Random Forest Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dissertation Empirical Decision Support Suite | Author: Tanaka Richard Eusebio Muzunze (W230307)</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.warning("⚠️ **Access Restricted:** Please enter the authorized supervisor password in the sidebar to unlock the application interface.")
    st.stop()

# --- HEADER SECTION ---
st.markdown('<p class="main-header">🛡️ Predicting High-Risk Drug and Substance Use Among Youths in Harare Province Using a Random Forest Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive Machine Learning Decision Support & Dissertation Defense Dashboard</p>', unsafe_allow_html=True)
st.caption("**Author:** Tanaka Richard Eusebio Muzunze (W230307) | **Model Architecture:** Balanced Ensemble Random Forest Classifier")
st.markdown("---")

# Sidebar - Dataset Upload
st.sidebar.markdown("### 📁 Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Master CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Dataset Loaded Successfully!\n- **Rows:** {df.shape[0]}\n- **Features:** {df.shape[1]}")
    
    # Preprocessing setup
    X = df.drop(columns=['Youth_ID', 'Substance_Abuse_Risk'], errors='ignore')
    y = df['Substance_Abuse_Risk']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    numeric_features = ['Age', 'Youth_Activity_Level']
    categorical_features = [
        'Gender', 'Peer_Influence', 'Family_Environment', 
        'Psychological_Status', 'Social_Environment', 
        'Drug_Exposure', 'Received_Counselling'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred_eval = pipeline.predict(X_test)
    
    # --- NAVIGATION TABS ---
    tab_home, tab1, tab2, tab3 = st.tabs([
        "🏠 Home & Objectives", 
        "🎯 Single Profile Interactive Inference", 
        "📊 Dissertation Visual Figures", 
        "📋 Full Diagnostics & Metrics"
    ])
    
    with tab_home:
        st.subheader("Welcome to the Dissertation Evaluation Portal")
        
        col_h1, col_h2 = st.columns([1.5, 1])
        
        with col_h1:
            st.markdown("""
            <div class="highlight-box">
            <h4>📌 Core Research Objective</h4>
            <p>This research evaluates and predicts high-risk substance use patterns among youths within Harare Province. By leveraging socio-ecological indicators, peer influence dynamics, and family environment structures, this system deploys an optimized Random Forest machine learning pipeline to identify vulnerability tiers and support early intervention strategies.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🔑 Key System Capabilities
            * **Real-time Predictive Inference:** Test individual youth risk profiles by modifying socio-environmental parameters.
            * **Academic Visualizations:** Explore empirical density plots, confusion matrices, and multi-class ROC curves.
            * **Balanced Optimization:** Mitigates class imbalance to ensure robust sensitivity and specificity across all risk categories.
            """)
            
        with col_h2:
            st.markdown("""
            <div class="card-container">
            <h4>📊 Quick Model Metrics Overview</h4>
            """, unsafe_allow_html=True)
            
            acc_val = accuracy_score(y_test, y_pred_eval)
            f1_val = f1_score(y_test, y_pred_eval, average='macro')
            
            st.metric(label="Global Test Accuracy", value=f"{acc_val*100:.2f}%")
            st.metric(label="Macro F1-Score", value=f"{f1_val:.4f}")
            st.metric(label="Evaluation Split", value=f"{len(y_test)} Cases (20%)")
            st.markdown("</div>", unsafe_allow_html=True)
            
    with tab1:
        st.subheader("Individual Youth Risk Profile Assessment (Harare Context)")
        st.markdown("Adjust the socio-ecological parameters below to evaluate real-time model risk classification, confidence probability breakdown, and actionable recommendations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Youth Age", 10, 25, 16)
            activity = st.slider("Youth Activity Level Index", 0.0, 10.0, 5.0)
            gender = st.selectbox("Gender", df['Gender'].unique() if 'Gender' in df else ['Male', 'Female'])
            peer = st.selectbox("Peer Influence Level", df['Peer_Influence'].unique() if 'Peer_Influence' in df else ['Low', 'Moderate', 'High'])
            family = st.selectbox("Family Environment Status", df['Family_Environment'].unique() if 'Family_Environment' in df else ['Stable', 'Unstable'])
            
        with col2:
            psych = st.selectbox("Psychological Status", df['Psychological_Status'].unique() if 'Psychological_Status' in df else ['Normal', 'Distressed'])
            social = st.selectbox("Social Environment Quality", df['Social_Environment'].unique() if 'Social_Environment' in df else ['Supportive', 'Hostile'])
            exposure = st.selectbox("Prior Drug Exposure History", df['Drug_Exposure'].unique() if 'Drug_Exposure' in df else ['No', 'Yes'])
            counselling = st.selectbox("Received Prior Counselling", df['Received_Counselling'].unique() if 'Received_Counselling' in df else ['No', 'Yes'])
            
        input_data = pd.DataFrame([{
            'Age': age,
            'Youth_Activity_Level': activity,
            'Gender': gender,
            'Peer_Influence': peer,
            'Family_Environment': family,
            'Psychological_Status': psych,
            'Social_Environment': social,
            'Drug_Exposure': exposure,
            'Received_Counselling': counselling
        }])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Predictive Inference Analysis", type="primary", use_container_width=True):
            pred_class_idx = pipeline.predict(input_data)[0]
            pred_probs = pipeline.predict_proba(input_data)[0]
            risk_label = classes[pred_class_idx]
            
            st.markdown("---")
            st.subheader("📋 Evaluation Results & Interpretation")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                if risk_label == "High Risk":
                    st.error(f"### 🔴 Evaluated Risk Tier: **{risk_label}**")
                    st.markdown("**Actionable Insight:** Immediate intervention, professional psychological support, and targeted community counselling recommended.")
                elif risk_label == "Moderate Risk":
                    st.warning(f"### 🟠 Evaluated Risk Tier: **{risk_label}**")
                    st.markdown("**Actionable Insight:** Proactive mentorship, family engagement, and routine monitoring advised to prevent escalation.")
                else:
                    st.success(f"### 🟢 Evaluated Risk Tier: **{risk_label}**")
                    st.markdown("**Actionable Insight:** Stable behavioral profile. Continue positive community engagement and healthy activity levels.")
                    
            with res_col2:
                st.markdown("#### Confidence Probability Distribution:")
                prob_df = pd.DataFrame({'Risk Tier': classes, 'Confidence Probability': pred_probs})
                st.bar_chart(prob_df.set_index('Risk Tier'))

    with tab2:
        st.subheader("Dissertation Graphical Analysis Suite")
        st.markdown("All figures comply with academic APA/IEEE formatting standards.")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Figure 4.0: Feature Density Violin Plot")
            fig4, ax4 = plt.subplots(figsize=(7, 4.5))
            sns.violinplot(
                x='Substance_Abuse_Risk', 
                y='Youth_Activity_Level', 
                data=df, 
                order=['Low Risk', 'Moderate Risk', 'High Risk'],
                palette=['#2ca02c', '#ff7f0e', '#d62728'],
                inner='quartile',
                ax=ax4
            )
            ax4.set_title('Feature Density Distribution Across Risk Tiers', weight='bold', fontsize=10)
            ax4.set_xlabel('Target Substance Abuse Risk Status', fontsize=9)
            ax4.set_ylabel('Youth Activity Level', fontsize=9)
            ax4.grid(axis='y', linestyle=':', alpha=0.5)
            st.pyplot(fig4)
            
        with col_v2:
            st.markdown("#### Figure 4.1: Confusion Matrix Heatmap")
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            cm = confusion_matrix(y_test, y_pred_eval)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', edgecolors='gray', linewidths=0.5,
                        xticklabels=classes, yticklabels=classes, ax=ax1, cbar_kws={'label': 'Cases'})
            ax1.set_title('Confusion Matrix Heatmap', weight='bold', fontsize=10)
            ax1.set_xlabel('Predicted Risk Status', fontsize=9)
            ax1.set_ylabel('Actual Risk Status', fontsize=9)
            st.pyplot(fig1)
            
        col_v3, col_v4 = st.columns(2)
        
        with col_v3:
            st.markdown("#### Figure 4.2: Multi-Class ROC Curves")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            y_prob_eval = pipeline.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for i, color in zip(range(len(classes)), colors):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_eval[:, i])
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')

            ax2.plot([0, 1], [0, 1], color='dimgray', lw=1.5, linestyle='--')
            ax2.set_xlabel('False Positive Rate', fontsize=9)
            ax2.set_ylabel('True Positive Rate', fontsize=9)
            ax2.set_title('Receiver Operating Characteristic (ROC)', weight='bold', fontsize=10)
            ax2.legend(loc="lower right", fontsize=8)
            ax2.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig2)
            
        with col_v4:
            st.markdown("#### Figure 4.3: Feature Importance Ranking")
            cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
            encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
            all_features = numeric_features + encoded_cat_features
            importances = pipeline.named_steps['classifier'].feature_importances_

            feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=True)

            fig3, ax3 = plt.subplots(figsize=(7, 4.5))
            ax3.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='#2c3e50', height=0.65)
            ax3.set_xlabel('Gini Information Gain Weight', fontsize=9)
            ax3.set_ylabel('Predictors', fontsize=9)
            ax3.set_title('Ensemble Feature Importance Ranking', weight='bold', fontsize=10)
            ax3.grid(axis='x', linestyle=':', alpha=0.5)
            st.pyplot(fig3)

    with tab3:
        st.subheader("Comprehensive Classification Report & Metrics")
        st.markdown("Detailed breakdown of precision, recall, and F1-score across all risk strata evaluated on the test set.")
        st.code(classification_report(y_test, y_pred_eval, target_names=classes))
else:
    st.info("👈 Please enter the security password in the sidebar, then upload your `ZIM_Survey_Youth_Substance_Abuse_Dataset.csv` dataset using the sidebar file uploader to launch the system.")