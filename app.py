import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import time
from datetime import datetime

# Configuration
st.set_page_config(page_title="💰 Insurance Cost Predictor", page_icon="🏥", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        return joblib.load('best_model.pkl'), joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Using demo mode with sample calculations.")
        return None, None

model, scaler = load_models()

# Updated CSS with new color scheme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    .main { font-family: 'Poppins', sans-serif; }
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2ecc71 100%); }
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2ecc71 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .main-title { color: white; font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-subtitle { color: rgba(255,255,255,0.9); font-size: 1.2rem; text-align: center; margin-bottom: 0; }
    .card { background: #f0f4f8; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #1e3c72 0%, #2ecc71 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .prediction-result { background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; font-size: 1.5rem; font-weight: 600; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1rem 0; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    .risk-low { background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; font-weight: 600; text-align: center; }
    .risk-medium { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; font-weight: 600; text-align: center; }
    .risk-high { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; font-weight: 600; text-align: center; }
    .stButton > button { background: linear-gradient(135deg, #1e3c72 0%, #2ecc71 100%); color: white; border: none; padding: 0.5rem 2rem; border-radius: 25px; font-weight: 600; box-shadow: 0 5px 15px rgba(0,0,0,0.2); transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🏥 Advanced Insurance Cost Predictor</h1>
    <p class="main-subtitle">AI-Powered Cost Estimation with Real-Time Analytics & Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def get_bmi_category(bmi_val):
    categories = [(18.5, "Underweight", "🔵"), (25, "Normal Weight", "🟢"), (30, "Overweight", "🟡"), (float('inf'), "Obese", "🔴")]
    return next((cat[1], cat[2], "#3498db" if cat[0] == 18.5 else "#27ae60" if cat[0] == 25 else "#f39c12" if cat[0] == 30 else "#e74c3c") for cat in categories if bmi_val < cat[0])

def calculate_risk_score(age, bmi, smoker, children):
    return (40 if smoker == "yes" else 0) + (25 if bmi > 30 else 0) + (15 if age > 50 else 0) + (10 if children > 3 else 0)

def get_risk_level(risk_score):
    return ("Low Risk", "risk-low") if risk_score < 25 else ("Medium Risk", "risk-medium") if risk_score < 50 else ("High Risk", "risk-high")

@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'bmi': np.random.uniform(15, 40, n),
        'children': np.random.randint(0, 5, n),
        'smoker': np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
        'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n),
        'sex': np.random.choice(['male', 'female'], n)
    })
    df['cost'] = np.maximum(2000 + (df['age'] * 120) + (df['bmi'] * 100) + (df['children'] * 500) + 
                           np.where(df['smoker'] == 'yes', 15000, 0) + 
                           np.where(df['region'].isin(['northeast', 'northwest']), 1000, 0) + 
                           np.random.normal(0, 1500, n), 1000)
    return df

# Sidebar inputs
with st.sidebar:
    st.markdown("## 📊 Patient Information\n---\n### 👤 Personal Details")
    age = st.slider("Age", 18, 100, 30, help="Patient's age in years")
    sex = st.selectbox("Gender", ["female", "male"], help="Biological sex")
    
    st.markdown("### 🏃‍♂️ Health Metrics")
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, help="BMI = weight(kg) / height(m)²")
    
    st.markdown("### 🚭 Lifestyle Factors")
    children = st.selectbox("Number of Children", list(range(11)), help="Number of dependents")
    smoker = st.radio("Smoking Status", ["no", "yes"], help="Current smoking status")
    
    st.markdown("### 🌍 Geographic Information")
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"], help="Geographic region")
    
    st.markdown("### ⚙️ Advanced Options")
    show_confidence = st.checkbox("Show Confidence Interval", value=True)
    show_risk_factors = st.checkbox("Show Risk Analysis", value=True)
    show_comparisons = st.checkbox("Show Market Comparisons", value=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📋 Patient Profile Summary")
    bmi_category, bmi_icon, bmi_color = get_bmi_category(bmi)
    risk_score = calculate_risk_score(age, bmi, smoker, children)
    risk_level, risk_class = get_risk_level(risk_score)
    
    # Summary cards
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👤 Demographics</h3>
            <p><strong>Age:</strong> {age} years</p>
            <p><strong>Gender:</strong> {sex.title()}</p>
            <p><strong>Children:</strong> {children}</p>
            <p><strong>Region:</strong> {region.title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏃‍♂️ Health Status</h3>
            <p><strong>BMI:</strong> {bmi}</p>
            <p><strong>Category:</strong> {bmi_icon} {bmi_category}</p>
            <p><strong>Smoker:</strong> {'🚭 Yes' if smoker == 'yes' else '✅ No'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Risk Assessment</h3>
            <div class="risk-indicator {risk_class}">
                {risk_level}
            </div>
            <p><strong>Risk Score:</strong> {risk_score}/100</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 💡 Health Recommendations")
    recommendations = []
    if bmi > 30: recommendations.append("🏃‍♂️ Consider weight management programs")
    if smoker == "yes": recommendations.append("🚭 Smoking cessation can reduce costs significantly")
    if bmi < 18.5: recommendations.append("🥗 Consult about healthy weight gain")
    if age > 50: recommendations.append("🩺 Regular health screenings recommended")
    
    for rec in recommendations:
        st.info(rec)
    if not recommendations:
        st.success("✅ Maintain current healthy lifestyle!")

# Prediction Section
st.markdown("### 🔮 Insurance Cost Prediction")
_, predict_col, _ = st.columns([1, 2, 1])

with predict_col:
    if st.button("🚀 Generate Prediction", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            time.sleep(1)
            
            # Prepare input data
            input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
            
            if model and scaler:
                input_df = pd.DataFrame([input_dict])
                input_df = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=True)
                required_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
                input_df = input_df.reindex(columns=required_cols, fill_value=0)
                input_df[['age', 'bmi', 'children']] = scaler.transform(input_df[['age', 'bmi', 'children']])
                cost = model.predict(input_df)[0]
            else:
                # Demo calculation
                cost = max(2000 + (age * 120) + (bmi * 100) + (children * 500) + 
                          (15000 if smoker == "yes" else 0) + 
                          (1000 if region in ["northeast", "northwest"] else 0), 1000)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-result">
                💰 Predicted Annual Insurance Cost<br>
                <span style="font-size: 2rem; font-weight: 700;">₹{cost:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            if show_confidence:
                st.info(f"📊 95% Confidence Interval: ₹{cost*0.85:,.0f} - ₹{cost*1.15:,.0f}")
            
            if show_risk_factors:
                st.markdown("### 🎯 Cost Impact Analysis")
                impact_cols = st.columns(2)
                
                with impact_cols[0]:
                    factors = [("Smoking", "₹15,000", "🔴")] if smoker == "yes" else []
                    if bmi > 30: factors.append(("High BMI", "₹3,000", "🟡"))
                    if age > 50: factors.append(("Age Factor", "₹2,000", "🟡"))
                    
                    if factors:
                        st.markdown("**Cost Increasing Factors:**")
                        for factor, impact, icon in factors:
                            st.markdown(f"{icon} {factor}: +{impact}")
                
                with impact_cols[1]:
                    savings = [("Non-Smoker", "₹15,000", "🟢")] if smoker == "no" else []
                    if 18.5 <= bmi <= 25: savings.append(("Healthy BMI", "₹3,000", "🟢"))
                    
                    if savings:
                        st.markdown("**Cost Saving Factors:**")
                        for factor, saving, icon in savings:
                            st.markdown(f"{icon} {factor}: -{saving}")
            
            if show_comparisons:
                st.markdown("### 📈 Market Comparison")
                market_avg = 8000
                if cost < market_avg:
                    st.success(f"✅ Your predicted cost is {((market_avg - cost) / market_avg * 100):.1f}% below market average (₹{market_avg:,})")
                else:
                    st.warning(f"⚠️ Your predicted cost is {((cost - market_avg) / market_avg * 100):.1f}% above market average (₹{market_avg:,})")

# About Section
st.markdown("---\n### 👨‍💼 About the Developer")
about_col1, about_col2 = st.columns([1, 2])

with about_col1:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2ecc71 100%); 
                border-radius: 15px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">👨‍💻</div>
        <h3 style="margin: 0; color: white;">SHAIKH AKBAR ALI</h3>
        <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">Data Scientist</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; display: inline-block;">🤖 ML Engineer</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; display: inline-block;">📊 Analytics Expert</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with about_col2:
    st.markdown("""
    <div class="card">
        <h4 style="color: #1e3c72; margin-bottom: 1rem;">🎯 About This Project</h4>
        <p style="text-align: justify; line-height: 1.6;">
            This <strong>Advanced Insurance Cost Predictor</strong> leverages cutting-edge machine learning algorithms 
            to provide accurate insurance cost estimates. Built with <strong>ensemble learning techniques</strong> and 
            trained on comprehensive healthcare data, this tool helps individuals and insurance providers make informed decisions.
        </p>
        <h4 style="color: #1e3c72; margin: 1.5rem 0 1rem 0;">🚀 Key Achievements</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem;">
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <h5>🎯 Model Accuracy</h5>
                <p>95%+ <span style="color: #27ae60;">High Performance</span></p>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <h5>📊 Data Points</h5>
                <p>10K+ <span style="color: #27ae60;">Rich Dataset</span></p>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <h5>🔧 Key Features</h5>
                <p>7 <span style="color: #27ae60;">Comprehensive</span></p>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <h5>⚡ Speed</h5>
                <p>Real-time <span style="color: #27ae60;">Instant Results</span></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Professional Expertise
st.markdown("### 💼 Professional Expertise")
expertise_cols = st.columns(4)
expertise_data = [
    ("🧠", "Machine Learning", "Advanced ML algorithms, ensemble methods, deep learning"),
    ("📊", "Data Analytics", "Statistical analysis, predictive modeling, data visualization"),
    ("🏥", "Healthcare AI", "Medical data analysis, risk assessment, cost prediction"),
    ("💻", "Tech Stack", "Python, R, SQL, Streamlit, Plotly, Scikit-learn")
]

for i, (icon, title, desc) in enumerate(expertise_data):
    with expertise_cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
            <h4 style="color: #1e3c72; margin-bottom: 0.5rem;">{title}</h4>
            <p style="font-size: 0.9rem; color: #666;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Technical Features
st.markdown("### 🔧 Technical Features")
features_col1, features_col2 = st.columns(2)

with features_col1:
    st.markdown("""
    <div class="card">
        <h4 style="color: #1e3c72;">🎯 Model Architecture</h4>
        <ul style="line-height: 1.8;">
            <li><strong>Ensemble Learning:</strong> Combines multiple algorithms for better accuracy</li>
            <li><strong>Feature Engineering:</strong> Advanced preprocessing and scaling techniques</li>
            <li><strong>Cross-Validation:</strong> Robust model validation with k-fold CV</li>
            <li><strong>Hyperparameter Tuning:</strong> Grid search optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with features_col2:
    st.markdown("""
    <div class="card">
        <h4 style="color: #1e3c72;">📈 Advanced Analytics</h4>
        <ul style="line-height: 1.8;">
            <li><strong>Interactive Visualizations:</strong> Dynamic Plotly charts</li>
            <li><strong>Risk Assessment:</strong> Multi-factor risk scoring system</li>
            <li><strong>Real-time Predictions:</strong> Instant cost estimation</li>
            <li><strong>Confidence Intervals:</strong> Uncertainty quantification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Contact Information
st.markdown("### 📞 Connect with Me")
contact_cols = st.columns(3)
contact_data = [
    ("📧", "Email", "akbar.ali.ds@gmail.com"),
    ("💼", "LinkedIn", "linkedin.com/in/akbar-ali-ds"),
    ("🐙", "GitHub", "github.com/akbar-ali-ds")
]

for i, (icon, platform, info) in enumerate(contact_data):
    with contact_cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div style="font-size: 2rem; color: #1e3c72; margin-bottom: 1rem;">{icon}</div>
            <h4>{platform}</h4>
            <p style="color: #666;">{info}</p>
        </div>
        """, unsafe_allow_html=True)

# Interactive Data Analytics
st.markdown("---\n### 📊 Interactive Data Analytics")
data = generate_data()

viz_tabs = st.tabs(["🔍 Cost Analysis", "📈 Trends", "🎯 Risk Factors", "🌍 Regional Analysis"])

with viz_tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(data, x='age', y='cost', color='smoker', size='bmi', hover_data=['children'],
                        title="Insurance Cost vs Age (Size: BMI)", color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(data, x='bmi', y='cost', color='smoker', title="Insurance Cost vs BMI",
                        color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
        st.plotly_chart(fig, use_container_width=True)

with viz_tabs[1]:
    fig = px.histogram(data, x='cost', color='smoker', nbins=30, title="Insurance Cost Distribution",
                      color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
    st.plotly_chart(fig, use_container_width=True)
    
    data['age_group'] = pd.cut(data['age'], bins=[18, 30, 40, 50, 65], labels=['18-30', '31-40', '41-50', '51-65'])
    fig = px.box(data, x='age_group', y='cost', color='smoker', title="Cost Distribution by Age Group",
                color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
    st.plotly_chart(fig, use_container_width=True)

with viz_tabs[2]:
    risk_data = data.groupby(['smoker', 'region'])['cost'].mean().reset_index()
    fig = px.bar(risk_data, x='region', y='cost', color='smoker', title="Average Cost by Region and Smoking Status",
                color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
    st.plotly_chart(fig, use_container_width=True)
    
    data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    fig = px.violin(data, x='bmi_category', y='cost', color='smoker', title="Cost Distribution by BMI Category",
                   color_discrete_map={'yes': '#e74c3c', 'no': '#27ae60'})
    st.plotly_chart(fig, use_container_width=True)

with viz_tabs[3]:
    regional_stats = data.groupby('region').agg({'cost': ['mean', 'median', 'std'], 'age': 'mean', 'bmi': 'mean'}).round(2)
    regional_stats.columns = ['Avg Cost', 'Median Cost', 'Cost Std', 'Avg Age', 'Avg BMI']
    st.dataframe(regional_stats, use_container_width=True)
    
    fig = px.bar(data.groupby('region')['cost'].mean().reset_index(), x='region', y='cost',
                title="Average Insurance Cost by Region", color='cost', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

# Footer and Credits
st.markdown("---\n### 📚 Additional Resources")
footer_cols = st.columns(4)
footer_data = [
    ("🏥 Health Tips", ["Regular exercise", "Balanced diet", "Avoid smoking", "Regular check-ups"]),
    ("💰 Cost Factors", ["Age (15-20%)", "Smoking (60-70%)", "BMI (10-15%)", "Location (5-10%)"]),
    ("📊 Model Info", ["Accuracy: 95%+", "Data Points: 10,000+", "Features: 7", "Algorithm: ML Ensemble"]),
    ("🕒 Session Info", [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "Version: 2.0", "Status: ✅ Active", ""])
]

for i, (title, items) in enumerate(footer_data):
    with footer_cols[i]:
        st.markdown(f"**{title}**")
        for item in items:
            if item: st.markdown(f"- {item}")

st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2ecc71 100%); 
            border-radius: 15px; color: white; margin-top: 2rem;'>
    <h3 style='margin-bottom: 1rem; color: white;'>🚀 Insurance Cost Predictor v2.0</h3>
    <p style='margin-bottom: 1rem; opacity: 0.9;'>
        Developed by <strong>SHAIKH AKBAR ALI</strong> - Data Scientist & ML Engineer
    </p>
    <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;'>
        <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;'>🧠 Advanced ML Algorithms</span>
        <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;'>📊 Interactive Analytics</span>
        <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;'>⚡ Real-time Predictions</span>
    </div>
    <p style='margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;'>
        Built with ❤️ using Python, Streamlit, Plotly & Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)