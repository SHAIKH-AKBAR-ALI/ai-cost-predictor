"""
Modern Insurance Cost Predictor - Premium UI/UX Design
=====================================================

A visually stunning, modern insurance cost prediction application with:
- Glassmorphism design with frosted glass effects
- Smooth animations and micro-interactions
- Dark/Light theme toggle
- Real-time predictions and comparisons
- Premium dashboard with interactive visualizations

Author: Insurance ML Team
Version: 3.0 - Modern UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import json

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="InsuranceAI - Premium Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
def init_session_state():
    """Initialize session state for theme and user data."""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {
            'age': 30, 'sex': 'male', 'bmi': 25.0,
            'children': 0, 'smoker': 'no', 'region': 'northeast'
        }
    if 'comparison_scenarios' not in st.session_state:
        st.session_state.comparison_scenarios = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'predictor'

# Modern CSS with glassmorphism and animations
def load_modern_css():
    """Load modern CSS with glassmorphism effects and animations."""
    theme_colors = {
        'dark': {
            'primary': '#6366f1',
            'secondary': '#8b5cf6', 
            'accent': '#06b6d4',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'bg_primary': '#0f172a',
            'bg_secondary': 'rgba(30, 41, 59, 0.8)',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1'
        },
        'light': {
            'primary': '#6366f1',
            'secondary': '#8b5cf6',
            'accent': '#06b6d4',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'bg_primary': '#ffffff',
            'bg_secondary': 'rgba(248, 250, 252, 0.8)',
            'text_primary': '#1e293b',
            'text_secondary': '#64748b'
        }
    }
    
    colors = theme_colors[st.session_state.theme]
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 50%, {colors['accent']} 100%);
        min-height: 100vh;
    }}
    
    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Glassmorphism Cards */
    .glass-card {{
        background: {colors['bg_secondary']};
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }}
    
    /* Prediction Result Card */
    .prediction-card {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        color: white;
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
        animation: slideInRight 0.8s ease-out;
    }}
    
    .prediction-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }}
    
    /* Health Score Gauge */
    .health-gauge {{
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #ef4444 0deg, #f59e0b 120deg, #10b981 240deg, #10b981 360deg);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2rem auto;
        position: relative;
        animation: rotateIn 1s ease-out;
    }}
    
    .health-gauge::before {{
        content: '';
        width: 160px;
        height: 160px;
        background: {colors['bg_primary']};
        border-radius: 50%;
        position: absolute;
    }}
    
    .health-score {{
        font-size: 2rem;
        font-weight: 700;
        color: {colors['text_primary']};
        z-index: 1;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: {colors['bg_secondary']};
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .metric-label {{
        color: {colors['text_secondary']};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideInRight {{
        from {{
            opacity: 0;
            transform: translateX(50px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes rotateIn {{
        from {{
            opacity: 0;
            transform: rotate(-180deg) scale(0.5);
        }}
        to {{
            opacity: 1;
            transform: rotate(0deg) scale(1);
        }}
    }}
    
    @keyframes shimmer {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .glass-card {{
            padding: 1.5rem;
            margin: 0.5rem 0;
        }}
        
        .prediction-card {{
            padding: 2rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Data loading functions
@st.cache_data(show_spinner=False)
def load_insurance_data():
    """Load insurance dataset."""
    try:
        if Path("testing.ipynb/insurance.csv").exists():
            return pd.read_csv("testing.ipynb/insurance.csv")
        elif Path("insurance.csv").exists():
            return pd.read_csv("insurance.csv")
        else:
            # Generate sample data
            np.random.seed(42)
            data = {
                'age': np.random.randint(18, 65, 1000),
                'sex': np.random.choice(['male', 'female'], 1000),
                'bmi': np.clip(np.random.normal(28, 5, 1000), 15, 50),
                'children': np.random.randint(0, 6, 1000),
                'smoker': np.random.choice(['yes', 'no'], 1000, p=[0.2, 0.8]),
                'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], 1000),
                'expenses': np.abs(np.random.normal(13000, 12000, 1000))
            }
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model."""
    try:
        if Path("best_xgboost_model.pkl").exists():
            with open('best_xgboost_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            # Create fallback model
            df = load_insurance_data()
            le_sex = LabelEncoder()
            le_smoker = LabelEncoder()
            le_region = LabelEncoder()
            
            df_processed = df.copy()
            df_processed['sex'] = le_sex.fit_transform(df_processed['sex'])
            df_processed['smoker'] = le_smoker.fit_transform(df_processed['smoker'])
            df_processed['region'] = le_region.fit_transform(df_processed['region'])
            
            X = df_processed.drop('expenses', axis=1)
            y = df_processed['expenses']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            model.le_sex = le_sex
            model.le_smoker = le_smoker
            model.le_region = le_region
            model.is_fallback = True
            
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_cost(model, inputs):
    """Make prediction with error handling."""
    try:
        if hasattr(model, 'is_fallback') and model.is_fallback:
            input_array = np.array([[
                inputs['age'],
                model.le_sex.transform([inputs['sex']])[0],
                inputs['bmi'],
                inputs['children'],
                model.le_smoker.transform([inputs['smoker']])[0],
                model.le_region.transform([inputs['region']])[0]
            ]])
            return model.predict(input_array)[0]
        else:
            input_data = pd.DataFrame({
                'age': [inputs['age']],
                'sex': [inputs['sex']],
                'bmi': [inputs['bmi']],
                'children': [inputs['children']],
                'smoker': [inputs['smoker']],
                'region': [inputs['region']]
            })
            return model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0

def calculate_health_score(inputs):
    """Calculate health score based on inputs."""
    score = 100
    
    # Age factor
    if inputs['age'] > 50:
        score -= 15
    elif inputs['age'] < 30:
        score += 5
    
    # BMI factor
    if inputs['bmi'] > 30:
        score -= 25
    elif 18.5 <= inputs['bmi'] <= 25:
        score += 10
    
    # Smoking factor
    if inputs['smoker'] == 'yes':
        score -= 30
    else:
        score += 15
    
    # Children factor
    if inputs['children'] > 3:
        score -= 5
    
    return max(0, min(100, score))

def render_navigation():
    """Render modern navigation using Streamlit buttons."""
    pages = {
        'predictor': '🏠 Predictor',
        'dashboard': '📊 Dashboard', 
        'analytics': '📈 Analytics',
        'comparison': '🔄 Compare'
    }
    
    # Create navigation using columns
    nav_cols = st.columns(len(pages))
    
    for i, (key, label) in enumerate(pages.items()):
        with nav_cols[i]:
            if st.button(label, key=f'nav_{key}', use_container_width=True):
                st.session_state.current_page = key
                st.rerun()

def render_theme_toggle():
    """Render theme toggle button."""
    theme_icon = '🌙' if st.session_state.theme == 'light' else '☀️'
    
    if st.button(theme_icon, key='theme_toggle', help='Toggle theme'):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

def render_predictor_page():
    """Render the main predictor page with modern UI."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Personal Information")
        
        # Age slider
        age = st.slider(
            "Age", 18, 65, st.session_state.user_inputs['age'],
            help="Your current age affects insurance premiums significantly"
        )
        
        # Gender selection
        sex = st.selectbox("Gender", ["male", "female"], 
                          index=0 if st.session_state.user_inputs['sex'] == 'male' else 1)
        
        # BMI input
        bmi = st.number_input(
            "BMI (Body Mass Index)", 15.0, 50.0, 
            st.session_state.user_inputs['bmi'], 0.1,
            help="BMI = weight(kg) / height(m)²"
        )
        
        # Children selector
        children = st.selectbox(
            "Number of Children", 
            list(range(6)),
            index=st.session_state.user_inputs['children']
        )
        
        # Smoking status
        smoker = st.selectbox("Smoking Status", ["no", "yes"],
                             index=0 if st.session_state.user_inputs['smoker'] == 'no' else 1)
        
        # Region
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"],
            index=["northeast", "northwest", "southeast", "southwest"].index(st.session_state.user_inputs['region'])
        )
        
        # Update session state
        st.session_state.user_inputs.update({
            'age': age, 'sex': sex, 'bmi': bmi,
            'children': children, 'smoker': smoker, 'region': region
        })
        
        # BMI Category with updated colors
        if bmi < 18.5:
            bmi_category, bmi_class = "Underweight", "bmi-underweight"
        elif bmi < 25:
            bmi_category, bmi_class = "Normal", "bmi-normal"
        elif bmi < 30:
            bmi_category, bmi_class = "Overweight", "bmi-overweight"
        else:
            bmi_category, bmi_class = "Obese", "bmi-obese"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div style="font-size: 1.2rem; margin-bottom: 1rem;">BMI Category</div>
            <div class="{bmi_class}" style="font-size: 2rem; font-weight: bold;">{bmi_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        # Predict button
        if st.button("🔮 Predict Insurance Cost", key='predict_btn', use_container_width=True):
            with st.spinner("Calculating your personalized prediction..."):
                time.sleep(1)  # Simulate processing
                
                model = load_model()
                if model:
                    prediction = predict_cost(model, st.session_state.user_inputs)
                    health_score = calculate_health_score(st.session_state.user_inputs)
                    
                    # Animated prediction card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="font-size: 1.2rem; margin-bottom: 1rem; opacity: 0.9;">
                            💰 Estimated Annual Cost
                        </div>
                        <div style="font-size: 3rem; font-weight: 700; margin: 1rem 0;">
                            ${prediction:,.0f}
                        </div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">
                            Range: ${prediction*0.85:,.0f} - ${prediction*1.15:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Health Score Gauge
                    st.markdown(f"""
                    <div class="health-gauge">
                        <div class="health-score">{health_score}</div>
                    </div>
                    <div style="text-align: center; margin-top: 1rem;">
                        <strong>Health Score: {health_score}/100</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factors
                    st.markdown("### 📋 Risk Analysis")
                    
                    risk_factors = []
                    if smoker == "yes":
                        risk_factors.append("🚬 Smoking significantly increases costs")
                    if bmi > 30:
                        risk_factors.append("⚖️ High BMI increases health risks")
                    if age > 50:
                        risk_factors.append("👴 Advanced age increases medical needs")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(factor)
                    else:
                        st.success("✅ Excellent health profile!")
                    
                    # Add to comparison
                    if st.button("➕ Add to Comparison", key='add_comparison'):
                        scenario = st.session_state.user_inputs.copy()
                        scenario['prediction'] = prediction
                        scenario['health_score'] = health_score
                        st.session_state.comparison_scenarios.append(scenario)
                        st.success("Added to comparison!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard_page():
    """Render interactive dashboard."""
    df = load_insurance_data()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Insurance Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${df['expenses'].mean():,.0f}</div>
            <div class="metric-label">Average Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${df['expenses'].max():,.0f}</div>
            <div class="metric-label">Maximum Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        smoker_pct = (df['smoker'] == 'yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{smoker_pct:.1f}%</div>
            <div class="metric-label">Smokers</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='expenses', nbins=50, title="Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='smoker', y='expenses', title="Cost by Smoking Status")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_analytics_page():
    """Render analytics page."""
    df = load_insurance_data()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='age', y='expenses', color='smoker', title="Age vs Insurance Cost")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(df, x='region', y='bmi', title="BMI Distribution by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_comparison_page():
    """Render scenario comparison."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 Scenario Comparison")
    
    if st.session_state.comparison_scenarios:
        cols = st.columns(len(st.session_state.comparison_scenarios))
        
        for i, scenario in enumerate(st.session_state.comparison_scenarios):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4>Scenario {i+1}</h4>
                    <div class="metric-value">${scenario['prediction']:,.0f}</div>
                    <div class="metric-label">Annual Cost</div>
                    <hr style="border-color: rgba(255,255,255,0.2);">
                    <p><strong>Age:</strong> {scenario['age']}</p>
                    <p><strong>BMI:</strong> {scenario['bmi']}</p>
                    <p><strong>Smoker:</strong> {scenario['smoker']}</p>
                    <p><strong>Health Score:</strong> {scenario.get('health_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All Comparisons"):
            st.session_state.comparison_scenarios = []
            st.rerun()
    else:
        st.info("No scenarios to compare. Add predictions from the main page!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    init_session_state()
    load_modern_css()
    
    # Theme toggle
    render_theme_toggle()
    
    # Main title
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; 
                   color: white;
                   animation: fadeInUp 1s ease-out;">
            InsuranceAI
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.8; animation: fadeInUp 1.2s ease-out;">
            Premium Healthcare Cost Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    render_navigation()
    
    # Page routing
    if st.session_state.current_page == 'predictor':
        render_predictor_page()
    elif st.session_state.current_page == 'dashboard':
        render_dashboard_page()
    elif st.session_state.current_page == 'analytics':
        render_analytics_page()
    elif st.session_state.current_page == 'comparison':
        render_comparison_page()
    else:
        render_predictor_page()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; 
                opacity: 0.6; border-top: 1px solid rgba(255,255,255,0.1);">
        <p>🏥 InsuranceAI v3.0 | Powered by Advanced Machine Learning</p>
        <p>✨ Modern UI with Glassmorphism Design</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()