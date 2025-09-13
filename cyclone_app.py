import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Label mapping based on training encoder
cyclone_levels = {0: 'No Cyclone', 1: 'Cyclone Likely'}
cyclone_emojis = {
    "No Cyclone": "✅",
    "Cyclone Likely": "🌪️"
}

# Streamlit page config
st.set_page_config(page_title="🌪️ Cyclone Predictor", page_icon="🌊", layout="centered")

# --- Title and Description ---
st.title("🌪️ Cyclone Prediction for Disaster Management")
st.markdown("""
Accurately predicting cyclones can help in **disaster preparedness** and reduce risks to life and property.  
Enter environmental measurements to check the likelihood of cyclone occurrence.
""")

st.info("Provide the environmental parameters below and click **Predict** to see the cyclone risk prediction.")

# --- Input Form ---
with st.form("input_form"):
    st.subheader("📝 Input Environmental Data")

    col1, col2 = st.columns(2)

    with col1:
        sea_surface_temp = st.number_input(
            "🌊 Sea Surface Temperature (°C)", 
            min_value=20.0, max_value=30.0, value=26.0, step=0.1
        )
        pressure = st.number_input(
            "🌡️ Atmospheric Pressure (hPa)", 
            min_value=980.0, max_value=1025.0, value=1003.0, step=0.1
        )
        humidity = st.slider(
            "💧 Humidity (%)", 
            min_value=30, max_value=100, value=67
        )
        latitude = st.number_input(
            "📍 Latitude (degrees)", 
            min_value=5.0, max_value=35.0, value=20.0, step=0.1
        )

    with col2:
        wind_shear = st.number_input(
            "💨 Wind Shear (m/s)", 
            min_value=5.0, max_value=30.0, value=16.0, step=0.1
        )
        vorticity = st.number_input(
            "🌀 Vorticity (s⁻¹)", 
            min_value=0.000001, max_value=0.000100, value=0.000039, step=0.000001, format="%.6f"
        )
        ocean_depth = st.number_input(
            "🌊 Ocean Depth (m)", 
            min_value=50.0, max_value=5000.0, value=1433.0, step=10.0
        )
        distance_coast = st.number_input(
            "🏝️ Proximity to Coastline (normalized)", 
            min_value=0.5, max_value=2.0, value=1.25, step=0.01
        )
        disturbance = st.selectbox(
            "⚡ Pre-existing Disturbance", 
            [0, 1]
        )

    submit_btn = st.form_submit_button("🚨 Predict Cyclone Likelihood")

# --- Prediction Result ---
if submit_btn:
    # Ensure the input order matches training
    input_data = np.array([[
        sea_surface_temp,
        pressure,
        humidity,
        wind_shear,
        vorticity,
        latitude,
        ocean_depth,
        distance_coast,
        disturbance
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    label = cyclone_levels.get(prediction, "Unknown")
    emoji = cyclone_emojis.get(label, "❓")

    # Background & text color
    bg_colors = {
        "No Cyclone": "#d4edda",
        "Cyclone Likely": "#f8d7da"
    }
    text_colors = {
        "No Cyclone": "#155724",
        "Cyclone Likely": "#721c24"
    }

    st.markdown("### 🔍 Prediction Result")
    st.markdown(f"""
    <div style='
        background-color: {bg_colors.get(label, "#f0f0f0")};
        color: {text_colors.get(label, "#000")};
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    '>
        {emoji} Cyclone Prediction: {label}
    </div>
    """, unsafe_allow_html=True)

    # Probabilities
    probs = model.predict_proba(input_scaled)[0]
    st.markdown("### 📊 Prediction Probabilities")
    for idx, prob in enumerate(probs):
        prob_label = cyclone_levels.get(idx, f"Class {idx}")
        st.write(f"{cyclone_emojis.get(prob_label, '')} **{prob_label}**: {prob:.2%}")

# --- Sidebar ---
st.sidebar.markdown("📘 **About the Model**")
st.sidebar.info("""
This cyclone prediction model is trained using:
- **XGBoost Classifier**
- **Features:** Sea Surface Temp, Atmospheric Pressure, Humidity, Wind Shear, Vorticity, Latitude, Ocean Depth, Proximity to Coastline, Pre-existing Disturbance
- **Labels:** Cyclone Likely / No Cyclone
""")

