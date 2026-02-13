"""
Streamlit Application for Engine Predictive Maintenance
"""

import streamlit as st
import pandas as pd
import os
import sys

# Print to console (will show in HF Space logs)
print("=" * 70, file=sys.stderr)
print("APP STARTING - INITIALIZATION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Page Configuration MUST be first
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
try:
    print("Importing huggingface_hub...", file=sys.stderr)
    from huggingface_hub import hf_hub_download, login
    print("Importing joblib...", file=sys.stderr)
    import joblib
    print("✓ All imports successful", file=sys.stderr)
except Exception as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    st.error(f"Import failed: {e}")
    st.stop()

# CRITICAL: Feature columns must EXACTLY match model training
FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 10px;
}
.sub-header {
    font-size: 18px;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}
.normal {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #c3e6cb;
}
.maintenance {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model from Hugging Face with detailed logging and retries"""
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("LOADING MODEL FROM HUGGING FACE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # CORRECT: Use HF_TOKEN (as configured in your HF Space secrets)
            hf_token = os.environ.get("HF_TOKEN")
            print(f"HF_TOKEN found: {hf_token is not None}", file=sys.stderr)
            
            if hf_token:
                print("Authenticating with Hugging Face...", file=sys.stderr)
                login(token=hf_token)
                print("✓ Authentication successful", file=sys.stderr)
            else:
                print("⚠ No HF_TOKEN - attempting public access", file=sys.stderr)
            
            # Download model
            print("\nDownloading model...", file=sys.stderr)
            print("  Repo: Quantum9999/xgb-predictive-maintenance", file=sys.stderr)
            print("  File: xgb_tuned_model.joblib", file=sys.stderr)
            
            model_path = hf_hub_download(
                repo_id="Quantum9999/xgb-predictive-maintenance",
                filename="xgb_tuned_model.joblib",
                token=hf_token,
                cache_dir="/tmp/hf_cache"  # Use tmp for faster access
            )
            print(f"✓ Model downloaded: {model_path}", file=sys.stderr)
            
            # Load model
            print("Loading model into memory...", file=sys.stderr)
            model = joblib.load(model_path)
            print("✓ Model loaded successfully", file=sys.stderr)
            
            # Verify model features
            if hasattr(model, 'feature_names_in_'):
                print(f"Model expects features: {model.feature_names_in_}", file=sys.stderr)
            
            print("=" * 70 + "\n", file=sys.stderr)
            
            return model, None
            
        except Exception as e:
            retry_count += 1
            error_msg = f"Model loading attempt {retry_count}/{max_retries} failed: {str(e)}"
            print(f"✗ {error_msg}", file=sys.stderr)
            
            if retry_count < max_retries:
                import time
                wait_time = 2 * retry_count
                print(f"Retrying in {wait_time} seconds...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                import traceback
                print(f"Final traceback:\n{traceback.format_exc()}", file=sys.stderr)
                print("=" * 70 + "\n", file=sys.stderr)
                return None, error_msg


def main():
    """Main application"""
    
    print("Starting main application...", file=sys.stderr)
    
    # Header
    st.markdown(
        '<div class="main-header">🔧 Engine Predictive Maintenance System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">AI-powered engine health monitoring & failure prediction</div>',
        unsafe_allow_html=True
    )

    # Load model with progress indicator
    with st.spinner("Loading AI model... This may take a moment."):
        model, error = load_model()
    
    if model is None:
        st.error(f"❌ Failed to load prediction model")
        st.code(error)
        
        with st.expander("🔍 Troubleshooting"):
            st.write("**Possible Issues:**")
            st.write("1. HF_TOKEN not set in Space secrets")
            st.write("2. Model repository is private")
            st.write("3. Model filename is incorrect")
            st.write("4. Network connectivity issue")
            
            st.write("\n**Current Configuration:**")
            st.write(f"- HF_TOKEN set: {os.environ.get('HF_TOKEN') is not None}")
            st.write("- Expected repo: Quantum9999/xgb-predictive-maintenance")
            st.write("- Expected file: xgb_tuned_model.joblib")
            
            st.write("\n**Your Setup (from screenshots):**")
            st.write("✅ HF Space has HF_TOKEN secret (Image 1)")
            st.write("✅ GitHub has HF_EN_TOKEN secret (Image 2)")
            st.write("✅ GitHub token for pushing code (Image 3)")
            
            st.write("\n**Next Steps:**")
            st.write("1. Verify HF_TOKEN secret exists in Space settings")
            st.write("2. Check Space logs for detailed error messages")
            st.write("3. Ensure model repo is accessible")
        
        st.stop()
    
    st.success("✓ Model loaded successfully!")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "This application predicts engine maintenance needs using "
            "machine learning analysis of 6 critical sensor parameters."
        )
        
        st.header("📊 Model Information")
        st.markdown("""
        - **Algorithm**: XGBoost Classifier
        - **Features**: 6 sensor readings
        - **Target Classes**: 
          - 0: Normal Operation
          - 1: Maintenance Required
        - **Training Data**: 19,535 records
        """)
        
        st.header("🎯 How to Use")
        st.markdown("""
        1. Enter current sensor readings
        2. Click 'Predict Engine Condition'
        3. Review prediction and confidence
        4. Take action based on results
        """)
        
        st.header("📈 Sensor Ranges")
        st.markdown("""
        **Normal Operating Ranges:**
        - RPM: 161 - 2,239
        - Lub Oil Pressure: 0.003 - 7.3 bar
        - Fuel Pressure: 0.003 - 21.1 bar
        - Coolant Pressure: 0.002 - 7.5 bar
        - Lub Oil Temp: 71 - 90 °C
        - Coolant Temp: 62 - 196 °C
        """)

    # Main content
    st.header("📝 Enter Engine Sensor Readings")
    st.markdown("---")

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Speed & Pressure Sensors")
        
        engine_rpm = st.number_input(
            "Engine RPM (Revolutions per Minute)",
            min_value=100.0,
            max_value=2500.0,
            value=791.0,
            step=10.0,
            help="Engine speed - Normal range: 161-2,239 RPM"
        )
        
        lub_oil_pressure = st.number_input(
            "Lubrication Oil Pressure (bar)",
            min_value=0.0,
            max_value=10.0,
            value=3.3,
            step=0.1,
            help="Lubricating oil pressure - Normal range: 0.003-7.266 bar"
        )
        
        fuel_pressure = st.number_input(
            "Fuel Pressure (bar)",
            min_value=0.0,
            max_value=25.0,
            value=6.7,
            step=0.1,
            help="Fuel delivery pressure - Normal range: 0.003-21.138 bar"
        )

    with col2:
        st.subheader("🌡️ Temperature & Coolant Sensors")
        
        coolant_pressure = st.number_input(
            "Coolant Pressure (bar)",
            min_value=0.0,
            max_value=10.0,
            value=2.3,
            step=0.1,
            help="Coolant system pressure - Normal range: 0.002-7.479 bar"
        )
        
        lub_oil_temp = st.number_input(
            "Lubrication Oil Temperature (°C)",
            min_value=60.0,
            max_value=100.0,
            value=77.6,
            step=0.5,
            help="Lubricating oil temperature - Normal range: 71.3-89.6 °C"
        )
        
        coolant_temp = st.number_input(
            "Coolant Temperature (°C)",
            min_value=50.0,
            max_value=200.0,
            value=78.4,
            step=0.5,
            help="Engine coolant temperature - Normal range: 61.7-195.5 °C"
        )

    # Prediction button
    st.markdown("---")
    
    if st.button("🔍 Predict Engine Condition", use_container_width=True, type="primary"):
        # Create input DataFrame with exact column names
        input_df = pd.DataFrame([{
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp
        }])

        try:
            print(f"Making prediction with input: {input_df.to_dict()}", file=sys.stderr)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            print(f"Prediction: {prediction}, Probabilities: {proba}", file=sys.stderr)

            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Result")

            if prediction == 0:
                st.markdown(
                    '<div class="prediction-box normal">✅ Engine Operating Normally</div>',
                    unsafe_allow_html=True
                )
                st.success("✓ No maintenance required at this time. Engine is functioning within normal parameters.")
            else:
                st.markdown(
                    '<div class="prediction-box maintenance">⚠️ Maintenance Required</div>',
                    unsafe_allow_html=True
                )
                st.warning("⚠ Engine shows signs of potential failure. Schedule maintenance as soon as possible to prevent breakdown.")

            # Confidence scores
            st.subheader("📊 Prediction Confidence")
            
            conf_col1, conf_col2 = st.columns(2)
            
            with conf_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Normal Operation Probability",
                    value=f"{proba[0]:.2%}",
                    help="Confidence that engine is operating normally"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with conf_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Maintenance Required Probability",
                    value=f"{proba[1]:.2%}",
                    help="Confidence that engine requires maintenance"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Input summary
            with st.expander("📋 View Input Summary"):
                st.dataframe(
                    input_df.T.rename(columns={0: "Value"}),
                    use_container_width=True
                )
                
            # Recommendations
            with st.expander("💡 Recommendations"):
                if prediction == 0:
                    st.markdown("""
                    **Current Status: Healthy**
                    - Continue regular monitoring
                    - Maintain current maintenance schedule
                    - Monitor for any sudden changes in sensor readings
                    - Schedule next routine inspection as planned
                    """)
                else:
                    st.markdown("""
                    **Immediate Actions Required:**
                    - Schedule comprehensive engine inspection
                    - Check lubrication system
                    - Inspect cooling system
                    - Review fuel delivery system
                    - Monitor engine closely until serviced
                    - Consider reducing operational load
                    """)
        
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            print(f"✗ {error_msg}", file=sys.stderr)
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            
            st.error(f"❌ {error_msg}")
            st.info("Please verify all sensor values are within valid ranges and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 14px;'>"
        "🤖 Built with XGBoost & Streamlit | 🤗 Model hosted on Hugging Face<br>"
        "Developed as part of ML Deployment & Automation Project"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    print("Entering main()...", file=sys.stderr)
    try:
        main()
        print("✓ Main completed successfully", file=sys.stderr)
    except Exception as e:
        print(f"✗ FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        st.error(f"Application error: {e}")
