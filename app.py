import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Smart City Traffic AI", page_icon="🚦", layout="wide")

# Custom CSS for an executive dashboard appearance
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #4F46E5; color: white; font-weight: bold; }
    div.stForm { background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    h2 { color: #2563EB; }
    </style>
""", unsafe_allow_html=True)  # <-- Changed from unsafe_style to unsafe_allow_html

# ==========================================
# 2. INITIALIZE SESSION STATE SECURITY MEMORY
# ==========================================
if "user_db" not in st.session_state:
    st.session_state.user_db = {"admin": "admin123"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ==========================================
# 3. LOAD THE TRAINED BRAIN (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('traffic_model.pkl')

model = load_model()

# ==========================================
# 4. AUTHENTICATION GATEWAY (LOGIN / SIGNUP)
# ==========================================
if not st.session_state.logged_in:
    st.title("🚦 Smart City Traffic Analytics Portal")
    st.markdown("### Secure Department Gateway")
    
    auth_mode = st.tabs(["🔒 Account Login", "📝 Create Official Account"])
    
    # --- TAB 1: LOGIN ---
    with auth_mode[0]:
        with st.form("login_form"):
            st.subheader("Login Credentials")
            username = st.text_input("Username / Badge ID").strip()
            password = st.text_input("Password", type="password").strip()
            submit_login = st.form_submit_button("Access Portal")
            
            if submit_login:
                if username in st.session_state.user_db and st.session_state.user_db[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success(f"🔓 Access granted. Welcome back, Officer {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Please check credentials.")
                    
    # --- TAB 2: SIGN UP ---
    with auth_mode[1]:
        with st.form("signup_form"):
            st.subheader("Register New Authority Account")
            new_username = st.text_input("Choose Username / Badge ID").strip()
            new_password = st.text_input("Create Secure Password", type="password").strip()
            confirm_password = st.text_input("Confirm Secure Password", type="password").strip()
            submit_signup = st.form_submit_button("Register Account")
            
            if submit_signup:
                if not new_username or not new_password:
                    st.warning("⚠️ Fields cannot be blank.")
                elif new_username in st.session_state.user_db:
                    st.error("❌ Username already registered in system database.")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    st.session_state.user_db[new_username] = new_password
                    st.success("✅ Registration successful! Proceed to the Login tab.")

# ==========================================
# 5. PROTECTED SYSTEM APPARATUS
# ==========================================
else:
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3180/3180111.png", width=80)
    st.sidebar.title("City Traffic Center")
    st.sidebar.write(f"👤 **Active Operator:** {st.session_state.current_user}")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio(
        "Select Dashboard Mode:",
        ["🏠 Individual Query Predictor", "📊 Regional Batch Forecasting & Reporting"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info("🤖 **AI Model Status:** Online\n\n🎯 **Validated Accuracy:** 96.9%")
    
    if st.sidebar.button("🚪 Log Out / Terminate Session"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

    # --- MODE 1: INDIVIDUAL QUERY PREDICTOR ---
    if app_mode == "🏠 Individual Query Predictor":
        st.title("🚦 Smart City Traffic Estimator")
        st.markdown("Select specific operational parameters below to evaluate traffic density instantly.")

        col1, col2, col3 = st.columns(3)
        with col1:
            junction = st.selectbox("🎯 Target Junction", [1, 2, 3, 4])
            month = st.slider("📅 Calendar Month", 1, 12, 6)
        with col2:
            day = st.slider("📆 Day of Month", 1, 31, 15)
            day_of_week = st.selectbox("🗓️ Day of Week", [0,1,2,3,4,5,6], 
                            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        with col3:
            hour = st.slider("⏰ Operational Hour (24h)", 0, 23, 17)
            year = st.selectbox("📆 Evaluation Year", [2017, 2018, 2019, 2020])

        st.markdown("###")
        if st.button("🔮 Calculate Traffic Density"):
            input_data = pd.DataFrame({
                'Junction': [junction], 'Hour': [hour], 'Day': [day],
                'Month': [month], 'Year': [year], 'DayOfWeek': [day_of_week]
            })
            
            prediction = model.predict(input_data)[0]
            vehicles_count = int(prediction.round())
            
            st.success(f"### 🚗 Expected Traffic Volume: **{vehicles_count} Vehicles**")
            
            if vehicles_count > 100:
                st.warning("⚠️ **Operational Alert:** High congestion zone. Deploy peak-hour management strategies.")
            else:
                st.info("✅ **Operational Status:** Normal flow. Standard traffic patterns expected.")

    # --- MODE 2: BATCH FORECASTING & EXCEL REPORTS (ZERO FILE CLUTTER) ---
    elif app_mode == "📊 Regional Batch Forecasting & Reporting":
        st.title("📈 Strategic Regional Traffic Forecast Engine")
        st.markdown("Upload future operational planning templates to run rapid batch projections and auto-generate executive visual reports.")

        uploaded_file = st.file_uploader("📥 Drop your future timeline dataset here:", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                required_cols = ['DateTime', 'Junction']
                if not all(col in df.columns for col in required_cols):
                    st.error("❌ **Format Conflict:** File must contain 'DateTime' and 'Junction' columns.")
                else:
                    with st.spinner("⚡ AI processing timeline logic... Please hold."):
                        df['DateTime'] = pd.to_datetime(df['DateTime'])
                        df['Hour'] = df['DateTime'].dt.hour
                        df['Day'] = df['DateTime'].dt.day
                        df['Month'] = df['DateTime'].dt.month
                        df['Year'] = df['DateTime'].dt.year
                        df['DayOfWeek'] = df['DateTime'].dt.dayofweek

                        feature_cols = ['Junction', 'Hour', 'Day', 'Month', 'Year', 'DayOfWeek']
                        predictions = model.predict(df[feature_cols])
                        df['Predicted Vehicles'] = predictions.round().astype(int)

                        display_cols = ['DateTime', 'Junction', 'Predicted Vehicles']
                        if 'ID' in df.columns: display_cols.append('ID')
                        final_df = df[display_cols]

                    st.success("🎯 Batch pipeline computed successfully!")

                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown("### 📋 Generated Forecast Data")
                        st.dataframe(final_df.head(100), height=380, use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        final_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Data Report (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="executive_traffic_forecast.csv",
                            mime="text/csv"
                        )

                    with col_right:
                        st.markdown("### 📊 Infrastructure Planning Analysis")
                        
                        fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                        sns.barplot(data=final_df, x='Junction', y='Predicted Vehicles', estimator='mean', hue='Junction', palette='Blues_d', ax=ax1, legend=False)
                        ax1.set_title("Average Predicted Congestion by Junction Location", fontsize=10)
                        ax1.set_xlabel("Junction Index", fontsize=8)
                        ax1.set_ylabel("Mean Predicted Traffic", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig1)
                        plt.close(fig1) # Discards plot cleanly from memory layout context

                        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                        hourly_trends = df.groupby('Hour')['Predicted Vehicles'].mean().reset_index()
                        sns.lineplot(data=hourly_trends, x='Hour', y='Predicted Vehicles', marker='o', color='#4F46E5', ax=ax2)
                        ax2.axhline(df['Predicted Vehicles'].mean() * 1.3, color='red', linestyle='--', label='Congestion Threshold')
                        ax2.set_title("City-Wide Peak Rush Hour Allocations", fontsize=10)
                        ax2.set_xlabel("Hour of the day", fontsize=8)
                        ax2.set_ylabel("Predicted Vehicles Volume", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2) # Discards plot cleanly from memory layout context

                    st.info("💡 **Pro-Tip:** Hover over the top right corner of any chart to save the visual graphics instantly as a high-resolution image file.")

            except Exception as e:
                st.error(f"⚠️ **Execution Error:** {str(e)}")