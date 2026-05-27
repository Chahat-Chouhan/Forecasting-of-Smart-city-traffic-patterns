import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time
import datetime
import holidays

# ==========================================
# 1. UI CONFIGURATION & THEME STYLING
# ==========================================

st.set_page_config(page_title="Smart City Traffic AI",page_icon="🚦",layout="wide")

# Custom CSS for a sleek, aesthetic "Smart City Command Center" vibe
st.markdown("""
    <style>
        .reportview-container {
        background: #0f172a;
        }
        .stButton>button {
        background-color: #0284c7;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        width: 100%;
        }
        .stButton>button:hover {
        background-color: #38bdf8;
        color: black;
        }
    
        /* --- METRIC CARD FIXES --- */
        .metric-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
        color: #f8fafc !important; /* Forces default card text to off-white */
        }
        .metric-box b {
        color: #94a3b8 !important; /* Light gray-blue for the titles (e.g. "Daily Peak Volume") */
        font-size: 1.1rem;
        }
        .metric-box h1 {
        color: #06b6d4 !important; /* Bright neon cyan for large numbers */
        font-weight: bold;
        margin-top: 10px;
        }
        .metric-box h2 {
        color: #38bdf8 !important; /* Bright neon blue for standard headers */
        font-weight: bold;
        margin-top: 10px;
        }
        .metric-box h4 {
        color: #e2e8f0 !important; /* Clean silver for the holiday status */
        margin-top: 10px;
        }
        .metric-box p {
        color: #64748b !important;
        margin: 0;
        }

        /* --- CHATBOT CONTAINER FIXES --- */
        .chat-box {
        background-color: #1e293b;
        color: #e2e8f0 !important; /* High contrast text for the chatbot paragraph */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #06b6d4;
        margin-top: 10px;
        line-height: 1.6;
        }
        .chat-box strong, .chat-box b {
        color: #38bdf8 !important; /* Highlights chatbot alert headers in neon blue */
        }
    </style>
    """, unsafe_allow_html=True)


# Initialize Session States for Authentication
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'viewing_signup' not in st.session_state:
    st.session_state['viewing_signup'] = False

# Hardcoded User Database for demonstration
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {"admin": "password123", "planner": "smartcity2026"}


# ==========================================
# 2. MODEL & UTILITY FUNCTIONS
# ==========================================

@st.cache_resource
def load_model():
    try:
        return joblib.load('traffic_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ 'traffic_model.pkl' not found. Please run your backend training script first!")
        return None
    

model = load_model()
local_holidays = holidays.country_holidays('IN')  # Change country code if needed

def predict_24h_traffic(junction, selected_date):
    """Generates traffic predictions for all 24 hours of a given date."""
    data_list =[]
    for hour in range(24):
        day = selected_date.day
        month = selected_date.month
        year = selected_date.year
        day_of_week = selected_date.weekday() # 0=Mon, 6=Sun
        is_holiday = 1 if selected_date in local_holidays else 0

        data_list.append([junction, hour, day, month, year, day_of_week, is_holiday])

    columns = ['Junction','Hour','Day','Month','Year','DayOfWeek','Is_Holiday']
    df_features = pd.DataFrame(data_list, columns=columns)

    # Run predictions
    predictions = model.predict(df_features)
    df_features['Predicted_Vehicles'] = predictions.round().astype(int) # Round to nearest integer for vehicle counts
    return df_features


# ==========================================
# 3. AUTHENTICATION INTERFACE (GATEWAY)
# ==========================================
if not st.session_state['logged_in']:
    st.title("🚦 Smart City AI Command Center")
    st.subheader("Government Traffic Forecasting & Infrastructure Planning Portal")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if not st.session_state['viewing_signup']:
            st.markdown("<div class='metric-box'><h3>🔑 Portal Login</h3>", unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username in st.session_state['user_db'] and st.session_state['user_db'][username] == password:
                    st.session_state['logged_in'] = True
                    st.success(f"Welcome back, officer {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

            if st.button("Need an account? Sign Up Here"):
                st.session_state['viewing_signup'] = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("<div class='metric-box'><h3>📝 Register New Official Account</h3>", unsafe_allow_html=True)
            new_username = st.text_input("Create Username")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Register Account"):
                if not new_username or not new_password:
                    st.error("Fields cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif new_username in st.session_state['user_db']:
                    st.error("Username already exists.")
                else:
                    st.session_state['user_db'][new_username] = new_password
                    st.success(f"Account created successfully! Please log in.")
                    st.session_state['viewing_signup'] = False
                    st.rerun()

            if st.button("Back to Login"):
                st.session_state['viewing_signup'] = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. MAIN DASHBOARD AREA (AUTHENTICATED)
# ==========================================
else:
    # Sidebar control
    st.sidebar.title("👮 Executive Controls")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    
    st.sidebar.write("---")
    st.sidebar.markdown("### 🗓️ Forecast Scope")

    # Primary interactive fields
    target_junction = st.sidebar.selectbox("Select Target Junction", [1, 2, 3, 4])
    target_date = st.sidebar.date_input("Select Forecast Date", datetime.date.today())
    danger_threshold = st.sidebar.slider("🚨 General Warning Threshold (Vehicles)", 30, 150, 80)

    # NEW FEATURE: Hour selection slider in the sidebar
    target_hour = st.sidebar.slider("⏰ Select Hour for Micro-Analysis", 0, 23, 17)

    # Dashboard layout header
    st.title("📈 Live Traffic Intelligence & Recommendation Dashboard")
    st.markdown(f"**Analyzing Junction {target_junction} for target calendar date: `{target_date}`**")

    if model is not None:
        # Compute predictions
        with st.spinner("🤖 Consulting AI Engine for real-time patterns..."):
            df_forecast = predict_24h_traffic(target_junction, target_date)
            time.sleep(0.2)

        # Performance Summary Cards
        peak_volume = df_forecast['Predicted_Vehicles'].max()
        peak_hour = df_forecast.loc[df_forecast['Predicted_Vehicles'].idxmax(), 'Hour']
        is_holiday = "Yes (Special Traffic Schedule)" if df_forecast['Is_Holiday'].iloc[0] == 1 else "No (Standard Business Day)"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-box'>🏁 <b>Daily Peak Volume</b><h2>{peak_volume} vph</h2></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'>⏰ <b>Daily Peak Hour</b><h2>{peak_hour}:00</h2></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-box'>📅 <b>Official Holiday?</b><h4>{is_holiday}</h4></div>", unsafe_allow_html=True)

        st.write("---")

        col_gragh, col_compare = st.columns([2,1])

        with col_gragh:
            st.subheader("🚨 24-Hour Macro Trend Tracking")

            # Line graph showing the 24 hour trend
            fig_line = px.line(df_forecast, x='Hour', y='Predicted_Vehicles',
                               labels={'Hour':'Hour of the Day', 'Predicted_Vehicles':'Vehicle Count'},
                               markers=True, template="plotly_dark")
            
            fig_line.add_hline(y=danger_threshold, line_dash="dash", line_color="red",
                               annotation_text="🚨 Traffic Saturation Limit", annotation_position="top left")
            fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_line, use_container_width=True)

            with col_compare:
                st.subheader("🚦 Cross-Junction Impact")
                junction_comparison = []
                for j in [1, 2, 3, 4]:
                    df_j = predict_24h_traffic(j, target_date)
                    junction_comparison.append({'Junction': f"Junction {j}", 'Max Traffic': df_j['Predicted_Vehicles'].max()})
                df_comp = pd.DataFrame(junction_comparison)

                fig_bar = px.bar(df_comp, x='Junction', y='Max Traffic', color='Junction',
                                 labels={'Max Traffic':'Peak Traffic Volume'}, template="plotly_dark")
                fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            # ==========================================
            # 5. NEW FEATURE: HOURLY DEEP DIVE & SUGGESTIONS
            # ==========================================

            st.write("---")
            st.subheader(f"🔍 Hourly Micro-Analysis Deep Dive: {target_hour}:00 - {target_hour+1}:00")

            # Extract vehicle count for that specific hour
            selected_hour_traffic = int(df_forecast.loc[df_forecast['Hour'] == target_hour, 'Predicted_Vehicles'].values[0])

            h_col1, h_col2 = st.columns([1, 2])

            with h_col1:
                st.markdown(f"""
                <div class='metric-box' style='border: 2px solid #06b6d4;'>
                    📊 <b>Volume at {target_hour}:00</b>
                    <h1 style='color: #06b6d4;'>{selected_hour_traffic}</h1>
                    <p>Vehicles Expected</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("---")

                # Formulating localized explanations and advice depending on load severity
                st.markdown("#### 📝 Hourly AI Diagnostic & Suggestion")
                if selected_hour_traffic > danger_threshold:
                    status_color = "🔴 CRITICAL OVERLOAD"
                    explanation = f"Traffic at {target_hour}:00 exceeds your safety threshold of {danger_threshold} vehicles. This will result in severe bottlenecking and cascading delays across connected intersections."
                    suggestion = "🔹 **Action Items:**\n1. Automatically extend green light duration for this junction by 25%.\n2. Issue a real-time navigation alert to divert commuter traffic via secondary routes.\n3. Deploy physical traffic warden control to prevent intersection blocking."

                elif selected_hour_traffic >= (danger_threshold * 0.75):
                    status_color = "🟡 MODERATE/HIGH TRAFFIC"
                    explanation = f"Traffic is building steadily up to {selected_hour_traffic} vehicles. While below the absolute saturation line, it is approaching a critical state."
                    suggestion = "🔹 **Action Items:**\n1. Synchronize electronic signs to encourage smooth lane distribution.\n2. Keep rapid response towing units on standby nearby to clear any incidents quickly."

                else:
                    status_color = "🟢 fluid / low traffic"
                    explanation = f"The junction is operating efficiently at normal capacities with plenty of buffer room."
                    suggestion = "🔹 **Action Items:**\n1. Maintain standard automated signal cycles.\n2. Ideal window for scheduled roadside maintenance or street cleaning crews."

                st.markdown(f"**Current Status:** `{status_color}`")
                st.markdown(explanation)
                st.markdown(suggestion)

            with h_col2:
                # Graph to visualize exactly where this hour stands in the daily timeline
                # We construct a bar chart for the entire day, but highlight the selected hour in a distinct accent color!
                df_forecast['Highlight'] = df_forecast['Hour'].apply(lambda x: f"Selected ({target_hour}:00)" if x == target_hour else "Other Hours")

                fig_hourly_bar = px.bar(
                    df_forecast, 
                    x='Hour', 
                    y='Predicted_Vehicles', 
                    color='Highlight',
                    color_discrete_map={f"Selected ({target_hour}:00)": "#06b6d4", "Other Hours": "#475569"},
                    labels={'Hour':'Hour of Day', 'Predicted_Vehicles':'Vehicles'},
                    title=f"Hourly Traffic Distribution (Highlighting {target_hour}:00)",
                    template="plotly_dark"
                )
                fig_hourly_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=True)
                st.plotly_chart(fig_hourly_bar, use_container_width=True)


            # ==========================================
            # 6. GLOBAL AI CHATBOT CONSULTANT
            # ==========================================

            st.write("---")
            st.subheader("🤖 Executive Smart City Planning Summary")

            hours_above_threshold = df_forecast[df_forecast['Predicted_Vehicles'] > danger_threshold]['Hour'].tolist()
            bot_response = f"**Command Center Summary Brief:**\n\n"

            if len(hours_above_threshold) > 0:
                formatted_hours = ", ".join([f"{h}:00" for h in hours_above_threshold])
                bot_response += f"⚠️ **SYSTEM ALERTS RUNNING:** Junction {target_junction} will experience systemic capacity violations across `{len(hours_above_threshold)} hours` today, specifically during: `{formatted_hours}`. \n\n"
            else:
                bot_response += f"✅ **SYSTEM STATUS: OPTIMAL.** No comprehensive threshold failures are expected for Junction {target_junction} over this 24-hour cycle.\n\n"

            busiest_j = df_comp.loc[df_comp['Max Traffic'].idxmax(), 'Junction']
            bot_response += f"🏗️ **Macro Infrastructure Note:** Looking at the wider network for {target_date}, **{busiest_j}** registers the absolute highest congestion peak. Ensure long-term expansion funds favor {busiest_j} blueprints."

            st.markdown(f"<div class='chat-box'>{bot_response}</div>", unsafe_allow_html=True)










            



    


