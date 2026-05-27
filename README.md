# Smart City Traffic Forecasting AI Command Center

**[Click Here to Access the Live Traffic Command Center Dashboard](https://forecasting-of-smart-city-traffic-patterns-lcktdt2xgs5i7hkyy8g.streamlit.app/)**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![UI Framework](https://img.shields.io/badge/UI%20Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Data Visualization](https://img.shields.io/badge/Graphics-Plotly%20Express-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Machine Learning](https://img.shields.io/badge/ML%20Engine-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

An end-to-end Machine Learning web application engineered for municipal planners and traffic law enforcement agencies. This system models, analyzes, and forecasts real-time traffic volume anomalies across **4 major urban junctions** simultaneously, helping cities plan ahead for peak hours and maximize civic infrastructure return on investment (ROI).

## Key Engineering & Numerical Achievements 

> **Why this project stands out:** Instead of stopping at a typical classroom notebook simulation, this repository implements a highly optimized data pipeline and real-time inference web application.

* **📉 76.5% Model Footprint Reduction:** Overcame GitHub's strict **100.00 MB** single-file upload restriction by applying binary serialization compression. Successfully shrunk a massive multi-tree Random Forest Regressor from **127.47 MB down to a lightweight 30.01 MB** without a single drop in predictive capability.
* **⏱️ < 400ms Sub-Second Inference Latency:** Engineered an optimized multi-row matrix generation script that synthesizes operational vectors and outputs complete 24-hour traffic trend forecasts across any targeted calendar date in **under 400 milliseconds**.
* **🗓️ 100% Automated Holiday Vectoring:** Built a real-time calendar interpolation engine utilizing the `holidays` database. Dynamically marks and encodes binary holiday weight indicators (`Is_Holiday: 0 or 1`) to accurately flag localized commuter anomalies on non-working city days.
* **📊 7-Dimension Feature Engineering Pipeline:** Transformed raw sequential timeline stamps into high-signal numerical arrays across **7 distinct feature vectors** trained inside an advanced `RandomForestRegressor (n_estimators=100)`.
* **💡 Dynamic Infrastructure Impact Index:** Computes maximum absolute traffic burden across **all 4 junctions simultaneously** on the fly, instantly rendering context-aware budgeting suggestions for municipal expansion.

---

## Core System Features

### 1. Gateway Security (Simulated RBAC Role Access)
* Equipped with a built-in session state authentication portal supporting instant account registration and credential verification to replicate secure municipal data handling.

### 2. 24-Hour Macro Trend Visualizer
* Renders interactive, high-fidelity vector line plots detailing complete 24-hour traffic volume changes.
* Features a responsive **General Warning Threshold Slider** that dynamically draws a live, bright red **Traffic Saturation Limit Line** across the dataset to flag overflow vulnerabilities.

### 3. Hourly Micro-Analysis Deep-Dive Selector
* Allows operators to isolate **any specific hour (0-23)** of the target date via a precision slider.
* The system dynamically alters the UI theme: the selected hour glows in **neon cyan (`#06b6d4`)**, while surrounding timeline hours fade into a dark slate background for clear visual comparison.
* Generates instant threat matrix classifications: `🔴 CRITICAL OVERLOAD`, `🟡 MODERATE/HIGH TRAFFIC`, or `🟢 FLUID / LOW TRAFFIC`.

### 4. Embedded Rule-Based AI Analytical Consultant
* A localized decision intelligence script that reads real-time chart states and automatically outputs tactical directives (e.g., *"Extend signal duration by 25% at 17:00"* or *"Station ground units 30 minutes prior to peak hours"*).

---

## The Technical Stack

| Layer | Component / Library | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python 3.10+ | Primary development platform |
| **ML Engine** | Scikit-Learn | Training, local validation scoring, and inference engine |
| **Model Optimization**| Joblib | Binary serialization and zlib compression pipeline |
| **Feature Extraction** | Pandas / Datetime / Holidays | Dynamic temporal data synthesis and holiday flagging |
| **User Interface** | Streamlit Architecture | Web portal hosting and active state session caching |
| **Data Visualization**| Plotly Express | Real-time chart rendering and threshold manipulation |

### Machine Learning Feature Map
Every inference query compiles raw inputs into the exact mathematical schema expected by the trained model:
```python
# The 7-dimensional feature array required for real-time inference
feature_columns = ['Junction', 'Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'Is_Holiday']
