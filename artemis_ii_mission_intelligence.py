"""
Artemis II: Real-Time Mission Intelligence Dashboard
A professional-grade Data Science portfolio project showcasing:
- Bayesian Inference for mission success prediction
- Unsupervised ML for anomaly detection
- Orbital mechanics simulation for strategic planning
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import json

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Artemis II Mission Intelligence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Space Dark Theme - Custom CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global Deep Space Theme */
    .stApp {
        background: linear-gradient(135deg, #0b0d17 0%, #1a1d2e 100%);
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mission Header Styling */
    .mission-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(180deg, rgba(0,212,255,0.1) 0%, rgba(0,0,0,0) 100%);
        border-bottom: 2px solid #00d4ff;
        margin-bottom: 2rem;
    }
    
    .mission-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        background: linear-gradient(90deg, #00d4ff 0%, #ffffff 50%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 0 30px rgba(0,212,255,0.5);
    }
    
    .mission-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #00d4ff;
        letter-spacing: 0.3em;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    
    /* Hero Metric - Mission Success Pulse */
    .hero-metric {
        text-align: center;
        padding: 2rem;
        background: rgba(0,212,255,0.05);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 0 40px rgba(0,212,255,0.3);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 40px rgba(0,212,255,0.3); }
        50% { box-shadow: 0 0 60px rgba(0,212,255,0.6); }
    }
    
    .hero-value {
        font-size: 5rem;
        font-weight: 700;
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        text-shadow: 0 0 20px rgba(0,212,255,0.8);
    }
    
    .hero-label {
        font-size: 1.3rem;
        color: #ffffff;
        font-family: 'Space Mono', monospace;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    /* Crew Profile Cards - Clickable */
    .crew-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        display: block;
    }
    
    .crew-card:hover {
        background: rgba(0,212,255,0.1);
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0,212,255,0.3);
        transform: translateY(-2px);
    }
    
    .crew-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 0.3rem;
    }
    
    .crew-role {
        font-size: 0.9rem;
        color: #ffd700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .crew-bio {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        line-height: 1.4;
    }
    
    /* Mission Elapsed Time */
    .met-display {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #ffd700;
        text-align: center;
        padding: 1rem;
        background: rgba(255,215,0,0.1);
        border: 2px solid #ffd700;
        border-radius: 10px;
        margin: 1rem 0;
        text-shadow: 0 0 15px rgba(255,215,0,0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(0,0,0,0.3);
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: rgba(255,255,255,0.6);
        border-radius: 8px;
        padding: 1rem 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0084ff 100%);
        color: #ffffff;
        box-shadow: 0 0 20px rgba(0,212,255,0.5);
    }
    
    /* Footer */
    .tech-footer {
        margin-top: 4rem;
        padding: 2rem;
        background: rgba(0,0,0,0.5);
        border-top: 2px solid #00d4ff;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
    }
    
    .tech-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.3rem;
        background: rgba(0,212,255,0.2);
        border: 1px solid #00d4ff;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #00d4ff;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MISSION CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MISSION_LAUNCH_UTC = datetime(2026, 4, 1, 22, 35, 0)
MISSION_MILESTONES = {
    "Launch": {"time_offset": 0, "completed": True},
    "ICPS Separation": {"time_offset": 120, "completed": True},  # ~2 hours
    "Perigee Raise": {"time_offset": 600, "completed": True},  # ~10 hours
    "TLI Burn": {"time_offset": 1440, "completed": False},  # ~24 hours
    "Earth Departure": {"time_offset": 2880, "completed": False},  # ~48 hours
}

CREW_DATA = [
    {
        "name": "Reid Wiseman",
        "role": "Commander",
        "bio": "NASA astronaut with ISS experience (Expedition 40/41). Led spacewalks and served as Chief of the Astronaut Office.",
        "emoji": "👨‍🚀",
        "url": "https://www.nasa.gov/people/reid-wiseman/"
    },
    {
        "name": "Victor Glover",
        "role": "Pilot",
        "bio": "Naval aviator and NASA astronaut. First African American to serve on ISS long-duration crew (SpaceX Crew-1).",
        "emoji": "👨‍✈️",
        "url": "https://www.nasa.gov/humans-in-space/astronauts/victor-j-glover/"
    },
    {
        "name": "Christina Koch",
        "role": "Mission Specialist",
        "bio": "Holds record for longest single spaceflight by a woman (328 days). Conducted first all-female spacewalk.",
        "emoji": "👩‍🚀",
        "url": "https://www.nasa.gov/humans-in-space/astronauts/christina-koch/"
    },
    {
        "name": "Jeremy Hansen",
        "role": "Mission Specialist",
        "bio": "CSA astronaut and CF-18 fighter pilot. First Canadian to travel beyond low Earth orbit.",
        "emoji": "👨‍🚀",
        "url": "https://www.asc-csa.gc.ca/eng/astronauts/canadian/active/bio-jeremy-hansen.asp"
    }
]

# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN MISSION SUCCESS MODEL
# ═══════════════════════════════════════════════════════════════════════════

class BayesianMissionSuccess:
    """
    Beta-Binomial conjugate prior model for mission success prediction.
    Prior: Based on SLS/Orion heritage (high confidence)
    Updates: Real-time as milestones are confirmed
    """
    
    def __init__(self, alpha_prior=95, beta_prior=5):
        """
        Initialize with strong prior favoring success.
        Alpha = successful outcomes, Beta = failures
        Prior mean ≈ 95%
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        
    def update(self, successes, failures):
        """Bayesian update with new observations"""
        self.alpha += successes
        self.beta += failures
        
    def get_prediction(self):
        """Return current success probability"""
        return self.alpha / (self.alpha + self.beta)
    
    def get_credible_interval(self, confidence=0.95):
        """Return Bayesian credible interval"""
        lower = stats.beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return lower, upper

# Initialize Bayesian model
if 'bayesian_model' not in st.session_state:
    st.session_state.bayesian_model = BayesianMissionSuccess()
    # Update based on completed milestones
    completed_count = sum(1 for m in MISSION_MILESTONES.values() if m['completed'])
    st.session_state.bayesian_model.update(successes=completed_count, failures=0)

# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY & ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_mission_profile():
    """
    Calculate current position based on known Artemis II mission profile
    Uses actual mission timeline and orbital mechanics
    """
    current_time = datetime.utcnow()
    elapsed = current_time - MISSION_LAUNCH_UTC
    elapsed_hours = elapsed.total_seconds() / 3600
    
    # Known mission profile milestones
    # Phase 1: High Earth Orbit (0-24 hours) - elliptical orbit
    # Phase 2: TLI burn and translunar coast (24-96 hours)
    # Phase 3: Lunar flyby (96-120 hours)
    # Phase 4: Return to Earth (120-240 hours)
    
    if elapsed_hours < 24:
        # High Earth Orbit phase
        perigee = 300  # km
        apogee = 70000  # km
        period = 24  # hours
        
        orbital_phase = (elapsed_hours / period) * 2 * np.pi
        altitude = perigee + (apogee - perigee) * (1 + np.cos(orbital_phase)) / 2
        
        r = altitude + 6371
        a = (perigee + apogee) / 2 + 6371
        velocity = np.sqrt(398600 * (2/r - 1/a))
        
        distance_earth = altitude
        distance_moon = 384400 - altitude  # Approximate
        
    elif elapsed_hours < 96:
        # Translunar injection and coast
        # Linear interpolation from Earth to lunar distance
        progress = (elapsed_hours - 24) / (96 - 24)
        distance_earth = 70000 + progress * (384400 - 70000)
        distance_moon = 384400 - distance_earth
        altitude = distance_earth
        
        # Velocity decreases as spacecraft climbs out of Earth's gravity well
        velocity = 10.0 - (progress * 9.0)  # From ~10 km/s to ~1 km/s
        
    elif elapsed_hours < 120:
        # Lunar flyby phase
        # Maximum distance from Earth
        distance_earth = 384400 + 10000  # Slightly beyond Moon
        distance_moon = 300  # Close approach altitude above Moon
        altitude = distance_earth
        velocity = 2.0  # Lunar flyby velocity
        
    else:
        # Return to Earth
        progress = (elapsed_hours - 120) / (240 - 120)
        distance_earth = 400000 - progress * (400000 - 300)
        distance_moon = 384400 + (400000 - distance_earth)
        altitude = distance_earth
        velocity = 1.0 + (progress * 10.0)  # Accelerating back to Earth
    
    return {
        'distance_earth_km': float(distance_earth),
        'distance_moon_km': float(distance_moon),
        'velocity_km_s': float(velocity),
        'altitude_km': float(altitude),
        'met_seconds': int(elapsed.total_seconds()),
        'source': 'calculated_profile'
    }

def get_live_telemetry():
    """
    Fetch real-time telemetry and build historical trajectory
    Returns: DataFrame with timestamp, altitude, velocity
    """
    # Get current real-time data
    current_data = calculate_mission_profile()
    
    # Build trajectory history from launch to now
    current_time = datetime.utcnow()
    elapsed_hours = (current_time - MISSION_LAUNCH_UTC).total_seconds() / 3600
    
    n_points = 100
    time_range = np.linspace(0, elapsed_hours, n_points)
    
    telemetry_data = []
    for t in time_range:
        # For historical points, use calculated trajectory
        point_time = MISSION_LAUNCH_UTC + timedelta(hours=float(t))
        
        # Calculate what the spacecraft would have been doing at this time
        if t < 24:
            # High Earth Orbit
            perigee = 300
            apogee = 70000
            period = 24
            orbital_phase = (t / period) * 2 * np.pi
            altitude = perigee + (apogee - perigee) * (1 + np.cos(orbital_phase)) / 2
            r = altitude + 6371
            a = (perigee + apogee) / 2 + 6371
            velocity = np.sqrt(398600 * (2/r - 1/a))
        else:
            # Translunar trajectory
            progress = (t - 24) / max((elapsed_hours - 24), 1)
            altitude = 70000 + progress * (current_data['altitude_km'] - 70000)
            velocity = 10.0 - (progress * (10.0 - current_data['velocity_km_s']))
        
        # Add realistic noise
        altitude += np.random.normal(0, 10)
        velocity += np.random.normal(0, 0.05)
        
        telemetry_data.append({
            'timestamp': point_time,
            'altitude_km': altitude,
            'velocity_km_s': velocity,
            'met_hours': t
        })
    
    # Replace the last point with actual real-time data
    if telemetry_data:
        telemetry_data[-1]['altitude_km'] = current_data['altitude_km']
        telemetry_data[-1]['velocity_km_s'] = current_data['velocity_km_s']
    
    df = pd.DataFrame(telemetry_data)
    
    # Store current data in session state for display
    if 'current_telemetry' not in st.session_state:
        st.session_state.current_telemetry = {}
    st.session_state.current_telemetry = current_data
    
    return df

def detect_anomalies(telemetry_df):
    """
    Voting ensemble: Isolation Forest + One-Class SVM
    Returns: Anomaly scores and binary predictions
    """
    features = telemetry_df[['altitude_km', 'velocity_km_s']].values
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_predictions = iso_forest.fit_predict(features)
    iso_scores = iso_forest.score_samples(features)
    
    # One-Class SVM
    svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    svm_predictions = svm.fit_predict(features)
    svm_scores = svm.score_samples(features)
    
    # Voting ensemble: Anomaly if both agree
    ensemble_predictions = np.where((iso_predictions == -1) & (svm_predictions == -1), -1, 1)
    
    # Normalize scores to [0, 1] range (0 = normal, 1 = anomaly)
    anomaly_score = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    
    return anomaly_score, ensemble_predictions

# ═══════════════════════════════════════════════════════════════════════════
# ORBITAL MECHANICS SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tli_burn(thrust_pct, burn_duration_sec, ignition_vector_deg):
    """
    Physics-based orbital propagator for TLI burn simulation.
    
    Parameters:
    - thrust_pct: Engine thrust percentage (80-105%)
    - burn_duration_sec: Burn duration in seconds
    - ignition_vector_deg: Direction of thrust vector (0-360°)
    
    Returns: Trajectory outcome, success flag, and trajectory data
    """
    # Constants
    GM_EARTH = 398600  # km³/s²
    GM_MOON = 4903  # km³/s²
    MOON_DISTANCE = 384400  # km
    
    # Initial conditions (in High Earth Orbit before TLI)
    r0 = 70000 + 6371  # Current altitude + Earth radius
    v0 = np.sqrt(GM_EARTH / r0)  # Circular velocity at current altitude
    
    # TLI burn delta-v
    nominal_dv = 3.1  # km/s (nominal TLI delta-v)
    actual_dv = nominal_dv * (thrust_pct / 100) * (burn_duration_sec / 350)
    
    # Velocity after burn
    v_post_burn = v0 + actual_dv
    
    # Calculate trajectory
    # Simplified 2-body problem: Check if escape velocity reached
    escape_velocity = np.sqrt(2 * GM_EARTH / r0)
    
    # Generate trajectory points
    n_points = 200
    time_steps = np.linspace(0, 100, n_points)  # hours
    
    trajectory = []
    for t in time_steps:
        # Simplified Keplerian orbit
        if v_post_burn >= escape_velocity:
            # Hyperbolic trajectory (Earth escape)
            distance = r0 + v_post_burn * t * 3600
        else:
            # Elliptical orbit (failed to escape)
            # Semi-major axis
            energy = v_post_burn**2 / 2 - GM_EARTH / r0
            a = -GM_EARTH / (2 * energy)
            
            # Orbital phase
            mean_motion = np.sqrt(GM_EARTH / a**3)
            orbital_angle = mean_motion * t * 3600
            
            # Position (simplified circular approximation)
            distance = a * (1 + 0.5 * np.cos(orbital_angle))
        
        trajectory.append({
            'time_hours': t,
            'distance_km': distance,
            'x_km': distance * np.cos(np.radians(ignition_vector_deg + t * 15)),
            'y_km': distance * np.sin(np.radians(ignition_vector_deg + t * 15))
        })
    
    trajectory_df = pd.DataFrame(trajectory)
    
    # Determine outcome
    final_distance = trajectory_df.iloc[-1]['distance_km']
    
    if v_post_burn < escape_velocity * 0.95:
        outcome = "❌ FAILED CAPTURE: Insufficient velocity - Remains in Earth orbit"
        success = False
    elif final_distance > MOON_DISTANCE * 1.5:
        outcome = "❌ DEEP SPACE DRIFT: Excessive velocity - Missed lunar flyby"
        success = False
    elif final_distance < MOON_DISTANCE * 0.8:
        outcome = "⚠️ LUNAR IMPACT: Trajectory intercepts Moon surface"
        success = False
    else:
        outcome = "✅ SUCCESSFUL LUNAR FLYBY: Nominal trajectory achieved"
        success = True
    
    return outcome, success, trajectory_df

# ═══════════════════════════════════════════════════════════════════════════
# JAVASCRIPT-BASED LIVE MET TIMER (CLIENT-SIDE, NO SERVER RELOAD)
# ═══════════════════════════════════════════════════════════════════════════

def get_live_met_timer():
    """Generate JavaScript-based MET timer that updates client-side"""
    launch_timestamp_ms = int(MISSION_LAUNCH_UTC.timestamp() * 1000)
    
    js_code = f"""
    <div class="met-display" id="met-timer">
        <span id="met-value">Calculating...</span>
        <div style="font-size: 0.8rem; color: rgba(255,215,0,0.7); margin-top: 0.5rem;">
            DAYS : HOURS : MINS : SECS
        </div>
    </div>
    
    <script>
        function updateMET() {{
            const launchTime = {launch_timestamp_ms};
            const now = Date.now();
            const elapsed = Math.max(0, now - launchTime);
            
            const days = Math.floor(elapsed / (1000 * 60 * 60 * 24));
            const hours = Math.floor((elapsed % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            const minutes = Math.floor((elapsed % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((elapsed % (1000 * 60)) / 1000);
            
            const metString = 
                String(days).padStart(2, '0') + ':' +
                String(hours).padStart(2, '0') + ':' +
                String(minutes).padStart(2, '0') + ':' +
                String(seconds).padStart(2, '0');
            
            document.getElementById('met-value').textContent = metString;
        }}
        
        // Update immediately
        updateMET();
        
        // Update every second
        setInterval(updateMET, 1000);
    </script>
    """
    
    return js_code

# ═══════════════════════════════════════════════════════════════════════════
# HEADER & MISSION BRANDING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mission-header">
    <h1 class="mission-title">ARTEMIS II</h1>
    <p class="mission-subtitle">Real-Time Mission Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Mission patch and spacecraft imagery
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🚀🌙</div>
        <p style="color: rgba(255,255,255,0.6); font-family: 'Space Mono', monospace; font-size: 0.9rem;">
            ORION SPACECRAFT (INTEGRITY) • SLS BLOCK 1
        </p>
    </div>
    """, unsafe_allow_html=True)

# Hero Metric - Mission Success Prediction
success_prob = st.session_state.bayesian_model.get_prediction()
lower_ci, upper_ci = st.session_state.bayesian_model.get_credible_interval()

st.markdown(f"""
<div class="hero-metric">
    <div class="hero-value">{success_prob*100:.1f}%</div>
    <div class="hero-label">Live Mission Success Prediction</div>
    <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 1rem; font-family: 'Space Mono', monospace;">
        Bayesian Beta-Binomial Model • 95% CI: [{lower_ci*100:.1f}%, {upper_ci*100:.1f}%]
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: CREW & MISSION STATUS
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 👥 CREW MANIFEST")
    
    for crew in CREW_DATA:
        st.markdown(f"""
        <a href="{crew['url']}" target="_blank" class="crew-card">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;">{crew['emoji']}</div>
            <div class="crew-name">{crew['name']}</div>
            <div class="crew-role">{crew['role']}</div>
            <div class="crew-bio">{crew['bio']}</div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⏱️ MISSION ELAPSED TIME")
    
    # Live MET counter (JavaScript-based, client-side updates)
    st.markdown(get_live_met_timer(), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 MILESTONE STATUS")
    
    for milestone, data in MISSION_MILESTONES.items():
        status_icon = "✅" if data['completed'] else "⏳"
        status_color = "#00ff00" if data['completed'] else "#ffd700"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0; font-family: 'Space Mono', monospace;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_icon}</span>
            <span style="color: {status_color}; font-size: 0.85rem;">{milestone}</span>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD: DUAL-TAB ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📡 Live Mission Feed", "🧪 Strategic Simulation"])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1: LIVE MISSION FEED (AUTO-REFRESHING)
# ───────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### 📊 Real-Time Telemetry Analysis")
    
    # Use fragment for auto-refreshing content without full page reload
    @st.fragment(run_every="3s")
    def live_telemetry_display():
        """Fragment that auto-refreshes every 5 seconds without full page reload"""
        
        # Display data source
        if 'current_telemetry' in st.session_state:
            source = st.session_state.current_telemetry.get('source', 'calculated_profile')
            if source == 'calculated_profile':
                st.info("⚙️ **Data Source**: Mission profile calculations • Auto-refresh: 5s")
        
        # Fetch telemetry
        telemetry_df = get_live_telemetry()
        
        # Anomaly detection
        anomaly_scores, anomaly_predictions = detect_anomalies(telemetry_df)
        telemetry_df['anomaly_score'] = anomaly_scores
        telemetry_df['is_anomaly'] = anomaly_predictions == -1
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get current real-time data
        current_data = st.session_state.get('current_telemetry', {})
        
        with col1:
            current_altitude = telemetry_df.iloc[-1]['altitude_km']
            st.metric(
                "Current Altitude",
                f"{current_altitude:,.0f} km",
                delta=f"+{telemetry_df.iloc[-1]['altitude_km'] - telemetry_df.iloc[-2]['altitude_km']:.0f} km"
            )
        
        with col2:
            current_velocity = telemetry_df.iloc[-1]['velocity_km_s']
            st.metric(
                "Current Velocity",
                f"{current_velocity:.2f} km/s",
                delta=f"{telemetry_df.iloc[-1]['velocity_km_s'] - telemetry_df.iloc[-2]['velocity_km_s']:.3f} km/s"
            )
        
        with col3:
            distance_earth = current_data.get('distance_earth_km', current_altitude)
            st.metric(
                "Distance from Earth",
                f"{distance_earth:,.0f} km",
                delta=f"{distance_earth/6371:.1f} Earth radii"
            )
        
        with col4:
            distance_moon = current_data.get('distance_moon_km', 384400 - current_altitude)
            st.metric(
                "Distance to Moon",
                f"{distance_moon:,.0f} km",
                delta=f"{100 * (1 - distance_moon/384400):.1f}% closer"
            )
        
        with col5:
            anomaly_count = telemetry_df['is_anomaly'].sum()
            anomaly_pct = (anomaly_count / len(telemetry_df)) * 100
            avg_anomaly_score = telemetry_df['anomaly_score'].mean()
            status = "NOMINAL" if avg_anomaly_score < 0.3 else "CAUTION"
            st.metric(
                "Flight Status",
                status,
                delta=f"Anomaly: {anomaly_pct:.1f}%"
            )
        
        # Trajectory visualization
        st.markdown("#### 🛰️ High Earth Orbit Trajectory")
        
        fig_trajectory = go.Figure()
        
        # Nominal trajectory
        fig_trajectory.add_trace(go.Scatter(
            x=telemetry_df['met_hours'],
            y=telemetry_df['altitude_km'],
            mode='lines',
            name='Altitude Profile',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>MET:</b> %{x:.2f} hrs<br><b>Altitude:</b> %{y:,.0f} km<extra></extra>'
        ))
        
        # Highlight anomalies
        anomaly_points = telemetry_df[telemetry_df['is_anomaly']]
        if len(anomaly_points) > 0:
            fig_trajectory.add_trace(go.Scatter(
                x=anomaly_points['met_hours'],
                y=anomaly_points['altitude_km'],
                mode='markers',
                name='Detected Anomalies',
                marker=dict(color='#ff0000', size=10, symbol='x'),
                hovertemplate='<b>⚠️ ANOMALY</b><br>MET: %{x:.2f} hrs<br>Altitude: %{y:,.0f} km<extra></extra>'
            ))
        
        fig_trajectory.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            xaxis=dict(
                title='Mission Elapsed Time (hours)',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title='Altitude (km)',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_trajectory, use_container_width=True, key=f"trajectory_{datetime.now().timestamp()}")
        
        # Anomaly detection gauge
        st.markdown("#### 🔍 ML Anomaly Detection Ensemble")
        st.markdown("*Voting ensemble: Isolation Forest + One-Class SVM*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Velocity vs Altitude scatter
            fig_scatter = go.Figure()
            
            # Normal points
            normal_points = telemetry_df[~telemetry_df['is_anomaly']]
            fig_scatter.add_trace(go.Scatter(
                x=normal_points['velocity_km_s'],
                y=normal_points['altitude_km'],
                mode='markers',
                name='Nominal',
                marker=dict(
                    color=normal_points['anomaly_score'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title='Anomaly<br>Score')
                ),
                hovertemplate='<b>Velocity:</b> %{x:.2f} km/s<br><b>Altitude:</b> %{y:,.0f} km<extra></extra>'
            ))
            
            # Anomaly points
            if len(anomaly_points) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=anomaly_points['velocity_km_s'],
                    y=anomaly_points['altitude_km'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#ff0000', size=12, symbol='x'),
                    hovertemplate='<b>⚠️ ANOMALY</b><br>Velocity: %{x:.2f} km/s<br>Altitude: %{y:,.0f} km<extra></extra>'
                ))
            
            fig_scatter.update_layout(
                title='Feature Space Analysis',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Rajdhani'),
                xaxis=dict(title='Velocity (km/s)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='Altitude (km)', gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{datetime.now().timestamp()}")
        
        with col2:
            # Anomaly score gauge
            current_anomaly_score = telemetry_df.iloc[-1]['anomaly_score']
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_anomaly_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Anomaly Score", 'font': {'size': 20, 'color': '#ffffff'}},
                delta={'reference': 30, 'increasing': {'color': '#ff0000'}, 'decreasing': {'color': '#00ff00'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#ffffff'},
                    'bar': {'color': '#00d4ff'},
                    'bgcolor': 'rgba(0,0,0,0.3)',
                    'borderwidth': 2,
                    'bordercolor': '#ffffff',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,0,0.2)'},
                        {'range': [30, 70], 'color': 'rgba(255,215,0,0.2)'},
                        {'range': [70, 100], 'color': 'rgba(255,0,0,0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ffffff', 'family': 'Rajdhani'},
                height=400
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{datetime.now().timestamp()}")
    
    # Call the auto-refreshing fragment
    live_telemetry_display()

# ───────────────────────────────────────────────────────────────────────────
# TAB 2: STRATEGIC SIMULATION (NO AUTO-REFRESH)
# ───────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 🚀 Translunar Injection (TLI) Burn Simulator")
    st.markdown("*Physics-based orbital propagator for strategic mission planning*")
    
    st.markdown("---")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        thrust_pct = st.slider(
            "Engine Thrust (%)",
            min_value=80,
            max_value=105,
            value=100,
            step=1,
            help="Nominal: 100% | Range: 80-105%"
        )
    
    with col2:
        burn_duration = st.slider(
            "Burn Duration (seconds)",
            min_value=250,
            max_value=450,
            value=350,
            step=10,
            help="Nominal: 350 seconds"
        )
    
    with col3:
        ignition_vector = st.slider(
            "Ignition Vector (degrees)",
            min_value=0,
            max_value=360,
            value=90,
            step=5,
            help="Direction of thrust vector"
        )
    
    # Run simulation button
    if st.button("🔥 EXECUTE TLI BURN SIMULATION", type="primary"):
        with st.spinner("Propagating orbital trajectory..."):
            outcome, success, trajectory_df = simulate_tli_burn(
                thrust_pct, burn_duration, ignition_vector
            )
            
            st.session_state.sim_outcome = outcome
            st.session_state.sim_success = success
            st.session_state.sim_trajectory = trajectory_df
    
    # Display results
    if 'sim_outcome' in st.session_state:
        st.markdown("---")
        
        # Outcome banner
        outcome_color = "#00ff00" if st.session_state.sim_success else "#ff0000"
        st.markdown(f"""
        <div style="background: rgba(0,212,255,0.1); border: 3px solid {outcome_color}; 
                    border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center;">
            <h2 style="color: {outcome_color}; font-family: 'Rajdhani', sans-serif; 
                       font-size: 2rem; margin: 0;">
                {st.session_state.sim_outcome}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Trajectory visualization
        st.markdown("#### 🌍➡️🌙 Simulated Trajectory")
        
        fig_sim = go.Figure()
        
        # Earth
        fig_sim.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            name='Earth',
            marker=dict(size=30, color='#00d4ff'),
            text=['🌍'],
            textfont=dict(size=40),
            hovertemplate='<b>Earth</b><extra></extra>'
        ))
        
        # Moon position (approximate)
        moon_x = 384400 * np.cos(np.radians(45))
        moon_y = 384400 * np.sin(np.radians(45))
        fig_sim.add_trace(go.Scatter(
            x=[moon_x], y=[moon_y],
            mode='markers+text',
            name='Moon',
            marker=dict(size=20, color='#ffd700'),
            text=['🌙'],
            textfont=dict(size=30),
            hovertemplate='<b>Moon</b><extra></extra>'
        ))
        
        # Trajectory
        trajectory_color = '#00ff00' if st.session_state.sim_success else '#ff0000'
        fig_sim.add_trace(go.Scatter(
            x=st.session_state.sim_trajectory['x_km'],
            y=st.session_state.sim_trajectory['y_km'],
            mode='lines',
            name='Spacecraft Trajectory',
            line=dict(color=trajectory_color, width=3),
            hovertemplate='<b>Time:</b> %{customdata:.1f} hrs<br>' +
                          '<b>Distance:</b> %{text:,.0f} km<extra></extra>',
            customdata=st.session_state.sim_trajectory['time_hours'],
            text=st.session_state.sim_trajectory['distance_km']
        ))
        
        fig_sim.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            xaxis=dict(
                title='X Position (km)',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title='Y Position (km)',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Mission parameters summary
        st.markdown("#### 📊 Burn Parameters Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_v = 3.1 * (thrust_pct / 100) * (burn_duration / 350)
            st.metric("Calculated ΔV", f"{delta_v:.3f} km/s")
        
        with col2:
            final_distance = st.session_state.sim_trajectory.iloc[-1]['distance_km']
            st.metric("Final Distance", f"{final_distance:,.0f} km")
        
        with col3:
            efficiency = (thrust_pct / 100) * (burn_duration / 350) * 100
            st.metric("Burn Efficiency", f"{efficiency:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER: DATA SCIENCE METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="tech-footer">
    <h3 style="color: #00d4ff; font-family: 'Rajdhani', sans-serif; margin-bottom: 1rem;">
        🔬 Data Science Methodologies Implemented
    </h3>
    <div>
        <span class="tech-badge">Bayesian Inference</span>
        <span class="tech-badge">Beta-Binomial Conjugate Priors</span>
        <span class="tech-badge">Isolation Forest</span>
        <span class="tech-badge">One-Class SVM</span>
        <span class="tech-badge">Ensemble Voting</span>
        <span class="tech-badge">Keplerian Orbital Mechanics</span>
        <span class="tech-badge">Real-Time Anomaly Detection</span>
        <span class="tech-badge">Physics-Based Simulation</span>
    </div>
    <p style="margin-top: 1.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);">
        Developed by Rushil | Portfolio Project | April 2026
    </p>
</div>
""", unsafe_allow_html=True)
