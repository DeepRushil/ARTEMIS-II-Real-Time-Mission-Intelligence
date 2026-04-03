"""
Artemis II: Real-Time Mission Intelligence Dashboard
A professional-grade Data Science portfolio project showcasing:
- Bayesian Inference for mission success prediction
- Unsupervised ML for anomaly detection
- Orbital mechanics simulation for strategic planning

FIXED VERSION - Addresses 7 critical vulnerabilities for production deployment
"""

import streamlit as st
import streamlit.components.v1 as components
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
    "Launch": {"time_offset_minutes": 0},
    "ICPS Separation": {"time_offset_minutes": 120},  # 2 hours
    "Perigee Raise": {"time_offset_minutes": 600},  # 10 hours
    "TLI Burn": {"time_offset_minutes": 1440},  # 24 hours
    "Earth Departure": {"time_offset_minutes": 2880},  # 48 hours
    "Lunar Flyby": {"time_offset_minutes": 5760},  # 96 hours (4 days)
}

def get_milestone_status():
    """Dynamically determine milestone completion based on current time"""
    current_time = datetime.utcnow()
    elapsed = current_time - MISSION_LAUNCH_UTC
    elapsed_minutes = elapsed.total_seconds() / 60
    
    milestone_status = {}
    for milestone, data in MISSION_MILESTONES.items():
        is_completed = elapsed_minutes >= data['time_offset_minutes']
        time_to_milestone = data['time_offset_minutes'] - elapsed_minutes
        milestone_status[milestone] = {
            'completed': is_completed,
            'time_offset_minutes': data['time_offset_minutes'],
            'time_remaining_minutes': max(0, time_to_milestone)
        }
    
    return milestone_status

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

# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY & ANOMALY DETECTION (FIXED: Session State Storage)
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
        distance_moon = 384400 - altitude
        
        # FIXED: Inject deliberate anomaly at ICPS separation (2 hours)
        if 1.9 < elapsed_hours < 2.1:
            altitude += np.random.normal(0, 100)  # Separation transient
            velocity += np.random.normal(0, 0.3)
        
    elif elapsed_hours < 96:
        # Translunar injection and coast
        progress = (elapsed_hours - 24) / (96 - 24)
        distance_earth = 70000 + progress * (384400 - 70000)
        distance_moon = 384400 - distance_earth
        altitude = distance_earth
        velocity = 10.0 - (progress * 9.0)
        
        # FIXED: Inject deliberate anomaly during TLI burn (24 hours)
        if 23.8 < elapsed_hours < 24.3:
            velocity += np.random.normal(0, 0.4)  # Burn variance
        
    elif elapsed_hours < 120:
        # Lunar flyby phase
        distance_earth = 384400 + 10000
        distance_moon = 300
        altitude = distance_earth
        velocity = 2.0
        
    else:
        # Return to Earth
        progress = (elapsed_hours - 120) / (240 - 120)
        distance_earth = 400000 - progress * (400000 - 300)
        distance_moon = 384400 + (400000 - distance_earth)
        altitude = distance_earth
        velocity = 1.0 + (progress * 10.0)
    
    return {
        'distance_earth_km': float(distance_earth),
        'distance_moon_km': float(distance_moon),
        'velocity_km_s': float(velocity),
        'altitude_km': float(altitude),
        'met_seconds': int(elapsed.total_seconds()),
        'timestamp': current_time,
        'source': 'simulated_profile'
    }

def get_live_telemetry():
    """
    FIXED: Efficient telemetry with session state storage
    Only recalculates new data points, not entire history
    """
    # Initialize session state storage
    if 'telemetry_history' not in st.session_state:
        st.session_state.telemetry_history = []
        st.session_state.last_telemetry_update = None
    
    # Get current data
    current_data = calculate_mission_profile()
    current_time = datetime.utcnow()
    
    # FIXED: Only add new point if at least 5 seconds have passed
    should_add = (
        st.session_state.last_telemetry_update is None or
        (current_time - st.session_state.last_telemetry_update).total_seconds() >= 5
    )
    
    if should_add:
        elapsed_hours = (current_time - MISSION_LAUNCH_UTC).total_seconds() / 3600
        
        # Add realistic noise to current point
        current_data['altitude_km'] += np.random.normal(0, 10)
        current_data['velocity_km_s'] += np.random.normal(0, 0.05)
        
        st.session_state.telemetry_history.append({
            'timestamp': current_time,
            'altitude_km': current_data['altitude_km'],
            'velocity_km_s': current_data['velocity_km_s'],
            'met_hours': elapsed_hours
        })
        
        st.session_state.last_telemetry_update = current_time
    
    # Keep last 100 points (FIXED: Memory management)
    if len(st.session_state.telemetry_history) > 100:
        st.session_state.telemetry_history = st.session_state.telemetry_history[-100:]
    
    df = pd.DataFrame(st.session_state.telemetry_history)
    
    # Store current data for display
    st.session_state.current_telemetry = current_data
    
    return df

def detect_anomalies(telemetry_df):
    """
    Voting ensemble: Isolation Forest + One-Class SVM
    Returns: Anomaly scores and binary predictions
    """
    if len(telemetry_df) < 10:
        # Not enough data for anomaly detection
        return np.zeros(len(telemetry_df)), np.ones(len(telemetry_df))
    
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
    
    # Normalize scores to [0, 1] range
    anomaly_score = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)
    
    return anomaly_score, ensemble_predictions

# ═══════════════════════════════════════════════════════════════════════════
# ORBITAL MECHANICS SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tli_burn(thrust_pct, burn_duration_sec, ignition_vector_deg):
    """
    Physics-based orbital propagator for TLI burn simulation.
    """
    GM_EARTH = 398600  # km³/s²
    GM_MOON = 4903  # km³/s²
    MOON_DISTANCE = 384400  # km
    
    r0 = 70000 + 6371
    v0 = np.sqrt(GM_EARTH / r0)
    
    nominal_dv = 3.1  # km/s
    actual_dv = nominal_dv * (thrust_pct / 100) * (burn_duration_sec / 350)
    
    v_post_burn = v0 + actual_dv
    
    escape_velocity = np.sqrt(2 * GM_EARTH / r0)
    
    n_points = 200
    time_steps = np.linspace(0, 100, n_points)
    
    trajectory = []
    for t in time_steps:
        if v_post_burn >= escape_velocity:
            distance = r0 + v_post_burn * t * 3600
        else:
            energy = v_post_burn**2 / 2 - GM_EARTH / r0
            a = -GM_EARTH / (2 * energy)
            mean_motion = np.sqrt(GM_EARTH / a**3)
            orbital_angle = mean_motion * t * 3600
            distance = a * (1 + 0.5 * np.cos(orbital_angle))
        
        trajectory.append({
            'time_hours': t,
            'distance_km': distance,
            'x_km': distance * np.cos(np.radians(ignition_vector_deg + t * 15)),
            'y_km': distance * np.sin(np.radians(ignition_vector_deg + t * 15))
        })
    
    trajectory_df = pd.DataFrame(trajectory)
    
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
# JAVASCRIPT-BASED LIVE MET TIMER
# ═══════════════════════════════════════════════════════════════════════════

def display_live_met_timer():
    """Display JavaScript-based MET timer"""
    launch_timestamp_ms = int(MISSION_LAUNCH_UTC.timestamp() * 1000)
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
<style>
@import url(https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap);
body {{margin:0;padding:0;background:transparent}}
.met-display {{
    font-family:"Space Mono",monospace;
    font-size:2rem;
    font-weight:700;
    color:#ffd700;
    text-align:center;
    padding:1rem;
    background:rgba(255,215,0,0.1);
    border:2px solid #ffd700;
    border-radius:10px;
    text-shadow:0 0 15px rgba(255,215,0,0.5)
}}
.met-label {{font-size:0.8rem;color:rgba(255,215,0,0.7);margin-top:0.5rem}}
</style>
</head>
<body>
<div class="met-display">
<div id="met-value">00:00:00:00</div>
<div class="met-label">DAYS : HOURS : MINS : SECS</div>
</div>
<script>
function updateMET(){{
const launchTime={timestamp};
const now=Date.now();
const elapsed=Math.max(0,now-launchTime);
const days=Math.floor(elapsed/(1000*60*60*24));
const hours=Math.floor((elapsed%(1000*60*60*24))/(1000*60*60));
const minutes=Math.floor((elapsed%(1000*60*60))/(1000*60));
const seconds=Math.floor((elapsed%(1000*60))/1000);
const metString=String(days).padStart(2,'0')+':'+String(hours).padStart(2,'0')+':'+String(minutes).padStart(2,'0')+':'+String(seconds).padStart(2,'0');
document.getElementById('met-value').textContent=metString;
}}
updateMET();
setInterval(updateMET,1000);
</script>
</body>
</html>
""".format(timestamp=launch_timestamp_ms)
    
    components.html(html_template, height=120)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER & MISSION BRANDING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mission-header">
    <h1 class="mission-title">ARTEMIS II</h1>
    <p class="mission-subtitle">Real-Time Mission Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🚀🌙</div>
        <p style="color: rgba(255,255,255,0.6); font-family: 'Space Mono', monospace; font-size: 0.9rem;">
            ORION SPACECRAFT (INTEGRITY) • SLS BLOCK 1 • SIMULATED MISSION PROFILE
        </p>
    </div>
    """, unsafe_allow_html=True)

# FIXED: Renamed to "Milestone-Based Prediction" for accuracy
@st.fragment(run_every="5s")
def display_mission_success():
    """Display milestone-based mission success prediction"""
    milestone_status = get_milestone_status()
    completed_count = sum(1 for m in milestone_status.values() if m['completed'])
    
    # FIXED: Incorporate anomaly data into Bayesian model
    bayesian_model = BayesianMissionSuccess()
    
    # Get recent anomaly rate if telemetry exists
    anomaly_penalty = 0
    if 'telemetry_history' in st.session_state and len(st.session_state.telemetry_history) > 0:
        recent_df = pd.DataFrame(st.session_state.telemetry_history[-20:])
        if len(recent_df) >= 10:
            _, anomaly_preds = detect_anomalies(recent_df)
            recent_anomaly_rate = (anomaly_preds == -1).mean()
            anomaly_penalty = int(recent_anomaly_rate * 3)
    
    bayesian_model.update(successes=completed_count, failures=anomaly_penalty)
    
    success_prob = bayesian_model.get_prediction()
    lower_ci, upper_ci = bayesian_model.get_credible_interval()
    
    st.markdown(f"""
    <div class="hero-metric">
        <div class="hero-value">{success_prob*100:.1f}%</div>
        <div class="hero-label">Milestone-Based Success Prediction</div>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 1rem; font-family: 'Space Mono', monospace;">
            Bayesian Beta-Binomial Model • 95% CI: [{lower_ci*100:.1f}%, {upper_ci*100:.1f}%]<br>
            <span style="color: #00d4ff;">Completed: {completed_count}/{len(milestone_status)} milestones</span>
            {f' • Anomaly Penalty: {anomaly_penalty}' if anomaly_penalty > 0 else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)

display_mission_success()

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
    
    display_live_met_timer()
    
    # FIXED: Added update rate transparency
    st.markdown("""
    <div style="background: rgba(0,212,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 1rem 0;">
        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); font-family: 'Space Mono', monospace;">
            🔄 <b>Update Rates:</b><br>
            • MET Timer: Real-time (1s)<br>
            • Telemetry: Every 5s<br>
            • Milestones: Every 5s<br>
            • Prediction: Every 5s
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    @st.fragment(run_every="5s")
    def display_mission_phase():
        """Display current mission phase"""
        current_time = datetime.utcnow()
        elapsed = current_time - MISSION_LAUNCH_UTC
        elapsed_hours = elapsed.total_seconds() / 3600
        
        if elapsed_hours < 24:
            phase = "🌍 High Earth Orbit"
            phase_color = "#00d4ff"
        elif elapsed_hours < 96:
            phase = "🚀 Translunar Coast"
            phase_color = "#ffd700"
        elif elapsed_hours < 120:
            phase = "🌙 Lunar Flyby"
            phase_color = "#ffffff"
        else:
            phase = "🏠 Earth Return"
            phase_color = "#00ff00"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 0.8rem; background: rgba(0,0,0,0.3); 
                    border: 1px solid {phase_color}; border-radius: 8px; margin: 1rem 0;">
            <div style="color: {phase_color}; font-family: 'Space Mono', monospace; 
                        font-size: 0.9rem; font-weight: 600;">
                {phase}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    display_mission_phase()
    
    st.markdown("---")
    st.markdown("### 🎯 MILESTONE STATUS")
    
    # FIXED: Reduced refresh rate to 5s for better responsiveness
    @st.fragment(run_every="5s")
    def display_milestones():
        """Display dynamically updating milestone status with countdown"""
        milestone_status = get_milestone_status()
        
        for milestone, data in milestone_status.items():
            status_icon = "✅" if data['completed'] else "⏳"
            status_color = "#00ff00" if data['completed'] else "#ffd700"
            
            # FIXED: Show time to next milestone
            time_display = ""
            if not data['completed'] and data['time_remaining_minutes'] < 180:  # < 3 hours
                hours = int(data['time_remaining_minutes'] // 60)
                minutes = int(data['time_remaining_minutes'] % 60)
                time_display = f" ({hours}h {minutes}m)"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0; font-family: 'Space Mono', monospace;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_icon}</span>
                <span style="color: {status_color}; font-size: 0.85rem;">{milestone}{time_display}</span>
            </div>
            """, unsafe_allow_html=True)
    
    display_milestones()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD: DUAL-TAB ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📡 Live Mission Feed", "🧪 Strategic Simulation"])

with tab1:
    st.markdown("### 📊 Real-Time Telemetry Analysis")
    
    @st.fragment(run_every="5s")
    def live_telemetry_display():
        """Fragment that auto-refreshes every 5 seconds"""
        
        current_utc = datetime.utcnow().strftime("%H:%M:%S UTC")
        
        # FIXED: Clear labeling as simulated data
        st.info(f"⚙️ **Data Source**: Physics-based simulation (not real NASA telemetry) • Updates: 5s • Last: {current_utc}")
        
        telemetry_df = get_live_telemetry()
        
        if len(telemetry_df) < 2:
            st.warning("Collecting telemetry data...")
            return
        
        anomaly_scores, anomaly_predictions = detect_anomalies(telemetry_df)
        telemetry_df['anomaly_score'] = anomaly_scores
        telemetry_df['is_anomaly'] = anomaly_predictions == -1
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_data = st.session_state.get('current_telemetry', {})
        
        with col1:
            current_altitude = telemetry_df.iloc[-1]['altitude_km']
            delta_alt = telemetry_df.iloc[-1]['altitude_km'] - telemetry_df.iloc[-2]['altitude_km']
            st.metric("Current Altitude", f"{current_altitude:,.0f} km", delta=f"{delta_alt:+.0f} km")
        
        with col2:
            current_velocity = telemetry_df.iloc[-1]['velocity_km_s']
            delta_vel = telemetry_df.iloc[-1]['velocity_km_s'] - telemetry_df.iloc[-2]['velocity_km_s']
            st.metric("Current Velocity", f"{current_velocity:.2f} km/s", delta=f"{delta_vel:+.3f} km/s")
        
        with col3:
            distance_earth = current_data.get('distance_earth_km', current_altitude)
            st.metric("Distance from Earth", f"{distance_earth:,.0f} km", delta=f"{distance_earth/6371:.1f}R⊕")
        
        with col4:
            distance_moon = current_data.get('distance_moon_km', 384400)
            st.metric("Distance to Moon", f"{distance_moon:,.0f} km", delta=f"{100 * (1 - distance_moon/384400):.1f}%")
        
        with col5:
            anomaly_count = telemetry_df['is_anomaly'].sum()
            avg_anomaly_score = telemetry_df['anomaly_score'].mean()
            status = "NOMINAL" if avg_anomaly_score < 0.3 else "CAUTION"
            st.metric("Flight Status", status, delta=f"{anomaly_count} anomalies")
        
        st.markdown("#### 🛰️ Trajectory Profile")
        
        fig_trajectory = go.Figure()
        
        fig_trajectory.add_trace(go.Scatter(
            x=telemetry_df['met_hours'],
            y=telemetry_df['altitude_km'],
            mode='lines',
            name='Altitude',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>MET:</b> %{x:.2f}h<br><b>Alt:</b> %{y:,.0f}km<extra></extra>'
        ))
        
        anomaly_points = telemetry_df[telemetry_df['is_anomaly']]
        if len(anomaly_points) > 0:
            fig_trajectory.add_trace(go.Scatter(
                x=anomaly_points['met_hours'],
                y=anomaly_points['altitude_km'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#ff0000', size=10, symbol='x'),
                hovertemplate='<b>⚠️ ANOMALY</b><br>MET: %{x:.2f}h<extra></extra>'
            ))
        
        fig_trajectory.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            xaxis=dict(title='MET (hours)', gridcolor='rgba(255,255,255,0.1)', showgrid=True),
            yaxis=dict(title='Altitude (km)', gridcolor='rgba(255,255,255,0.1)', showgrid=True),
            hovermode='x unified',
            height=500
        )
        
        # FIXED: Static key to prevent re-rendering
        st.plotly_chart(fig_trajectory, use_container_width=True, key="trajectory_main")
        
        st.markdown("#### 🔍 ML Anomaly Detection")
        st.markdown("*Ensemble: Isolation Forest + One-Class SVM on simulated telemetry*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = go.Figure()
            
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
                    colorbar=dict(title='Score')
                ),
                hovertemplate='<b>V:</b> %{x:.2f}<br><b>Alt:</b> %{y:,.0f}<extra></extra>'
            ))
            
            if len(anomaly_points) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=anomaly_points['velocity_km_s'],
                    y=anomaly_points['altitude_km'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#ff0000', size=12, symbol='x'),
                    hovertemplate='<b>⚠️</b> V: %{x:.2f}<extra></extra>'
                ))
            
            fig_scatter.update_layout(
                title='Feature Space',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Rajdhani'),
                xaxis=dict(title='Velocity (km/s)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='Altitude (km)', gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            
            # FIXED: Static key
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_anomaly")
        
        with col2:
            current_anomaly_score = telemetry_df.iloc[-1]['anomaly_score']
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_anomaly_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Anomaly Score", 'font': {'size': 20, 'color': '#ffffff'}},
                delta={'reference': 30, 'increasing': {'color': '#ff0000'}},
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
                    'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 70}
                }
            ))
            
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ffffff', 'family': 'Rajdhani'},
                height=400
            )
            
            # FIXED: Static key
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_anomaly")
    
    live_telemetry_display()

with tab2:
    st.markdown("### 🚀 TLI Burn Simulator")
    st.markdown("*Keplerian orbital mechanics for strategic planning*")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        thrust_pct = st.slider("Engine Thrust (%)", 80, 105, 100, 1, help="Nominal: 100%")
    
    with col2:
        burn_duration = st.slider("Burn Duration (s)", 250, 450, 350, 10, help="Nominal: 350s")
    
    with col3:
        ignition_vector = st.slider("Ignition Vector (°)", 0, 360, 90, 5, help="Thrust direction")
    
    if st.button("🔥 EXECUTE TLI BURN SIMULATION", type="primary"):
        with st.spinner("Propagating trajectory..."):
            outcome, success, trajectory_df = simulate_tli_burn(thrust_pct, burn_duration, ignition_vector)
            st.session_state.sim_outcome = outcome
            st.session_state.sim_success = success
            st.session_state.sim_trajectory = trajectory_df
    
    if 'sim_outcome' in st.session_state:
        st.markdown("---")
        
        outcome_color = "#00ff00" if st.session_state.sim_success else "#ff0000"
        st.markdown(f"""
        <div style="background: rgba(0,212,255,0.1); border: 3px solid {outcome_color}; 
                    border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center;">
            <h2 style="color: {outcome_color}; font-family: 'Rajdhani', sans-serif; font-size: 2rem; margin: 0;">
                {st.session_state.sim_outcome}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🌍➡️🌙 Simulated Trajectory")
        
        fig_sim = go.Figure()
        
        fig_sim.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers+text', name='Earth',
            marker=dict(size=30, color='#00d4ff'), text=['🌍'], textfont=dict(size=40)
        ))
        
        moon_x = 384400 * np.cos(np.radians(45))
        moon_y = 384400 * np.sin(np.radians(45))
        fig_sim.add_trace(go.Scatter(
            x=[moon_x], y=[moon_y], mode='markers+text', name='Moon',
            marker=dict(size=20, color='#ffd700'), text=['🌙'], textfont=dict(size=30)
        ))
        
        trajectory_color = '#00ff00' if st.session_state.sim_success else '#ff0000'
        fig_sim.add_trace(go.Scatter(
            x=st.session_state.sim_trajectory['x_km'],
            y=st.session_state.sim_trajectory['y_km'],
            mode='lines', name='Trajectory',
            line=dict(color=trajectory_color, width=3),
            hovertemplate='<b>T:</b> %{customdata:.1f}h<br><b>D:</b> %{text:,.0f}km<extra></extra>',
            customdata=st.session_state.sim_trajectory['time_hours'],
            text=st.session_state.sim_trajectory['distance_km']
        ))
        
        fig_sim.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            xaxis=dict(title='X (km)', gridcolor='rgba(255,255,255,0.1)', scaleanchor="y", scaleratio=1),
            yaxis=dict(title='Y (km)', gridcolor='rgba(255,255,255,0.1)'),
            height=600
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.markdown("#### 📊 Burn Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_v = 3.1 * (thrust_pct / 100) * (burn_duration / 350)
            st.metric("ΔV", f"{delta_v:.3f} km/s")
        
        with col2:
            final_dist = st.session_state.sim_trajectory.iloc[-1]['distance_km']
            st.metric("Final Distance", f"{final_dist:,.0f} km")
        
        with col3:
            efficiency = (thrust_pct / 100) * (burn_duration / 350) * 100
            st.metric("Efficiency", f"{efficiency:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="tech-footer">
    <h3 style="color: #00d4ff; font-family: 'Rajdhani', sans-serif; margin-bottom: 1rem;">
        🔬 Data Science Portfolio Project
    </h3>
    <div>
        <span class="tech-badge">Bayesian Inference</span>
        <span class="tech-badge">Beta-Binomial Model</span>
        <span class="tech-badge">Isolation Forest</span>
        <span class="tech-badge">One-Class SVM</span>
        <span class="tech-badge">Ensemble Learning</span>
        <span class="tech-badge">Keplerian Mechanics</span>
        <span class="tech-badge">Real-Time Simulation</span>
    </div>
    <p style="margin-top: 1.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);">
        Developed by Deep Rushil | GODSPEED ARTEMIS II | April 2026
    </p>
</div>
""", unsafe_allow_html=True)
