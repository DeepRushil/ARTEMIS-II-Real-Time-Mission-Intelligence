"""
Artemis II: Real-Time Mission Intelligence Dashboard
- Live NASA Horizons API telemetry with simulation fallback
- Bayesian Inference for mission success prediction
- Unsupervised ML for anomaly detection (Isolation Forest + One-Class SVM)
- Orbital mechanics simulation for strategic planning
- Animated deep-space background with glassmorphism UI
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Artemis II Mission Intelligence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Animated Galaxy Background + Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Animated Galaxy Background ── */
.stApp {
    background: transparent;
    color: #ffffff;
    font-family: 'Rajdhani', sans-serif;
    position: relative;
}

/* Full-page canvas background injected via JS – see below */
#galaxy-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}

/* Make Streamlit root transparent so canvas shows */
.stApp > div:first-child {
    background: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
[data-testid="stHeader"] {
    background: rgba(5,8,20,0.6) !important;
    backdrop-filter: blur(12px);
}

/* ── Glassmorphism Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 35, 0.45) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border-right: 1px solid rgba(0,212,255,0.2) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] > div {
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Mission Header ── */
.mission-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(0,212,255,0.4);
    margin-bottom: 2rem;
    position: relative;
    z-index: 10;
}
.mission-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 700;
    letter-spacing: 0.18em;
    background: linear-gradient(90deg, #00d4ff 0%, #ffffff 50%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    margin: 0;
    filter: drop-shadow(0 0 24px rgba(0,212,255,0.6));
}
.mission-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: rgba(0,212,255,0.85);
    letter-spacing: 0.35em;
    margin-top: 0.6rem;
    text-transform: uppercase;
}

/* ── Hero Metric ── */
.hero-metric {
    text-align: center;
    padding: 2rem;
    background: rgba(0,212,255,0.06);
    border: 1.5px solid rgba(0,212,255,0.5);
    border-radius: 20px;
    margin: 1.5rem auto;
    max-width: 560px;
    box-shadow: 0 0 50px rgba(0,212,255,0.25), inset 0 0 30px rgba(0,212,255,0.05);
    backdrop-filter: blur(16px);
    animation: heroPulse 4s ease-in-out infinite;
    position: relative;
    z-index: 10;
}
@keyframes heroPulse {
    0%,100% { box-shadow: 0 0 40px rgba(0,212,255,0.25), inset 0 0 30px rgba(0,212,255,0.05); }
    50%     { box-shadow: 0 0 70px rgba(0,212,255,0.5),  inset 0 0 40px rgba(0,212,255,0.1); }
}
.hero-value {
    font-size: 5.5rem;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'Rajdhani', sans-serif;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,212,255,0.9);
}
.hero-label {
    font-size: 1.1rem;
    color: #ffffff;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.82rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    line-height: 1.6;
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: rgba(0,212,255,0.08);
    border-color: rgba(0,212,255,0.6);
    box-shadow: 0 0 20px rgba(0,212,255,0.2);
    transform: translateY(-2px);
}

/* ── Crew Cards ── */
.crew-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px;
    padding: 0.9rem;
    margin: 0.7rem 0;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    cursor: pointer;
    text-decoration: none;
    display: block;
}
.crew-card:hover {
    background: rgba(0,212,255,0.1);
    border-color: #00d4ff;
    box-shadow: 0 0 22px rgba(0,212,255,0.3);
    transform: translateY(-2px);
}
.crew-name { font-size: 1.1rem; font-weight: 700; color: #00d4ff; margin-bottom: 0.2rem; }
.crew-role { font-size: 0.8rem; color: #ffd700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.crew-bio  { font-size: 0.78rem; color: rgba(255,255,255,0.75); line-height: 1.4; }

/* ── MET Timer ── */
.met-display {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffd700;
    text-align: center;
    padding: 0.9rem;
    background: rgba(255,215,0,0.08);
    border: 1.5px solid rgba(255,215,0,0.5);
    border-radius: 10px;
    margin: 0.8rem 0;
    text-shadow: 0 0 15px rgba(255,215,0,0.6);
    backdrop-filter: blur(8px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    background: rgba(0,0,0,0.25);
    padding: 0.8rem;
    border-radius: 12px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,212,255,0.15);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: rgba(255,255,255,0.55);
    border-radius: 8px;
    padding: 0.8rem 1.8rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.3) 0%, rgba(0,132,255,0.3) 100%);
    color: #ffffff;
    box-shadow: 0 0 20px rgba(0,212,255,0.35);
    border: 1px solid rgba(0,212,255,0.5);
}

/* ── Info / status banner ── */
.stAlert {
    background: rgba(0,212,255,0.07) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Update badge ── */
.update-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    margin: 0.25rem;
    background: rgba(0,212,255,0.15);
    border: 1px solid rgba(0,212,255,0.4);
    border-radius: 20px;
    font-size: 0.72rem;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
}

/* ── Footer ── */
.tech-footer {
    margin-top: 4rem;
    padding: 2rem;
    background: rgba(0,0,0,0.35);
    border-top: 1px solid rgba(0,212,255,0.3);
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    backdrop-filter: blur(12px);
}
.tech-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    margin: 0.3rem;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.4);
    border-radius: 20px;
    font-size: 0.72rem;
    color: #00d4ff;
}

/* ── General z-index for main content ── */
[data-testid="stVerticalBlock"] { position: relative; z-index: 10; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ANIMATED GALAXY + MOON BACKGROUND (Canvas)
# ═══════════════════════════════════════════════════════════════════════════

GALAXY_BG_HTML = """
<canvas id="galaxy-bg"></canvas>
<script>
(function(){
  const canvas = document.getElementById('galaxy-bg');
  const ctx = canvas.getContext('2d');
  let W, H, stars=[], nebulae=[], moonAngle=0;
  const NUM_STARS = 380;

  function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; }
  window.addEventListener('resize', resize);
  resize();

  // ── Stars ──
  for(let i=0;i<NUM_STARS;i++){
    stars.push({
      x: Math.random()*W, y: Math.random()*H,
      r: Math.random()*1.6+0.2,
      a: Math.random(),
      speed: Math.random()*0.004+0.001,
      twinkle: Math.random()*Math.PI*2
    });
  }

  // ── Nebula clusters ──
  const nebulaColors = [
    [0,180,255], [120,60,200], [255,80,120], [0,220,180]
  ];
  for(let i=0;i<6;i++){
    nebulae.push({
      x: Math.random()*W, y: Math.random()*H,
      r: 180+Math.random()*220,
      color: nebulaColors[i%nebulaColors.length],
      alpha: 0.025+Math.random()*0.04
    });
  }

  // ── Moon ──
  const moon = {
    x: W*0.82, y: H*0.18, r: 72,
    craters: Array.from({length:9}, ()=>({
      ox: (Math.random()-0.5)*100,
      oy: (Math.random()-0.5)*100,
      r:  4+Math.random()*18
    }))
  };

  let t=0;
  function draw(){
    t += 0.008;
    moonAngle = t*0.015;
    // resize guard
    if(W!==window.innerWidth||H!==window.innerHeight) resize();

    // Deep space gradient
    const bg = ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.75);
    bg.addColorStop(0, '#0c1024');
    bg.addColorStop(0.5,'#070b1a');
    bg.addColorStop(1,  '#020408');
    ctx.fillStyle = bg;
    ctx.fillRect(0,0,W,H);

    // Nebulae
    nebulae.forEach(n=>{
      const ng = ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r);
      ng.addColorStop(0,  `rgba(${n.color[0]},${n.color[1]},${n.color[2]},${n.alpha})`);
      ng.addColorStop(0.5,`rgba(${n.color[0]},${n.color[1]},${n.color[2]},${n.alpha*0.4})`);
      ng.addColorStop(1,  `rgba(0,0,0,0)`);
      ctx.fillStyle = ng;
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fill();
    });

    // Stars with twinkle
    stars.forEach(s=>{
      s.twinkle += s.speed;
      const alpha = 0.4 + 0.6*Math.abs(Math.sin(s.twinkle));
      ctx.beginPath();
      ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fill();
      // Occasional bright star spike
      if(s.r>1.2){
        ctx.strokeStyle=`rgba(255,255,255,${alpha*0.3})`;
        ctx.lineWidth=0.5;
        ctx.beginPath();
        ctx.moveTo(s.x-s.r*3,s.y); ctx.lineTo(s.x+s.r*3,s.y);
        ctx.moveTo(s.x,s.y-s.r*3); ctx.lineTo(s.x,s.y+s.r*3);
        ctx.stroke();
      }
    });

    // ── Moon ──
    const mx = W*0.82 + Math.sin(moonAngle)*8;
    const my = H*0.18 + Math.cos(moonAngle*0.7)*5;
    const mr = 72;

    // Glow halo
    const glow = ctx.createRadialGradient(mx,my,mr*0.8,mx,my,mr*2.8);
    glow.addColorStop(0,  'rgba(200,220,255,0.12)');
    glow.addColorStop(0.5,'rgba(120,170,255,0.05)');
    glow.addColorStop(1,  'rgba(0,0,0,0)');
    ctx.fillStyle=glow;
    ctx.beginPath(); ctx.arc(mx,my,mr*2.8,0,Math.PI*2); ctx.fill();

    // Moon body
    const moonGrad = ctx.createRadialGradient(mx-mr*0.25,my-mr*0.25,mr*0.1,mx,my,mr);
    moonGrad.addColorStop(0,'#e8eaf0');
    moonGrad.addColorStop(0.6,'#c5c9d8');
    moonGrad.addColorStop(1,'#8a8fa8');
    ctx.beginPath(); ctx.arc(mx,my,mr,0,Math.PI*2);
    ctx.fillStyle=moonGrad; ctx.fill();

    // Moon craters
    moon.craters.forEach(c=>{
      const cx2=mx+c.ox, cy2=my+c.oy;
      if(Math.sqrt(c.ox*c.ox+c.oy*c.oy)+c.r>mr) return;
      const cg=ctx.createRadialGradient(cx2,cy2,0,cx2,cy2,c.r);
      cg.addColorStop(0,'rgba(80,85,105,0.55)');
      cg.addColorStop(1,'rgba(100,105,125,0)');
      ctx.beginPath(); ctx.arc(cx2,cy2,c.r,0,Math.PI*2);
      ctx.fillStyle=cg; ctx.fill();
    });

    // Terminator shadow
    ctx.save();
    ctx.beginPath(); ctx.arc(mx,my,mr,0,Math.PI*2);
    ctx.clip();
    const shadowX = mx+mr*0.35;
    const termGrad = ctx.createRadialGradient(shadowX,my,0,shadowX,my,mr*1.1);
    termGrad.addColorStop(0,'rgba(5,8,20,0)');
    termGrad.addColorStop(0.4,'rgba(5,8,20,0.35)');
    termGrad.addColorStop(1,'rgba(5,8,20,0.78)');
    ctx.fillStyle=termGrad; ctx.fillRect(mx-mr,my-mr,mr*2,mr*2);
    ctx.restore();

    // Shooting star occasionally
    if(Math.random()<0.004){
      const sx=Math.random()*W*0.6, sy=Math.random()*H*0.4;
      const len=80+Math.random()*120;
      const sg=ctx.createLinearGradient(sx,sy,sx+len,sy+len*0.4);
      sg.addColorStop(0,'rgba(255,255,255,0)');
      sg.addColorStop(0.5,'rgba(255,255,255,0.9)');
      sg.addColorStop(1,'rgba(255,255,255,0)');
      ctx.strokeStyle=sg; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(sx+len,sy+len*0.4);
      ctx.stroke();
    }

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

components.html(GALAXY_BG_HTML, height=0)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MISSION_LAUNCH_UTC = datetime(2026, 4, 1, 22, 35, 0)

MISSION_MILESTONES = {
    "Launch":          {"time_offset_minutes": 0},
    "ICPS Separation": {"time_offset_minutes": 120},
    "Perigee Raise":   {"time_offset_minutes": 600},
    "TLI Burn":        {"time_offset_minutes": 1440},
    "Earth Departure": {"time_offset_minutes": 2880},
    "Lunar Flyby":     {"time_offset_minutes": 5760},
}

CREW_DATA = [
    {"name":"Reid Wiseman",  "role":"Commander",         "emoji":"👨‍🚀",
     "bio":"NASA astronaut with ISS experience (Expedition 40/41). Led spacewalks and served as Chief of the Astronaut Office.",
     "url":"https://www.nasa.gov/people/reid-wiseman/"},
    {"name":"Victor Glover", "role":"Pilot",             "emoji":"👨‍✈️",
     "bio":"Naval aviator and NASA astronaut. First African American on ISS long-duration crew (SpaceX Crew-1).",
     "url":"https://www.nasa.gov/humans-in-space/astronauts/victor-j-glover/"},
    {"name":"Christina Koch","role":"Mission Specialist","emoji":"👩‍🚀",
     "bio":"Record holder for longest single spaceflight by a woman (328 days). Conducted first all-female spacewalk.",
     "url":"https://www.nasa.gov/humans-in-space/astronauts/christina-koch/"},
    {"name":"Jeremy Hansen", "role":"Mission Specialist","emoji":"👨‍🚀",
     "bio":"CSA astronaut and CF-18 fighter pilot. First Canadian to travel beyond low Earth orbit.",
     "url":"https://www.asc-csa.gc.ca/eng/astronauts/canadian/active/bio-jeremy-hansen.asp"},
]

# ═══════════════════════════════════════════════════════════════════════════
# NASA HORIZONS API — Real telemetry with simulation fallback
# ═══════════════════════════════════════════════════════════════════════════

# Orion NAIF ID for Artemis missions (Orion spacecraft SPICE target)
# Horizons uses "Orion" or target body code; we'll use the Horizons web API
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

def fetch_horizons_telemetry():
    """
    Query NASA JPL Horizons for real Orion/Artemis II position data.
    Returns dict or None on failure.
    """
    try:
        now_utc = datetime.utcnow()
        stop_utc = now_utc + timedelta(minutes=1)
        fmt = "%Y-%m-%d %H:%M"

        params = {
            "format": "json",
            "COMMAND": "'Orion'",          # Artemis II Orion spacecraft
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": "'500@399'",          # Geocentric (Earth center)
            "START_TIME": f"'{now_utc.strftime(fmt)}'",
            "STOP_TIME":  f"'{stop_utc.strftime(fmt)}'",
            "STEP_SIZE":  "'1 m'",
            "VEC_TABLE":  "3",              # position + velocity
            "REF_PLANE":  "ECLIPTIC",
            "REF_SYSTEM": "J2000",
            "OUT_UNITS":  "KM-S",
            "CSV_FORMAT": "NO",
        }

        resp = requests.get(HORIZONS_URL, params=params, timeout=6)
        if resp.status_code != 200:
            return None

        data = resp.json()
        result_text = data.get("result", "")

        # Parse $$SOE … $$EOE block
        if "$$SOE" not in result_text:
            return None

        soe_idx = result_text.index("$$SOE") + 5
        eoe_idx = result_text.index("$$EOE")
        block = result_text[soe_idx:eoe_idx].strip()
        lines = [l.strip() for l in block.splitlines() if l.strip()]

        # Horizons vector table: line 1 = time, line 2 = X Y Z, line 3 = VX VY VZ
        if len(lines) < 3:
            return None

        pos_vals = lines[1].split()
        vel_vals = lines[2].split()

        x_km = float(pos_vals[0])
        y_km = float(pos_vals[1])
        z_km = float(pos_vals[2])
        vx   = float(vel_vals[0])
        vy   = float(vel_vals[1])
        vz   = float(vel_vals[2])

        distance_earth = np.sqrt(x_km**2 + y_km**2 + z_km**2)
        velocity       = np.sqrt(vx**2 + vy**2 + vz**2)

        # Moon distance (approximate — Moon ~ 384400 km from Earth; use geometry)
        # For simplicity, distance_moon ≈ |moon_pos - craft_pos|
        # Moon pos approximation via angle
        elapsed_days = (now_utc - MISSION_LAUNCH_UTC).total_seconds() / 86400
        moon_angle_rad = elapsed_days * (2 * np.pi / 27.3)  # synodic period
        moon_x = 384400 * np.cos(moon_angle_rad)
        moon_y = 384400 * np.sin(moon_angle_rad)
        distance_moon = np.sqrt((x_km - moon_x)**2 + (y_km - moon_y)**2)

        return {
            "distance_earth_km": distance_earth,
            "distance_moon_km":  distance_moon,
            "velocity_km_s":     velocity,
            "altitude_km":       distance_earth - 6371,
            "x_km": x_km, "y_km": y_km, "z_km": z_km,
            "source": "NASA_Horizons",
        }

    except Exception:
        return None


def calculate_mission_profile_sim():
    """Physics-based fallback simulation."""
    now = datetime.utcnow()
    elapsed_hours = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600

    if elapsed_hours < 24:
        perigee, apogee = 300, 70000
        a = (perigee + apogee) / 2 + 6371
        r = perigee + 6371 + (apogee - perigee) * abs(np.sin(elapsed_hours / 24 * np.pi))
        velocity = np.sqrt(398600 * (2 / r - 1 / a))
        altitude = r - 6371
        distance_earth = altitude
        distance_moon  = 384400 - altitude
        if 1.9 < elapsed_hours < 2.1:
            altitude  += np.random.normal(0, 80)
            velocity  += np.random.normal(0, 0.25)
    elif elapsed_hours < 96:
        progress = (elapsed_hours - 24) / 72
        distance_earth = 70000 + progress * (384400 - 70000)
        distance_moon  = 384400 - distance_earth
        altitude = distance_earth
        velocity = 10.0 - progress * 9.0
        if 23.8 < elapsed_hours < 24.3:
            velocity += np.random.normal(0, 0.35)
    elif elapsed_hours < 120:
        distance_earth = 384400 + 10000
        distance_moon  = 300
        altitude = distance_earth
        velocity = 2.0
    else:
        progress = min(1.0, (elapsed_hours - 120) / 120)
        distance_earth = 400000 - progress * 399700
        distance_moon  = abs(384400 - distance_earth)
        altitude = distance_earth
        velocity = 1.0 + progress * 10.0

    return {
        "distance_earth_km": float(distance_earth),
        "distance_moon_km":  float(distance_moon),
        "velocity_km_s":     float(velocity),
        "altitude_km":       float(altitude),
        "source": "simulation",
    }


def get_live_telemetry():
    """
    Attempt NASA Horizons first; fall back to simulation.
    Stores history in session_state for anomaly detection.
    """
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = []
        st.session_state.last_telemetry_update = None
        st.session_state.data_source = "simulation"

    now = datetime.utcnow()
    should_add = (
        st.session_state.last_telemetry_update is None or
        (now - st.session_state.last_telemetry_update).total_seconds() >= 5
    )

    if should_add:
        live = fetch_horizons_telemetry()
        if live:
            current = live
            st.session_state.data_source = "NASA_Horizons"
        else:
            current = calculate_mission_profile_sim()
            current["altitude_km"]   += np.random.normal(0, 10)
            current["velocity_km_s"] += np.random.normal(0, 0.04)
            st.session_state.data_source = "simulation"

        st.session_state.current_telemetry = current
        elapsed_hours = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600

        st.session_state.telemetry_history.append({
            "timestamp":    now,
            "altitude_km":  current["altitude_km"],
            "velocity_km_s":current["velocity_km_s"],
            "met_hours":    elapsed_hours,
            "source":       current["source"],
        })
        st.session_state.last_telemetry_update = now

    # Memory cap
    if len(st.session_state.telemetry_history) > 120:
        st.session_state.telemetry_history = st.session_state.telemetry_history[-120:]

    return pd.DataFrame(st.session_state.telemetry_history)


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN MODEL
# ═══════════════════════════════════════════════════════════════════════════

class BayesianMissionSuccess:
    def __init__(self, alpha=95, beta=5):
        self.alpha = alpha
        self.beta  = beta

    def update(self, successes, failures):
        self.alpha += successes
        self.beta  += failures

    def predict(self):
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, conf=0.95):
        lo = stats.beta.ppf((1 - conf) / 2, self.alpha, self.beta)
        hi = stats.beta.ppf(1 - (1 - conf) / 2, self.alpha, self.beta)
        return lo, hi


def get_milestone_status():
    now = datetime.utcnow()
    elapsed_min = (now - MISSION_LAUNCH_UTC).total_seconds() / 60
    out = {}
    for name, data in MISSION_MILESTONES.items():
        offset = data["time_offset_minutes"]
        remaining = offset - elapsed_min
        out[name] = {
            "completed": elapsed_min >= offset,
            "time_offset_minutes": offset,
            "time_remaining_minutes": max(0, remaining),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(df):
    if len(df) < 10:
        return np.zeros(len(df)), np.ones(len(df))
    features = df[["altitude_km", "velocity_km_s"]].values
    iso  = IsolationForest(contamination=0.05, random_state=42)
    svm  = OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
    i_pred = iso.fit_predict(features)
    s_pred = svm.fit_predict(features)
    i_score = iso.score_samples(features)
    ensemble = np.where((i_pred == -1) & (s_pred == -1), -1, 1)
    score_norm = 1 - (i_score - i_score.min()) / (i_score.max() - i_score.min() + 1e-10)
    return score_norm, ensemble


# ═══════════════════════════════════════════════════════════════════════════
# TLI SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tli_burn(thrust_pct, burn_dur, vector_deg):
    GM = 398600
    r0 = 70000 + 6371
    v0 = np.sqrt(GM / r0)
    dv = 3.1 * (thrust_pct / 100) * (burn_dur / 350)
    v1 = v0 + dv
    v_esc = np.sqrt(2 * GM / r0)
    n = 200
    ts = np.linspace(0, 100, n)
    traj = []
    for t in ts:
        if v1 >= v_esc:
            d = r0 + v1 * t * 3600
        else:
            energy = v1**2 / 2 - GM / r0
            a = -GM / (2 * energy)
            mm = np.sqrt(GM / a**3)
            ang = mm * t * 3600
            d = a * (1 + 0.5 * np.cos(ang))
        traj.append({
            "time_hours": t,
            "distance_km": d,
            "x_km": d * np.cos(np.radians(vector_deg + t * 15)),
            "y_km": d * np.sin(np.radians(vector_deg + t * 15)),
        })
    df = pd.DataFrame(traj)
    fd = df.iloc[-1]["distance_km"]
    if v1 < v_esc * 0.95:
        outcome, ok = "❌ FAILED CAPTURE: Insufficient velocity — Remains in Earth orbit", False
    elif fd > 384400 * 1.5:
        outcome, ok = "❌ DEEP SPACE DRIFT: Excessive velocity — Missed lunar flyby", False
    elif fd < 384400 * 0.8:
        outcome, ok = "⚠️ LUNAR IMPACT: Trajectory intercepts Moon surface", False
    else:
        outcome, ok = "✅ SUCCESSFUL LUNAR FLYBY: Nominal trajectory achieved", True
    return outcome, ok, df


# ═══════════════════════════════════════════════════════════════════════════
# LIVE MET TIMER
# ═══════════════════════════════════════════════════════════════════════════

def display_live_met_timer():
    ts_ms = int(MISSION_LAUNCH_UTC.timestamp() * 1000)
    html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');
body{{margin:0;padding:0;background:transparent}}
.met{{font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;color:#ffd700;
      text-align:center;padding:.85rem;background:rgba(255,215,0,0.08);
      border:1.5px solid rgba(255,215,0,0.55);border-radius:10px;
      text-shadow:0 0 14px rgba(255,215,0,0.7);backdrop-filter:blur(8px)}}
.lbl{{font-size:.7rem;color:rgba(255,215,0,0.6);margin-top:.4rem;letter-spacing:.15em}}
</style>
<div class="met">
  <div id="met">00:00:00:00</div>
  <div class="lbl">DAYS · HOURS · MINS · SECS</div>
</div>
<script>
(function(){{
  var launch={ts_ms};
  function tick(){{
    var e=Math.max(0,Date.now()-launch);
    var d=Math.floor(e/864e5),h=Math.floor(e%864e5/36e5),
        m=Math.floor(e%36e5/6e4),s=Math.floor(e%6e4/1e3);
    document.getElementById('met').textContent=
      String(d).padStart(2,'0')+':'+String(h).padStart(2,'0')+':'+
      String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
  }}
  tick(); setInterval(tick,1000);
}})();
</script>
"""
    components.html(html, height=110)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mission-header">
  <h1 class="mission-title">ARTEMIS II</h1>
  <p class="mission-subtitle">Real-Time Mission Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.markdown("""
    <div style="text-align:center;padding:.8rem;">
      <div style="font-size:4rem;margin-bottom:.6rem;">🚀🌙</div>
      <p style="color:rgba(255,255,255,0.5);font-family:'Space Mono',monospace;font-size:.8rem;">
        ORION SPACECRAFT · SLS BLOCK 1 · LIVE + SIMULATED DATA
      </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HERO — Mission Success (auto-refresh)
# ═══════════════════════════════════════════════════════════════════════════

@st.fragment(run_every="5s")
def display_mission_success():
    ms = get_milestone_status()
    completed = sum(1 for v in ms.values() if v["completed"])
    model = BayesianMissionSuccess()
    penalty = 0
    if "telemetry_history" in st.session_state and len(st.session_state.telemetry_history) >= 10:
        df_tmp = pd.DataFrame(st.session_state.telemetry_history[-20:])
        if len(df_tmp) >= 10:
            _, preds = detect_anomalies(df_tmp)
            penalty = int((preds == -1).mean() * 3)
    model.update(completed, penalty)
    prob = model.predict()
    lo, hi = model.credible_interval()
    src = st.session_state.get("data_source", "simulation")
    src_label = "🛰️ NASA Horizons API" if src == "NASA_Horizons" else "⚙️ Physics Simulation"
    penalty_txt = f" · Anomaly Penalty: {penalty}" if penalty else ""

    st.markdown(f"""
    <div class="hero-metric">
      <div class="hero-value">{prob*100:.1f}%</div>
      <div class="hero-label">Mission Success Probability</div>
      <div class="hero-sub">
        Bayesian Beta-Binomial · 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]<br>
        <span style="color:#00d4ff;">Milestones: {completed}/{len(ms)} complete</span>{penalty_txt}<br>
        <span style="color:#ffd700;">Data: {src_label}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

display_mission_success()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 👥 CREW MANIFEST")
    for c in CREW_DATA:
        st.markdown(f"""
        <a href="{c['url']}" target="_blank" class="crew-card">
          <div style="font-size:2.2rem;text-align:center;margin-bottom:.4rem;">{c['emoji']}</div>
          <div class="crew-name">{c['name']}</div>
          <div class="crew-role">{c['role']}</div>
          <div class="crew-bio">{c['bio']}</div>
        </a>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⏱️ MISSION ELAPSED TIME")
    display_live_met_timer()

    # Update rate info
    st.markdown("""
    <div class="glass-card" style="margin-top:.5rem;">
      <div style="font-size:.68rem;color:rgba(255,255,255,0.6);font-family:'Space Mono',monospace;line-height:1.8;">
        🔄 <b>Update Rates</b><br>
        · MET Timer: 1 s (real-time)<br>
        · Telemetry:  5 s<br>
        · Milestones: 5 s<br>
        · Prediction: 5 s
      </div>
    </div>
    """, unsafe_allow_html=True)

    @st.fragment(run_every="5s")
    def sidebar_phase():
        now = datetime.utcnow()
        h = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600
        if h < 24:       phase, col = "🌍 High Earth Orbit",  "#00d4ff"
        elif h < 96:     phase, col = "🚀 Translunar Coast",  "#ffd700"
        elif h < 120:    phase, col = "🌙 Lunar Flyby",       "#ffffff"
        else:            phase, col = "🏠 Earth Return",      "#00ff00"
        st.markdown(f"""
        <div style="text-align:center;padding:.7rem;background:rgba(0,0,0,0.25);
                    border:1px solid {col};border-radius:9px;margin:.8rem 0;backdrop-filter:blur(8px);">
          <div style="color:{col};font-family:'Space Mono',monospace;font-size:.88rem;font-weight:600;">{phase}</div>
        </div>""", unsafe_allow_html=True)
    sidebar_phase()

    st.markdown("---")
    st.markdown("### 🎯 MILESTONE STATUS")

    @st.fragment(run_every="5s")
    def sidebar_milestones():
        ms = get_milestone_status()
        for name, data in ms.items():
            icon  = "✅" if data["completed"] else "⏳"
            color = "#00ff00" if data["completed"] else "#ffd700"
            rem   = ""
            if not data["completed"] and data["time_remaining_minutes"] < 180:
                hh = int(data["time_remaining_minutes"] // 60)
                mm = int(data["time_remaining_minutes"] % 60)
                rem = f" ({hh}h {mm}m)"
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin:.45rem 0;font-family:'Space Mono',monospace;">
              <span style="font-size:1.1rem;margin-right:.5rem;">{icon}</span>
              <span style="color:{color};font-size:.8rem;">{name}{rem}</span>
            </div>""", unsafe_allow_html=True)
    sidebar_milestones()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📡 Live Mission Feed", "🧪 Strategic Simulation"])

# ── TAB 1: LIVE FEED ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Real-Time Telemetry Analysis")

    @st.fragment(run_every="5s")
    def live_telemetry_display():
        tdf = get_live_telemetry()
        src = st.session_state.get("data_source", "simulation")
        utc_str = datetime.utcnow().strftime("%H:%M:%S UTC")
        src_txt = "🛰️ **NASA Horizons API** (live)" if src == "NASA_Horizons" else "⚙️ **Physics Simulation** (NASA API unavailable)"
        st.info(f"{src_txt} · Last update: {utc_str} · Refresh: 5 s")

        if len(tdf) < 2:
            st.warning("Collecting telemetry…")
            return

        scores, preds = detect_anomalies(tdf)
        tdf = tdf.copy()
        tdf["anomaly_score"] = scores
        tdf["is_anomaly"]    = preds == -1

        cur = st.session_state.get("current_telemetry", {})
        c1,c2,c3,c4,c5 = st.columns(5)

        with c1:
            alt = tdf.iloc[-1]["altitude_km"]
            d_alt = alt - tdf.iloc[-2]["altitude_km"]
            st.metric("Altitude", f"{alt:,.0f} km", f"{d_alt:+.0f} km")
        with c2:
            vel = tdf.iloc[-1]["velocity_km_s"]
            d_vel = vel - tdf.iloc[-2]["velocity_km_s"]
            st.metric("Velocity", f"{vel:.3f} km/s", f"{d_vel:+.3f} km/s")
        with c3:
            de = cur.get("distance_earth_km", alt)
            st.metric("Dist. Earth", f"{de:,.0f} km", f"{de/6371:.2f} R⊕")
        with c4:
            dm = cur.get("distance_moon_km", 384400)
            pct = 100*(1-dm/384400)
            st.metric("Dist. Moon", f"{dm:,.0f} km", f"{pct:+.1f}%")
        with c5:
            a_cnt = int(tdf["is_anomaly"].sum())
            avg_s = tdf["anomaly_score"].mean()
            status = "NOMINAL" if avg_s < 0.3 else "⚠️ CAUTION"
            st.metric("Status", status, f"{a_cnt} anomalies")

        # ── Trajectory chart ──
        st.markdown("#### 🛰️ Trajectory Profile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tdf["met_hours"], y=tdf["altitude_km"],
            mode="lines", name="Altitude",
            line=dict(color="#00d4ff", width=2.5),
            hovertemplate="<b>MET:</b> %{x:.2f}h<br><b>Alt:</b> %{y:,.0f} km<extra></extra>"
        ))
        anom = tdf[tdf["is_anomaly"]]
        if len(anom):
            fig.add_trace(go.Scatter(
                x=anom["met_hours"], y=anom["altitude_km"],
                mode="markers", name="Anomaly",
                marker=dict(color="#ff3a3a", size=10, symbol="x"),
                hovertemplate="<b>⚠️ ANOMALY</b><br>MET: %{x:.2f}h<extra></extra>"
            ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff", family="Rajdhani"),
            xaxis=dict(title="MET (hours)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Altitude (km)", gridcolor="rgba(255,255,255,0.08)"),
            hovermode="x unified", height=440,
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(0,212,255,0.3)")
        )
        st.plotly_chart(fig, use_container_width=True, key="traj_main")

        # ── Velocity chart ──
        st.markdown("#### ⚡ Velocity Profile")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=tdf["met_hours"], y=tdf["velocity_km_s"],
            mode="lines", name="Velocity",
            line=dict(color="#ffd700", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,215,0,0.07)"
        ))
        fig_v.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff", family="Rajdhani"),
            xaxis=dict(title="MET (hours)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Velocity (km/s)", gridcolor="rgba(255,255,255,0.08)"),
            height=320
        )
        st.plotly_chart(fig_v, use_container_width=True, key="vel_main")

        # ── Anomaly section ──
        st.markdown("#### 🔍 ML Anomaly Detection")
        st.markdown("*Ensemble: Isolation Forest + One-Class SVM*")
        a1, a2 = st.columns(2)

        with a1:
            fig_sc = go.Figure()
            norm = tdf[~tdf["is_anomaly"]]
            fig_sc.add_trace(go.Scatter(
                x=norm["velocity_km_s"], y=norm["altitude_km"],
                mode="markers", name="Nominal",
                marker=dict(color=norm["anomaly_score"], colorscale="Plasma",
                            size=7, colorbar=dict(title="Score", thickness=10))
            ))
            if len(anom):
                fig_sc.add_trace(go.Scatter(
                    x=anom["velocity_km_s"], y=anom["altitude_km"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#ff3a3a", size=11, symbol="x")
                ))
            fig_sc.update_layout(
                title="Feature Space", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#fff", family="Rajdhani"),
                xaxis=dict(title="Velocity (km/s)", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="Altitude (km)", gridcolor="rgba(255,255,255,0.08)"),
                height=400
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="scatter_anom")

        with a2:
            cur_score = float(tdf.iloc[-1]["anomaly_score"]) * 100
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cur_score,
                title={"text": "Anomaly Score", "font": {"size": 18, "color": "#fff"}},
                delta={"reference": 30, "increasing": {"color": "#ff3a3a"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#fff"},
                    "bar": {"color": "#00d4ff"},
                    "bgcolor": "rgba(0,0,0,0.2)",
                    "borderwidth": 1.5, "bordercolor": "#fff",
                    "steps": [
                        {"range": [0,  30], "color": "rgba(0,255,100,0.15)"},
                        {"range": [30, 70], "color": "rgba(255,215,0,0.15)"},
                        {"range": [70,100], "color": "rgba(255,50,50,0.2)"},
                    ],
                    "threshold": {"line": {"color": "red","width": 3}, "value": 70}
                }
            ))
            fig_g.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#fff", "family": "Rajdhani"}, height=400
            )
            st.plotly_chart(fig_g, use_container_width=True, key="gauge_anom")

    live_telemetry_display()

# ── TAB 2: SIMULATION ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🚀 TLI Burn Simulator")
    st.markdown("*Keplerian orbital mechanics for strategic planning*")
    st.markdown("---")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        thrust_pct   = st.slider("Engine Thrust (%)",     80, 105, 100, 1,  help="Nominal: 100%")
    with sc2:
        burn_duration = st.slider("Burn Duration (s)",    250, 450, 350, 10, help="Nominal: 350 s")
    with sc3:
        ignition_vec  = st.slider("Ignition Vector (°)",  0,  360,  90,  5, help="Thrust direction")

    if st.button("🔥 EXECUTE TLI BURN SIMULATION", type="primary"):
        with st.spinner("Propagating trajectory…"):
            outcome, ok, traj_df = simulate_tli_burn(thrust_pct, burn_duration, ignition_vec)
            st.session_state.sim_outcome   = outcome
            st.session_state.sim_success   = ok
            st.session_state.sim_trajectory = traj_df

    if "sim_outcome" in st.session_state:
        st.markdown("---")
        oc = "#00ff88" if st.session_state.sim_success else "#ff3a3a"
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3);border:2.5px solid {oc};border-radius:16px;
                    padding:1.8rem;margin:1.5rem 0;text-align:center;backdrop-filter:blur(12px);">
          <h2 style="color:{oc};font-family:'Rajdhani',sans-serif;font-size:1.9rem;margin:0;">
            {st.session_state.sim_outcome}
          </h2>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🌍 ➡️ 🌙 Simulated Trajectory")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=[0],y=[0],mode="markers+text",name="Earth",
            marker=dict(size=28,color="#00d4ff"),text=["🌍"],textfont=dict(size=36)))
        mx = 384400*np.cos(np.radians(45)); my = 384400*np.sin(np.radians(45))
        fig_s.add_trace(go.Scatter(x=[mx],y=[my],mode="markers+text",name="Moon",
            marker=dict(size=18,color="#ffd700"),text=["🌙"],textfont=dict(size=28)))
        tc = "#00ff88" if st.session_state.sim_success else "#ff3a3a"
        fig_s.add_trace(go.Scatter(
            x=st.session_state.sim_trajectory["x_km"],
            y=st.session_state.sim_trajectory["y_km"],
            mode="lines", name="Trajectory", line=dict(color=tc, width=2.5),
        ))
        fig_s.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff", family="Rajdhani"),
            xaxis=dict(title="X (km)", gridcolor="rgba(255,255,255,0.07)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (km)", gridcolor="rgba(255,255,255,0.07)"),
            height=580
        )
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("#### 📊 Burn Analysis")
        ba1, ba2, ba3 = st.columns(3)
        dv = 3.1*(thrust_pct/100)*(burn_duration/350)
        with ba1: st.metric("ΔV",              f"{dv:.3f} km/s")
        with ba2: st.metric("Final Distance",  f"{st.session_state.sim_trajectory.iloc[-1]['distance_km']:,.0f} km")
        with ba3: st.metric("Burn Efficiency", f"{(thrust_pct/100)*(burn_duration/350)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="tech-footer">
  <h3 style="color:#00d4ff;font-family:'Rajdhani',sans-serif;margin-bottom:1rem;">
    🔬 Data Science Portfolio Project
  </h3>
  <div>
    <span class="tech-badge">Bayesian Inference</span>
    <span class="tech-badge">Beta-Binomial Model</span>
    <span class="tech-badge">Isolation Forest</span>
    <span class="tech-badge">One-Class SVM</span>
    <span class="tech-badge">Ensemble Learning</span>
    <span class="tech-badge">Keplerian Mechanics</span>
    <span class="tech-badge">NASA Horizons API</span>
    <span class="tech-badge">Real-Time Telemetry</span>
  </div>
  <p style="margin-top:1.5rem;font-size:.72rem;color:rgba(255,255,255,0.35);">
    Developed by Deep Rushil · GODSPEED ARTEMIS II · April 2026
  </p>
</div>
""", unsafe_allow_html=True)
