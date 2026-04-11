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
from datetime import datetime, timedelta, timezone
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
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body { background: #0d1128 !important; margin: 0; padding: 0; }

.stApp {
    background: transparent !important;
    color: #ffffff;
    font-family: 'Rajdhani', sans-serif;
}
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stMain"]             { background: transparent !important; }
.main .block-container             { background: transparent !important; padding-top: 1rem; }

[data-testid="stHeader"] {
    background: rgba(5, 8, 20, 0.70) !important;
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
    background: rgba(8, 12, 30, 0.55) !important;
    backdrop-filter: blur(28px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(160%) !important;
    border-right: 1px solid rgba(0,212,255,0.18) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] > div { background: transparent !important; }

[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.element-container, .stMarkdown { position: relative; z-index: 10; background: transparent !important; }

[data-testid="stMetric"] {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
    backdrop-filter: blur(10px) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

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

.stAlert {
    background: rgba(0,212,255,0.07) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    backdrop-filter: blur(8px) !important;
}

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

/* Thank-you banner buttons */
.nasa-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.75rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-decoration: none !important;
    text-transform: uppercase;
    transition: background 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    position: relative;
    z-index: 20;
}
.nasa-btn-cyan {
    background: rgba(0,212,255,0.12);
    border: 1.5px solid rgba(0,212,255,0.55);
    color: #00d4ff !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.15);
}
.nasa-btn-cyan:hover {
    background: rgba(0,212,255,0.28) !important;
    box-shadow: 0 0 32px rgba(0,212,255,0.45) !important;
    transform: translateY(-3px);
}
.nasa-btn-gold {
    background: rgba(255,215,0,0.10);
    border: 1.5px solid rgba(255,215,0,0.55);
    color: #ffd700 !important;
    box-shadow: 0 0 20px rgba(255,215,0,0.12);
}
.nasa-btn-gold:hover {
    background: rgba(255,215,0,0.26) !important;
    box-shadow: 0 0 32px rgba(255,215,0,0.40) !important;
    transform: translateY(-3px);
}
.thankyou-wrap {
    max-width: 860px;
    margin: 1.5rem auto 1.5rem;
    padding: 2.4rem 2.8rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.07) 0%, rgba(255,215,0,0.05) 100%);
    border: 1.5px solid rgba(0,212,255,0.35);
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(16px);
    box-shadow: 0 0 60px rgba(0,212,255,0.12), inset 0 0 40px rgba(255,215,0,0.04);
    position: relative;
    z-index: 10;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ANIMATED GALAXY + MOON BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<canvas id="artemis-galaxy-canvas" style="
  position:fixed;top:0;left:0;
  width:100vw;height:100vh;
  z-index:0;
  pointer-events:none;
  display:block;
"></canvas>

<script>
(function(){
  if(window.__artemisGalaxyRunning) return;
  window.__artemisGalaxyRunning = true;

  function boot(){
    const canvas = document.getElementById('artemis-galaxy-canvas');
    if(!canvas){ setTimeout(boot, 50); return; }
    const ctx = canvas.getContext('2d');

    let W, H, stars=[], nebulae=[], t=0;
    const NUM_STARS = 360;

    function resize(){
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    for(let i=0;i<NUM_STARS;i++){
      stars.push({
        x: Math.random()*W, y: Math.random()*H,
        r: Math.random()*1.6+0.2,
        speed: Math.random()*0.004+0.001,
        twinkle: Math.random()*Math.PI*2
      });
    }

    const NC = [[0,180,255],[120,60,200],[255,80,120],[0,220,180],[80,0,200],[0,160,220]];
    for(let i=0;i<6;i++){
      nebulae.push({
        x: Math.random()*W, y: Math.random()*H,
        r: 180+Math.random()*240,
        c: NC[i], a: 0.025+Math.random()*0.04
      });
    }

    const craters = [
      {ox:-18,oy:-22,r:9},{ox:14,oy:-8,r:6},{ox:-5,oy:18,r:11},
      {ox:24,oy:12,r:5},{ox:-28,oy:8,r:7},{ox:8,oy:-28,r:8},
      {ox:-12,oy:28,r:5},{ox:26,oy:-20,r:6},{ox:-22,oy:-5,r:4}
    ];

    let shooters=[];
    function spawnShooter(){
      shooters.push({
        x:Math.random()*W*0.7, y:Math.random()*H*0.45,
        len:90+Math.random()*130, life:1.0,
        dx:1, dy:0.38
      });
    }

    function draw(){
      t += 0.008;
      if(W!==window.innerWidth||H!==window.innerHeight) resize();

      const bg = ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.8);
      bg.addColorStop(0,'#0d1128');
      bg.addColorStop(0.5,'#070b1a');
      bg.addColorStop(1,'#020408');
      ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);

      nebulae.forEach(n=>{
        const ng=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r);
        ng.addColorStop(0,`rgba(${n.c[0]},${n.c[1]},${n.c[2]},${n.a})`);
        ng.addColorStop(0.55,`rgba(${n.c[0]},${n.c[1]},${n.c[2]},${n.a*0.35})`);
        ng.addColorStop(1,'rgba(0,0,0,0)');
        ctx.fillStyle=ng;
        ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2); ctx.fill();
      });

      stars.forEach(s=>{
        s.twinkle+=s.speed;
        const al=0.35+0.65*Math.abs(Math.sin(s.twinkle));
        ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
        ctx.fillStyle=`rgba(255,255,255,${al})`; ctx.fill();
        if(s.r>1.15){
          ctx.strokeStyle=`rgba(255,255,255,${al*0.28})`;
          ctx.lineWidth=0.5;
          ctx.beginPath();
          ctx.moveTo(s.x-s.r*3,s.y); ctx.lineTo(s.x+s.r*3,s.y);
          ctx.moveTo(s.x,s.y-s.r*3); ctx.lineTo(s.x,s.y+s.r*3);
          ctx.stroke();
        }
      });

      const moonAngle = t*0.014;
      const mx = W*0.83+Math.sin(moonAngle)*7;
      const my = H*0.17+Math.cos(moonAngle*0.65)*5;
      const mr = Math.min(W,H)*0.075;

      const atm=ctx.createRadialGradient(mx,my,mr*0.7,mx,my,mr*3.2);
      atm.addColorStop(0,'rgba(180,210,255,0.10)');
      atm.addColorStop(0.4,'rgba(100,160,255,0.04)');
      atm.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=atm;
      ctx.beginPath(); ctx.arc(mx,my,mr*3.2,0,Math.PI*2); ctx.fill();

      const surf=ctx.createRadialGradient(mx-mr*0.28,my-mr*0.28,mr*0.05,mx,my,mr);
      surf.addColorStop(0,'#eceef6');
      surf.addColorStop(0.55,'#c8ccdb');
      surf.addColorStop(1,'#898ea8');
      ctx.beginPath(); ctx.arc(mx,my,mr,0,Math.PI*2);
      ctx.fillStyle=surf; ctx.fill();

      craters.forEach(c=>{
        const cr=c.r*(mr/72);
        const cx2=mx+c.ox*(mr/72), cy2=my+c.oy*(mr/72);
        if(Math.sqrt((cx2-mx)**2+(cy2-my)**2)+cr>mr*0.96) return;
        const cg=ctx.createRadialGradient(cx2,cy2,0,cx2,cy2,cr);
        cg.addColorStop(0,'rgba(70,75,100,0.6)');
        cg.addColorStop(1,'rgba(100,108,130,0)');
        ctx.beginPath(); ctx.arc(cx2,cy2,cr,0,Math.PI*2);
        ctx.fillStyle=cg; ctx.fill();
      });

      ctx.save();
      ctx.beginPath(); ctx.arc(mx,my,mr,0,Math.PI*2); ctx.clip();
      const shx=mx+mr*0.38;
      const sh=ctx.createRadialGradient(shx,my,0,shx,my,mr*1.15);
      sh.addColorStop(0,'rgba(4,6,18,0)');
      sh.addColorStop(0.38,'rgba(4,6,18,0.38)');
      sh.addColorStop(1,'rgba(4,6,18,0.82)');
      ctx.fillStyle=sh; ctx.fillRect(mx-mr,my-mr,mr*2,mr*2);
      ctx.restore();

      if(Math.random()<0.003) spawnShooter();
      shooters=shooters.filter(s=>{
        s.life-=0.025;
        if(s.life<=0) return false;
        const sg=ctx.createLinearGradient(s.x,s.y,s.x+s.len*s.dx,s.y+s.len*s.dy);
        sg.addColorStop(0,'rgba(255,255,255,0)');
        sg.addColorStop(0.5,`rgba(255,255,255,${s.life*0.9})`);
        sg.addColorStop(1,'rgba(255,255,255,0)');
        ctx.strokeStyle=sg; ctx.lineWidth=1.4;
        ctx.beginPath();
        ctx.moveTo(s.x,s.y);
        ctx.lineTo(s.x+s.len*s.dx,s.y+s.len*s.dy);
        ctx.stroke();
        return true;
      });

      requestAnimationFrame(draw);
    }
    draw();
  }

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MISSION_LAUNCH_UTC  = datetime(2026, 4,  1, 22, 35, 0, tzinfo=timezone.utc)
MISSION_END_UTC     = datetime(2026, 4, 10, 18, 30, 0, tzinfo=timezone.utc)
ARTEMIS_III_UTC     = datetime(2027,  9, 15, 14,  0, 0, tzinfo=timezone.utc)  # EST. target
MISSION_TOTAL_HOURS = (MISSION_END_UTC - MISSION_LAUNCH_UTC).total_seconds() / 3600  # ~211.9h

MISSION_MILESTONES = {
    "🚀 Launch — SLS Liftoff":        {"time": datetime(2026, 4, 1, 22, 35,  0, tzinfo=timezone.utc), "detail": "SLS lifts off from LC-39B. Smooth first-attempt launch."},
    "⚡ LAS Jettisoned":              {"time": datetime(2026, 4, 1, 22, 38, 18, tzinfo=timezone.utc), "detail": "T+3:18. Earth limb views from onboard camera at 78 mi downrange."},
    "☀ Solar Array Deployment":       {"time": datetime(2026, 4, 1, 22, 53,  0, tzinfo=timezone.utc), "detail": "All 4 SAWs fully deployed. ~63 ft wingspan. Power generation nominal."},
    "🔩 ICPS Separation":             {"time": datetime(2026, 4, 2,  0,  5,  0, tzinfo=timezone.utc), "detail": "ICPS completes upper-stage burns and separates from Orion."},
    "🕹 Proximity Ops Demo":          {"time": datetime(2026, 4, 2,  0, 35,  0, tzinfo=timezone.utc), "detail": "Glover hand-flies Integrity ~70 min. 'This flies very nicely.'"},
    "🔥 Perigee Raise Burn":          {"time": datetime(2026, 4, 2, 11, 30,  0, tzinfo=timezone.utc), "detail": "43-sec SM engine burn raises perigee. Crew woken to 'Sleepyhead'."},
    "🌙 Translunar Injection (TLI)":  {"time": datetime(2026, 4, 2, 23, 49,  0, tzinfo=timezone.utc), "detail": "5m 50s burn. ΔV +1,274 ft/s. 'We do not leave Earth — we choose it.' — Koch"},
    "🌍 Earth Departure / SOI Exit":  {"time": datetime(2026, 4, 3,  6,  0,  0, tzinfo=timezone.utc), "detail": "Orion exits Earth sphere of influence. Fully on translunar coast."},
    "⚙ Mid-Course Correction 1":      {"time": datetime(2026, 4, 4, 12,  0,  0, tzinfo=timezone.utc), "detail": "Small SM burn to fine-tune free-return trajectory. Flight Day 3."},
    "↔ Halfway to Moon":              {"time": datetime(2026, 4, 4, 23, 49,  0, tzinfo=timezone.utc), "detail": "~192,200 km from Earth. Velocity decreasing due to Earth gravity."},
    "⚙ Mid-Course Correction 2":      {"time": datetime(2026, 4, 5, 18,  0,  0, tzinfo=timezone.utc), "detail": "Second trajectory refinement. Final targeting for lunar flyby."},
    "🌙 Closest Lunar Approach":      {"time": datetime(2026, 4, 6, 10,  0,  0, tzinfo=timezone.utc), "detail": "8,000 km from lunar surface. ~6h observation. Solar eclipse from lunar shadow."},
    "📏 Record Earth Distance":       {"time": datetime(2026, 4, 6, 18,  0,  0, tzinfo=timezone.utc), "detail": "252,021 statute miles — shattering Apollo 13 record by 3,366 miles."},
    "🏠 Return Translunar Coast":     {"time": datetime(2026, 4, 7,  6,  0,  0, tzinfo=timezone.utc), "detail": "Moon gravity slingshot complete. Orion accelerating back toward Earth."},
    "⚙ Mid-Course Correction 3":      {"time": datetime(2026, 4, 8, 12,  0,  0, tzinfo=timezone.utc), "detail": "Return trajectory refinement. Targeting splashdown off San Diego."},
    "🔴 Earth Entry Interface":       {"time": datetime(2026, 4, 10, 16,  0,  0, tzinfo=timezone.utc), "detail": "Orion enters atmosphere at ~400,000 ft. Peak heating ~5,000°F."},
    "🌊 Splashdown — Pacific Ocean":  {"time": datetime(2026, 4, 10, 18, 30,  0, tzinfo=timezone.utc), "detail": "Parachute descent. Recovery by USS San Diego off California coast."},
}

CREW_DATA = [
    {"name":"Reid Wiseman",   "role":"Commander",          "emoji":"👨‍🚀",
     "bio":"NASA astronaut with ISS experience (Expedition 40/41). Led spacewalks and served as Chief of the Astronaut Office.",
     "url":"https://www.nasa.gov/people/reid-wiseman/"},
    {"name":"Victor Glover",  "role":"Pilot",              "emoji":"👨‍✈️",
     "bio":"Naval aviator and NASA astronaut. First African American on ISS long-duration crew (SpaceX Crew-1).",
     "url":"https://www.nasa.gov/humans-in-space/astronauts/victor-j-glover/"},
    {"name":"Christina Koch", "role":"Mission Specialist", "emoji":"👩‍🚀",
     "bio":"Record holder for longest single spaceflight by a woman (328 days). Conducted first all-female spacewalk.",
     "url":"https://www.nasa.gov/humans-in-space/astronauts/christina-koch/"},
    {"name":"Jeremy Hansen",  "role":"Mission Specialist", "emoji":"👨‍🚀",
     "bio":"CSA astronaut and CF-18 fighter pilot. First Canadian to travel beyond low Earth orbit.",
     "url":"https://www.asc-csa.gc.ca/eng/astronauts/canadian/active/bio-jeremy-hansen.asp"},
]

# ═══════════════════════════════════════════════════════════════════════════
# DATA SOURCE META
# ═══════════════════════════════════════════════════════════════════════════

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

SOURCE_META = {
    "AROW":       {"label": "🛰️ NASA AROW",    "color": "#00ff88",
                   "detail": "Live Orion sensor telemetry via NASA Mission Control"},
    "Horizons":   {"label": "🔭 JPL Horizons",  "color": "#00d4ff",
                   "detail": "Ephemeris from NASA/JPL (obj −1024, periodic MCC uplinks)"},
    "simulation": {"label": "⚙️ Physics Model", "color": "#ffd700",
                   "detail": "Keplerian simulation — AROW & Horizons unreachable"},
}

# ═══════════════════════════════════════════════════════════════════════════
# HORIZONS QUERY HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _query_horizons_vectors(command: str, center: str = "'500@399'") -> dict | None:
    try:
        now_utc  = datetime.now(timezone.utc)
        stop_utc = now_utc + timedelta(minutes=2)
        fmt      = "%Y-%m-%d %H:%M"
        params = {
            "format":     "json",
            "COMMAND":    command,
            "OBJ_DATA":   "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER":     center,
            "START_TIME": f"'{now_utc.strftime(fmt)}'",
            "STOP_TIME":  f"'{stop_utc.strftime(fmt)}'",
            "STEP_SIZE":  "'1 m'",
            "VEC_TABLE":  "3",
            "REF_PLANE":  "FRAME",
            "REF_SYSTEM": "J2000",
            "OUT_UNITS":  "KM-S",
            "CSV_FORMAT": "NO",
        }
        resp = requests.get(HORIZONS_URL, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        result_text = resp.json().get("result", "")
        if "$$SOE" not in result_text or "$$EOE" not in result_text:
            return None
        soe   = result_text.index("$$SOE") + 5
        eoe   = result_text.index("$$EOE")
        block = result_text[soe:eoe].strip()
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            return None

        def _extract(line):
            vals = []
            for token in line.split():
                if "=" in token:
                    try:
                        vals.append(float(token.split("=")[1]))
                    except ValueError:
                        pass
            if not vals:
                for token in line.split():
                    try:
                        vals.append(float(token))
                    except ValueError:
                        pass
            return vals

        pos = _extract(lines[1])
        vel = _extract(lines[2])
        if len(pos) < 3 or len(vel) < 3:
            return None
        return {
            "x_km": pos[0], "y_km": pos[1], "z_km": pos[2],
            "vx":   vel[0], "vy":   vel[1], "vz":   vel[2],
        }
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# MOON POSITION
# ═══════════════════════════════════════════════════════════════════════════

def get_moon_position_km() -> tuple[float, float, float]:
    now = datetime.now(timezone.utc)
    cached    = st.session_state.get("moon_pos_cache")
    cache_time= st.session_state.get("moon_pos_cache_time")
    if cached and cache_time and (now - cache_time).total_seconds() < 300:
        return cached

    sv = _query_horizons_vectors("'301'", center="'500@399'")
    if sv:
        pos = (sv["x_km"], sv["y_km"], sv["z_km"])
        st.session_state.moon_pos_cache      = pos
        st.session_state.moon_pos_cache_time = now
        return pos

    jd = (now - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0
    L  = np.radians((218.316 + 13.176396 * jd) % 360)
    M  = np.radians((134.963 + 13.064993 * jd) % 360)
    F  = np.radians((93.272  + 13.229350 * jd) % 360)
    lon  = L + np.radians(6.289 * np.sin(M))
    lat  = np.radians(5.128 * np.sin(F))
    dist = 385001 - 20905 * np.cos(M)
    eps  = np.radians(23.439)
    mx   = dist * np.cos(lat) * np.cos(lon)
    my   = dist * (np.cos(eps) * np.cos(lat) * np.sin(lon) - np.sin(eps) * np.sin(lat))
    mz   = dist * (np.sin(eps) * np.cos(lat) * np.sin(lon) + np.cos(eps) * np.sin(lat))
    pos  = (mx, my, mz)
    st.session_state.moon_pos_cache      = pos
    st.session_state.moon_pos_cache_time = now
    return pos


def compute_dist_moon(craft_x: float, craft_y: float, craft_z: float) -> float:
    mx, my, mz = get_moon_position_km()
    return float(np.sqrt((craft_x - mx)**2 + (craft_y - my)**2 + (craft_z - mz)**2))

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: NASA AROW
# ═══════════════════════════════════════════════════════════════════════════

def fetch_arow_state_vector():
    candidate_urls = [
        "https://nasa.gov/sites/default/files/atoms/files/artemis2_state_vectors.txt",
        "https://www.nasa.gov/sites/default/files/atoms/files/artemis_ii_ephemeris.txt",
        "https://artemis.nasa.gov/artemis-ii/state-vectors/latest.txt",
    ]
    for url in candidate_urls:
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "ArtemisII-Dashboard/1.0"})
            if resp.status_code != 200:
                continue
            text = resp.text.strip()
            data_lines = [l for l in text.splitlines()
                          if l.strip() and not l.strip().startswith("#")]
            if not data_lines:
                continue
            parts = data_lines[-1].split()
            if len(parts) < 7:
                continue
            x_km = float(parts[1]); y_km = float(parts[2]); z_km = float(parts[3])
            vx   = float(parts[4]); vy   = float(parts[5]); vz   = float(parts[6])
            dist_earth = np.sqrt(x_km**2 + y_km**2 + z_km**2)
            velocity   = np.sqrt(vx**2 + vy**2 + vz**2)
            dist_moon  = compute_dist_moon(x_km, y_km, z_km)
            return {
                "distance_earth_km": dist_earth,
                "distance_moon_km":  dist_moon,
                "velocity_km_s":     velocity,
                "x_km": x_km, "y_km": y_km, "z_km": z_km,
                "source": "AROW", "source_url": url,
            }
        except Exception:
            continue
    return None

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: JPL Horizons
# ═══════════════════════════════════════════════════════════════════════════

def fetch_horizons_telemetry():
    sv = _query_horizons_vectors("'-1024'", center="'500@399'")
    if not sv:
        return None
    x_km, y_km, z_km = sv["x_km"], sv["y_km"], sv["z_km"]
    vx, vy, vz       = sv["vx"],   sv["vy"],   sv["vz"]
    dist_earth = float(np.sqrt(x_km**2 + y_km**2 + z_km**2))
    velocity   = float(np.sqrt(vx**2 + vy**2 + vz**2))
    dist_moon  = compute_dist_moon(x_km, y_km, z_km)
    return {
        "distance_earth_km": dist_earth,
        "distance_moon_km":  dist_moon,
        "velocity_km_s":     velocity,
        "x_km": x_km, "y_km": y_km, "z_km": z_km,
        "source": "Horizons", "source_url": HORIZONS_URL,
    }

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: Physics Simulation
# ═══════════════════════════════════════════════════════════════════════════

def _sim_state_at_hours(elapsed_hours):
    h = max(0.0, min(elapsed_hours, MISSION_TOTAL_HOURS))

    if h < 1.5:
        p   = h / 1.5
        alt = p * 300
        vel = 0.5 + p * 7.5
    elif h < 25:
        p   = (h - 1.5) / 23.5
        alt = 300 + p * 69700
        vel = 8.0 - p * 0.5
    elif h < 27:
        p   = (h - 25) / 2
        alt = 70000 + p * 5000
        vel = 7.5 + p * 3.3
    elif h < 108:
        p   = (h - 27) / 81
        alt = 75000 + p * (407700 - 75000)
        vel = 10.8 - p * 9.9
    elif h < 120:
        p   = (h - 108) / 12
        alt = 407700 + p * 2000
        vel = 0.9 + p * 0.8
    elif h < 132:
        p   = (h - 120) / 12
        alt = 409700 - p * 5000
        vel = 1.7 + p * 0.5
    elif h < 192:
        p   = (h - 132) / 60
        alt = 404700 - p * 354700
        vel = 2.2 + p * 5.8
    elif h < 209:
        p   = (h - 192) / 17
        alt = max(0, 50000 - p * 50000)
        vel = 8.0 + p * 3.5
    else:
        alt = 0.0
        vel = 0.0

    craft_x   = float(max(0, alt))
    craft_y   = 0.0
    craft_z   = 0.0
    dist_moon = compute_dist_moon(craft_x, craft_y, craft_z)
    return float(max(0, alt)), float(max(0, vel)), float(max(0, dist_moon)), craft_x, craft_y, craft_z


def calculate_mission_profile_sim():
    now           = datetime.now(timezone.utc)
    elapsed_hours = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600

    if elapsed_hours < 0:
        dm = compute_dist_moon(0.0, 0.0, 0.0)
        return {
            "distance_earth_km": 0, "distance_moon_km": dm,
            "velocity_km_s": 0,
            "x_km": 0, "y_km": 0, "z_km": 0,
            "source": "simulation", "source_url": None,
        }

    alt, vel, dm, cx, cy, cz = _sim_state_at_hours(elapsed_hours)
    return {
        "distance_earth_km": alt,
        "distance_moon_km":  dm,
        "velocity_km_s":     vel,
        "x_km": cx, "y_km": cy, "z_km": cz,
        "source": "simulation", "source_url": None,
    }

# ═══════════════════════════════════════════════════════════════════════════
# FULL MISSION PROFILE (for charts)
# ═══════════════════════════════════════════════════════════════════════════

def get_full_mission_profile():
    now_h = (datetime.now(timezone.utc) - MISSION_LAUNCH_UTC).total_seconds() / 3600
    end_h = min(now_h, MISSION_TOTAL_HOURS)
    if end_h <= 0:
        return pd.DataFrame(columns=["met_hours", "altitude_km", "velocity_km_s"])
    n_pts = max(3, int(end_h) + 1)
    hours = np.linspace(0, end_h, n_pts)
    rows  = []
    for h in hours:
        alt, vel, _, _, _, _ = _sim_state_at_hours(h)
        rows.append({"met_hours": h, "altitude_km": alt, "velocity_km_s": vel})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════
# LIVE TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════

def get_live_telemetry():
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history       = []
        st.session_state.last_telemetry_update   = None
        st.session_state.data_source             = "simulation"
        st.session_state.horizons_consecutive_ok = 0
        st.session_state.arow_available          = False

    now        = datetime.now(timezone.utc)
    should_add = (
        st.session_state.last_telemetry_update is None or
        (now - st.session_state.last_telemetry_update).total_seconds() >= 5
    )

    if should_add:
        is_post_mission = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600 >= MISSION_TOTAL_HOURS

        # Post-mission: return static Earth surface values, skip API calls
        if is_post_mission:
            mx, my, mz = get_moon_position_km()
            dm_now     = float(np.sqrt(mx**2 + my**2 + mz**2))
            current = {
                "distance_earth_km": 0.0,
                "distance_moon_km":  dm_now,
                "velocity_km_s":     0.0,
                "x_km": 0.0, "y_km": 0.0, "z_km": 0.0,
                "source": "simulation", "source_url": None,
            }
            st.session_state.data_source = "simulation"
        else:
            current = fetch_arow_state_vector()
            if current:
                st.session_state.data_source    = "AROW"
                st.session_state.arow_available = True

            if not current:
                current = fetch_horizons_telemetry()
                if current:
                    st.session_state.data_source             = "Horizons"
                    st.session_state.horizons_consecutive_ok += 1
                else:
                    st.session_state.horizons_consecutive_ok = 0

            if not current:
                current = calculate_mission_profile_sim()
                current["distance_earth_km"] += np.random.normal(0, 10)
                current["velocity_km_s"]     += np.random.normal(0, 0.04)
                current["distance_earth_km"]  = max(0, current["distance_earth_km"])
                current["velocity_km_s"]      = max(0, current["velocity_km_s"])
                st.session_state.data_source  = "simulation"

        st.session_state.current_telemetry = current
        elapsed_hours = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600

        st.session_state.telemetry_history.append({
            "timestamp":          now,
            "altitude_km":        current["distance_earth_km"],
            "velocity_km_s":      current["velocity_km_s"],
            "distance_earth_km":  current["distance_earth_km"],
            "distance_moon_km":   current["distance_moon_km"],
            "met_hours":          elapsed_hours,
            "source":             current["source"],
        })
        st.session_state.last_telemetry_update = now

    if len(st.session_state.telemetry_history) > 120:
        st.session_state.telemetry_history = st.session_state.telemetry_history[-120:]

    return pd.DataFrame(st.session_state.telemetry_history)

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE STATUS
# ═══════════════════════════════════════════════════════════════════════════

def get_milestone_status():
    now = datetime.now(timezone.utc)
    out = {}
    for name, data in MISSION_MILESTONES.items():
        ms_time      = data["time"]
        diff_seconds = (now - ms_time).total_seconds()
        out[name] = {
            "completed":              diff_seconds >= 0,
            "time_utc":               ms_time,
            "time_remaining_minutes": max(0, -diff_seconds / 60),
            "detail":                 data.get("detail", ""),
        }
    return out

# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN MODEL
# ═══════════════════════════════════════════════════════════════════════════

class BayesianMissionSuccess:
    def __init__(self, alpha=92, beta=5):
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

# ═══════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(df):
    if len(df) < 10:
        return np.zeros(len(df)), np.ones(len(df))
    features = df[["distance_earth_km", "velocity_km_s"]].values
    iso      = IsolationForest(contamination=0.05, random_state=42)
    svm      = OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
    i_pred   = iso.fit_predict(features)
    s_pred   = svm.fit_predict(features)
    i_score  = iso.score_samples(features)
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
    n  = 200
    ts = np.linspace(0, 100, n)
    traj = []
    for t in ts:
        if v1 >= v_esc:
            d = r0 + v1 * t * 3600
        else:
            energy = v1**2 / 2 - GM / r0
            a   = -GM / (2 * energy)
            mm  = np.sqrt(GM / a**3)
            ang = mm * t * 3600
            d   = a * (1 + 0.5 * np.cos(ang))
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
# MET / COUNTDOWN TIMER  (sidebar + hero)
# ═══════════════════════════════════════════════════════════════════════════

def display_live_met_timer(compact=False):
    """
    compact=False → full sidebar version (two stacked boxes, height 185)
    compact=True  → hero version (slightly taller for the centered hero area)
    """
    mission_duration = MISSION_END_UTC - MISSION_LAUNCH_UTC
    total_s  = int(mission_duration.total_seconds())
    dur_d    = total_s // 86400
    dur_h    = (total_s % 86400) // 3600
    dur_m    = (total_s % 3600)  // 60
    dur_s    = total_s % 60

    ts_a3_ms = int(ARTEMIS_III_UTC.timestamp() * 1000)
    height   = 185 if not compact else 200

    html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');
body{{margin:0;padding:0;background:transparent}}
.box{{font-family:'Space Mono',monospace;text-align:center;padding:.65rem .9rem;
      border-radius:10px;backdrop-filter:blur(8px);margin-bottom:.5rem}}
.complete-box{{background:rgba(0,255,136,0.08);border:1.5px solid rgba(0,255,136,0.45)}}
.complete-lbl{{font-size:.6rem;color:rgba(0,255,136,0.6);letter-spacing:.15em;margin-bottom:.25rem}}
.complete-val{{font-size:{'1.2rem' if compact else '1.1rem'};color:#00ff88;
               text-shadow:0 0 12px rgba(0,255,136,0.6);font-weight:700}}
.a3-box{{background:rgba(0,212,255,0.07);border:1.5px solid rgba(0,212,255,0.4)}}
.a3-lbl{{font-size:.6rem;color:rgba(0,212,255,0.6);letter-spacing:.15em;margin-bottom:.25rem}}
.a3-val{{font-size:{'1.1rem' if compact else '1.05rem'};color:#00d4ff;
          text-shadow:0 0 12px rgba(0,212,255,0.6);font-weight:700}}
.a3-sub{{font-size:.55rem;color:rgba(0,212,255,0.4);margin-top:.2rem;letter-spacing:.1em}}
</style>
<div class="box complete-box">
  <div class="complete-lbl">✅ ARTEMIS II — MISSION COMPLETE</div>
  <div class="complete-val">{dur_d:02d}d {dur_h:02d}h {dur_m:02d}m {dur_s:02d}s</div>
</div>
<div class="box a3-box">
  <div class="a3-lbl">🚀 ARTEMIS III — NEXT LAUNCH ETA</div>
  <div class="a3-val" id="a3cd_{compact}">CALCULATING…</div>
  <div class="a3-sub">EST. SEP 2027 · TARGET TBD BY NASA</div>
</div>
<script>
(function(){{
  var t3={ts_a3_ms};
  var el=document.getElementById('a3cd_{compact}');
  function tick(){{
    if(!el) return;
    var diff=Math.max(0,t3-Date.now());
    var d=Math.floor(diff/864e5),
        h=Math.floor(diff%864e5/36e5),
        m=Math.floor(diff%36e5/6e4),
        s=Math.floor(diff%6e4/1e3);
    el.textContent=
      String(d).padStart(3,'0')+'d '+
      String(h).padStart(2,'0')+'h '+
      String(m).padStart(2,'0')+'m '+
      String(s).padStart(2,'0')+'s';
  }}
  tick(); setInterval(tick,1000);
}})();
</script>
"""
    components.html(html, height=height)

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
        ORION "INTEGRITY" · SLS BLOCK 1 · FREE-RETURN TRAJECTORY
      </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HERO — Mission Success Probability
# ═══════════════════════════════════════════════════════════════════════════

@st.fragment(run_every="5s")
def display_mission_success():
    ms        = get_milestone_status()
    completed = sum(1 for v in ms.values() if v["completed"])
    total     = len(ms)
    model     = BayesianMissionSuccess()
    penalty   = 0

    if "telemetry_history" in st.session_state and len(st.session_state.telemetry_history) >= 10:
        df_tmp = pd.DataFrame(st.session_state.telemetry_history[-20:])
        if len(df_tmp) >= 10:
            _, preds = detect_anomalies(df_tmp)
            penalty  = int((preds == -1).mean() * 3)

    model.update(completed, penalty)

    now_h            = (datetime.now(timezone.utc) - MISSION_LAUNCH_UTC).total_seconds() / 3600
    mission_complete = now_h >= MISSION_TOTAL_HOURS

    # Force 100% when mission is done — prior beta never zeroes out otherwise
    if mission_complete:
        prob, lo, hi = 1.0, 1.0, 1.0
    else:
        prob   = model.predict()
        lo, hi = model.credible_interval()

    src         = st.session_state.get("data_source", "simulation")
    sm          = SOURCE_META.get(src, SOURCE_META["simulation"])
    penalty_txt = f" · Anomaly Penalty: {penalty}" if (penalty and not mission_complete) else ""

    ci_line = (
        "All milestones confirmed · Splashdown complete"
        if mission_complete
        else f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]"
    )

    complete_badge = (
        '<div style="color:#00ff88;font-family:Space Mono,monospace;font-size:.9rem;margin-top:.5rem;">'
        '🌊 MISSION COMPLETE — SPLASHDOWN CONFIRMED · USS SAN DIEGO</div>'
    ) if mission_complete else ""

    st.markdown(f"""
    <div class="hero-metric">
      <div class="hero-value">{prob*100:.1f}%</div>
      <div class="hero-label">Mission Success Probability</div>
      <div class="hero-sub">
        Bayesian Beta-Binomial · {ci_line}<br>
        <span style="color:#00d4ff;">Milestones: {completed}/{total} confirmed complete</span>{penalty_txt}<br>
        <span style="color:{sm['color']};">{sm['label']} · {sm['detail']}</span>
      </div>
      {complete_badge}
    </div>
    """, unsafe_allow_html=True)

display_mission_success()

# ═══════════════════════════════════════════════════════════════════════════
# HERO — MET / ARTEMIS III COUNTDOWN
# ═══════════════════════════════════════════════════════════════════════════

col_m1, col_m2, col_m3 = st.columns([1, 3, 1])
with col_m2:
    display_live_met_timer(compact=True)

# ═══════════════════════════════════════════════════════════════════════════
# THANK YOU BANNER + NASA LINKS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="thankyou-wrap">
  <div style="font-size:2.8rem;margin-bottom:0.6rem;">🌙🚀🌍</div>

  <h2 style="
    font-family:'Rajdhani',sans-serif;
    font-size:clamp(1.6rem,4vw,2.4rem);
    font-weight:700;
    letter-spacing:0.12em;
    background:linear-gradient(90deg,#00d4ff 0%,#ffffff 50%,#ffd700 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin:0 0 0.6rem;
    text-transform:uppercase;
  ">Thank You for Following Artemis II</h2>

  <p style="
    font-family:'Space Mono',monospace;
    font-size:0.82rem;
    color:rgba(255,255,255,0.55);
    line-height:1.8;
    max-width:620px;
    margin:0 auto 1.6rem;
    letter-spacing:0.04em;
  ">
    Humanity's first crewed lunar voyage since Apollo 17 — four astronauts, 252,021 miles from Earth,
    and a flawless free-return trajectory. Artemis II proved we are ready to return to the Moon. 🌕
  </p>

  <div style="display:flex;justify-content:center;gap:1.2rem;flex-wrap:wrap;">
    <a href="https://www.nasa.gov/artemis-ii-multimedia/"
       target="_blank"
       class="nasa-btn nasa-btn-cyan">
      📸 Artemis II Multimedia Gallery
    </a>
    <a href="https://www.nasa.gov/mission/artemis-iii/"
       target="_blank"
       class="nasa-btn nasa-btn-gold">
      🌕 What's Next — Artemis III Mission
    </a>
  </div>

  <p style="
    margin-top:1.4rem;
    font-family:'Space Mono',monospace;
    font-size:0.65rem;
    color:rgba(255,255,255,0.22);
    letter-spacing:0.12em;
    text-transform:uppercase;
  ">GODSPEED ARTEMIS II · For All Humanity</p>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("### 🏁 MISSION STATUS")
    display_live_met_timer(compact=False)

    @st.fragment(run_every="5s")
    def sidebar_source_badge():
        src      = st.session_state.get("data_source", "simulation")
        sm       = SOURCE_META.get(src, SOURCE_META["simulation"])
        arow_ok  = st.session_state.get("arow_available", False)
        hz_ok    = st.session_state.get("horizons_consecutive_ok", 0) > 0
        moon_cached = st.session_state.get("moon_pos_cache") is not None
        def dot(ok): return f"<span style='color:{'#00ff88' if ok else '#ff4444'};'>●</span>"
        st.markdown(f"""
        <div class="glass-card" style="margin-top:.5rem;">
          <div style="font-size:.68rem;color:rgba(255,255,255,0.6);font-family:'Space Mono',monospace;line-height:2;">
            <b style="color:{sm['color']};">Active: {sm['label']}</b><br>
            {dot(arow_ok)} Tier 1 · NASA AROW<br>
            {dot(hz_ok)}   Tier 2 · JPL Horizons (−1024)<br>
            {dot(not arow_ok and not hz_ok)} Tier 3 · Physics Model<br>
            {dot(moon_cached)} Moon pos · JPL/301 (5 min cache)<br>
            <hr style="border-color:rgba(255,255,255,0.1);margin:.4rem 0;">
            🔄 MET: 1 s · Telemetry: 5 s<br>
            🔄 Milestones: 5 s · Prediction: 5 s
          </div>
        </div>
        """, unsafe_allow_html=True)
    sidebar_source_badge()

    @st.fragment(run_every="5s")
    def sidebar_phase():
        now = datetime.now(timezone.utc)
        h   = (now - MISSION_LAUNCH_UTC).total_seconds() / 3600
        if h < 0:                         phase, col = "⏳ Pre-Launch",            "#888888"
        elif h < 1.5:                     phase, col = "🚀 Ascent / MECO",         "#ff6b35"
        elif h < 25:                      phase, col = "🌍 High Earth Orbit",       "#00d4ff"
        elif h < 27:                      phase, col = "🔥 TLI Burn",               "#ff4444"
        elif h < 108:                     phase, col = "🚀 Translunar Coast",       "#ffd700"
        elif h < 120:                     phase, col = "🌙 Lunar Flyby Phase",      "#ffffff"
        elif h < 132:                     phase, col = "📏 Record Distance",        "#ff88ff"
        elif h < 192:                     phase, col = "🏠 Earth Return Coast",     "#00ff88"
        elif h < 209:                     phase, col = "🔴 Earth Entry / Descent",  "#ff4444"
        elif h < MISSION_TOTAL_HOURS:     phase, col = "🌊 Splashdown",             "#00d4ff"
        else:                             phase, col = "✅ Mission Complete",        "#00ff88"
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
            completed = data["completed"]
            rem_min   = data["time_remaining_minutes"]
            if completed:
                icon, color, rem = "✅", "#00ff88", ""
            else:
                icon, color = "⏳", "#ffd700"
                rem = ""
                if rem_min < 180:
                    hh  = int(rem_min // 60)
                    mm  = int(rem_min % 60)
                    rem = f" ({hh}h {mm}m)"
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin:.4rem 0;font-family:'Space Mono',monospace;">
              <span style="font-size:1rem;margin-right:.5rem;flex-shrink:0;">{icon}</span>
              <div>
                <span style="color:{color};font-size:.75rem;display:block;">{name}{rem}</span>
                <span style="color:rgba(255,255,255,0.28);font-size:.62rem;">{data['time_utc'].strftime('%b %d %H:%M')} UTC</span>
              </div>
            </div>""", unsafe_allow_html=True)
    sidebar_milestones()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📡 Live Mission Feed", "🧪 Strategic Simulation"])

with tab1:
    st.markdown("### 📊 Real-Time Telemetry Analysis")

    @st.fragment(run_every="5s")
    def live_telemetry_display():
        tdf     = get_live_telemetry()
        src     = st.session_state.get("data_source", "simulation")
        sm      = SOURCE_META.get(src, SOURCE_META["simulation"])
        utc_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        now_h   = (datetime.now(timezone.utc) - MISSION_LAUNCH_UTC).total_seconds() / 3600
        mission_ended = now_h >= MISSION_TOTAL_HOURS

        tier_desc = {
            "AROW":       "Tier 1 · Live Orion sensor data via NASA Mission Control / AROW",
            "Horizons":   "Tier 2 · JPL Horizons ephemeris (obj −1024 · periodic MCC uplinks)",
            "simulation": "Tier 3 · Physics simulation — mission complete · Earth surface",
        }.get(src, "Tier 3 · Physics simulation")

        status_color = "#00ff88" if mission_ended else sm["color"]
        status_label = "✅ MISSION COMPLETE" if mission_ended else sm["label"]
        status_desc  = "Orion recovered · USS San Diego · Pacific Ocean" if mission_ended else tier_desc

        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3);border:1px solid {status_color};border-radius:10px;
                    padding:.7rem 1.2rem;margin-bottom:1rem;backdrop-filter:blur(8px);
                    display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
          <span style="color:{status_color};font-family:'Space Mono',monospace;font-size:.85rem;font-weight:700;">
            {status_label}
          </span>
          <span style="color:rgba(255,255,255,0.55);font-family:'Space Mono',monospace;font-size:.75rem;">
            {status_desc}
          </span>
          <span style="margin-left:auto;color:rgba(255,255,255,0.4);font-family:'Space Mono',monospace;font-size:.72rem;">
            Last: {utc_str} · Refresh: 5 s
          </span>
        </div>
        """, unsafe_allow_html=True)

        if len(tdf) < 2:
            st.warning("Collecting telemetry…")
            return

        scores, preds = detect_anomalies(tdf)
        tdf = tdf.copy()
        tdf["anomaly_score"] = scores
        tdf["is_anomaly"]    = preds == -1

        # ── 4 metrics ────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        if mission_ended:
            with c1:
                st.metric("Velocity", "0.000 km/s", "Recovered · Earth Surface")
            with c2:
                st.metric("Distance from Earth", "0 km", "Splashdown · Pacific Ocean")
            with c3:
                moon_cached = st.session_state.get("moon_pos_cache") is not None
                moon_src    = "JPL/301" if moon_cached else "Analytical"
                mx, my, mz  = get_moon_position_km()
                dm_now      = float(np.sqrt(mx**2 + my**2 + mz**2))
                st.metric(f"Moon Distance ({moon_src})", f"{dm_now:,.0f} km", "Current lunar position")
            with c4:
                st.metric("Status", "✅ COMPLETE", "Mission Success · All 17/17")
        else:
            cur = st.session_state.get("current_telemetry", {})
            with c1:
                vel   = tdf.iloc[-1]["velocity_km_s"]
                d_vel = vel - tdf.iloc[-2]["velocity_km_s"]
                st.metric("Velocity", f"{vel:.3f} km/s", f"{d_vel:+.4f} km/s")
            with c2:
                de      = cur.get("distance_earth_km", tdf.iloc[-1].get("distance_earth_km", 0))
                de_prev = tdf.iloc[-2].get("distance_earth_km", de) if len(tdf) >= 2 else de
                d_de    = de - de_prev
                st.metric("Distance from Earth", f"{de:,.0f} km", f"{d_de:+.0f} km")
            with c3:
                dm      = cur.get("distance_moon_km", tdf.iloc[-1].get("distance_moon_km", 384400))
                dm_prev = tdf.iloc[-2].get("distance_moon_km", dm) if len(tdf) >= 2 else dm
                d_dm    = dm - dm_prev
                moon_cached = st.session_state.get("moon_pos_cache") is not None
                moon_src    = "JPL/301" if moon_cached else "Analytical"
                st.metric(f"Distance to Moon ({moon_src})", f"{dm:,.0f} km", f"{d_dm:+.0f} km")
            with c4:
                a_cnt  = int(tdf["is_anomaly"].sum())
                avg_s  = tdf["anomaly_score"].mean()
                status = "NOMINAL" if avg_s < 0.3 else "⚠️ CAUTION"
                st.metric("Status", status, f"{a_cnt} anomalies")

        # ── Trajectory chart ──────────────────────────────────────────────
        st.markdown("#### 🛰️ Trajectory Profile")
        profile_df = get_full_mission_profile()

        fig = go.Figure()
        if len(profile_df) >= 2:
            fig.add_trace(go.Scatter(
                x=profile_df["met_hours"], y=profile_df["altitude_km"],
                mode="lines", name="Mission Profile",
                line=dict(color="rgba(0,212,255,0.22)", width=1.5, dash="dot"),
                hoverinfo="skip",
            ))

        if not mission_ended:
            fig.add_trace(go.Scatter(
                x=tdf["met_hours"], y=tdf["distance_earth_km"],
                mode="lines+markers", name="Live Telemetry",
                line=dict(color="#00d4ff", width=2.5),
                marker=dict(size=4, color="#00d4ff"),
                hovertemplate="<b>MET:</b> %{x:.2f}h<br><b>Dist Earth:</b> %{y:,.0f} km<extra></extra>"
            ))
            anom = tdf[tdf["is_anomaly"]]
            if len(anom):
                fig.add_trace(go.Scatter(
                    x=anom["met_hours"], y=anom["distance_earth_km"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#ff3a3a", size=10, symbol="x"),
                    hovertemplate="<b>⚠️ ANOMALY</b><br>MET: %{x:.2f}h<extra></extra>"
                ))

        ms_status = get_milestone_status()
        for ms_name, ms_data in ms_status.items():
            ms_h = (ms_data["time_utc"] - MISSION_LAUNCH_UTC).total_seconds() / 3600
            if 0 <= ms_h <= MISSION_TOTAL_HOURS:
                fig.add_vline(x=ms_h, line=dict(color="rgba(255,215,0,0.3)", width=1, dash="dot"))

        splashdown_h = MISSION_TOTAL_HOURS
        fig.add_vline(x=splashdown_h,
                      line=dict(color="rgba(0,255,136,0.7)", width=2),
                      annotation_text="SPLASHDOWN ✅",
                      annotation_font_color="#00ff88",
                      annotation_position="top left")

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff", family="Rajdhani"),
            xaxis=dict(title="MET (hours)", gridcolor="rgba(255,255,255,0.08)",
                       range=[0, MISSION_TOTAL_HOURS]),
            yaxis=dict(title="Distance from Earth (km)", gridcolor="rgba(255,255,255,0.08)"),
            hovermode="x unified", height=440,
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(0,212,255,0.3)")
        )
        st.plotly_chart(fig, use_container_width=True, key="traj_main")

        # ── Velocity chart ────────────────────────────────────────────────
        st.markdown("#### ⚡ Velocity Profile")
        fig_v = go.Figure()
        if len(profile_df) >= 2:
            fig_v.add_trace(go.Scatter(
                x=profile_df["met_hours"], y=profile_df["velocity_km_s"],
                mode="lines", name="Mission Profile",
                line=dict(color="rgba(255,215,0,0.2)", width=1.5, dash="dot"),
                hoverinfo="skip",
            ))

        if not mission_ended:
            fig_v.add_trace(go.Scatter(
                x=tdf["met_hours"], y=tdf["velocity_km_s"],
                mode="lines+markers", name="Live Velocity",
                line=dict(color="#ffd700", width=2.5),
                marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(255,215,0,0.07)"
            ))

        fig_v.add_vline(x=splashdown_h,
                        line=dict(color="rgba(0,255,136,0.7)", width=2),
                        annotation_text="SPLASHDOWN ✅",
                        annotation_font_color="#00ff88",
                        annotation_position="top left")

        fig_v.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff", family="Rajdhani"),
            xaxis=dict(title="MET (hours)", gridcolor="rgba(255,255,255,0.08)",
                       range=[0, MISSION_TOTAL_HOURS]),
            yaxis=dict(title="Velocity (km/s)", gridcolor="rgba(255,255,255,0.08)"),
            height=320
        )
        st.plotly_chart(fig_v, use_container_width=True, key="vel_main")

        # ── Anomaly section ───────────────────────────────────────────────
        if not mission_ended:
            st.markdown("#### 🔍 ML Anomaly Detection")
            st.markdown("*Ensemble: Isolation Forest + One-Class SVM*")
            a1, a2 = st.columns(2)

            with a1:
                fig_sc = go.Figure()
                norm = tdf[~tdf["is_anomaly"]]
                fig_sc.add_trace(go.Scatter(
                    x=norm["velocity_km_s"], y=norm["distance_earth_km"],
                    mode="markers", name="Nominal",
                    marker=dict(color=norm["anomaly_score"], colorscale="Plasma",
                                size=7, colorbar=dict(title="Score", thickness=10))
                ))
                anom = tdf[tdf["is_anomaly"]]
                if len(anom):
                    fig_sc.add_trace(go.Scatter(
                        x=anom["velocity_km_s"], y=anom["distance_earth_km"],
                        mode="markers", name="Anomaly",
                        marker=dict(color="#ff3a3a", size=11, symbol="x")
                    ))
                fig_sc.update_layout(
                    title="Feature Space", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#fff", family="Rajdhani"),
                    xaxis=dict(title="Velocity (km/s)", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="Distance from Earth (km)", gridcolor="rgba(255,255,255,0.08)"),
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
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 70}
                    }
                ))
                fig_g.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#fff", "family": "Rajdhani"}, height=400
                )
                st.plotly_chart(fig_g, use_container_width=True, key="gauge_anom")
        else:
            # Post-mission anomaly summary
            st.markdown("#### 🔍 ML Anomaly Detection — Mission Summary")
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.35);
                        border-radius:12px;padding:1.2rem 1.6rem;backdrop-filter:blur(8px);margin:.8rem 0;">
              <div style="color:#00ff88;font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;margin-bottom:.6rem;">
                ✅ ENSEMBLE ML — MISSION NOMINAL
              </div>
              <div style="color:rgba(255,255,255,0.6);font-family:'Space Mono',monospace;font-size:.75rem;line-height:1.8;">
                Isolation Forest + One-Class SVM monitored all 211.9 mission hours.<br>
                No critical anomalies detected across trajectory or velocity profiles.<br>
                Final anomaly rate: &lt;5% contamination threshold — well within bounds.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Mission events log ────────────────────────────────────────────
        st.markdown("#### 📋 Mission Events Log")
        ms_all = get_milestone_status()
        for ms_name, ms_data in ms_all.items():
            done    = ms_data["completed"]
            utc_str = ms_data["time_utc"].strftime("%Y-%m-%d %H:%M UTC")
            icon    = "✅" if done else "⏳"
            color   = "#00ff88" if done else "#ffd700"
            status  = "COMPLETE" if done else "UPCOMING"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-left:3px solid {color};border-radius:8px;padding:.6rem .9rem;
                        margin:.4rem 0;backdrop-filter:blur(8px);">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:{color};font-weight:700;font-size:.88rem;">{icon} {ms_name}</span>
                <span style="font-family:'Space Mono',monospace;font-size:.65rem;padding:.2rem .5rem;
                             background:rgba(255,255,255,0.06);border-radius:20px;color:{color};">{status}</span>
              </div>
              <div style="font-family:'Space Mono',monospace;font-size:.67rem;color:rgba(255,255,255,0.38);margin-top:.25rem;">{utc_str}</div>
              <div style="font-size:.75rem;color:rgba(255,255,255,0.55);margin-top:.2rem;">{ms_data['detail']}</div>
            </div>""", unsafe_allow_html=True)

    live_telemetry_display()


with tab2:
    st.markdown("### 🚀 TLI Burn Simulator")
    st.markdown("*Keplerian orbital mechanics for strategic planning*")
    st.markdown("---")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        thrust_pct    = st.slider("Engine Thrust (%)",    80, 105, 100, 1,  help="Nominal: 100%")
    with sc2:
        burn_duration = st.slider("Burn Duration (s)",   250, 450, 350, 10, help="Nominal: 350 s")
    with sc3:
        ignition_vec  = st.slider("Ignition Vector (°)",   0, 360,  90,  5, help="Thrust direction")

    if st.button("🔥 EXECUTE TLI BURN SIMULATION", type="primary"):
        with st.spinner("Propagating trajectory…"):
            outcome, ok, traj_df = simulate_tli_burn(thrust_pct, burn_duration, ignition_vec)
            st.session_state.sim_outcome    = outcome
            st.session_state.sim_success    = ok
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
        fig_s.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", name="Earth",
            marker=dict(size=28, color="#00d4ff"), text=["🌍"], textfont=dict(size=36)))
        mx = 384400 * np.cos(np.radians(45))
        my = 384400 * np.sin(np.radians(45))
        fig_s.add_trace(go.Scatter(x=[mx], y=[my], mode="markers+text", name="Moon",
            marker=dict(size=18, color="#ffd700"), text=["🌙"], textfont=dict(size=28)))
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
        dv = 3.1 * (thrust_pct / 100) * (burn_duration / 350)
        with ba1: st.metric("ΔV",             f"{dv:.3f} km/s")
        with ba2: st.metric("Final Distance", f"{st.session_state.sim_trajectory.iloc[-1]['distance_km']:,.0f} km")
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
    <span class="tech-badge">🛰️ NASA AROW (Tier 1)</span>
    <span class="tech-badge">🔭 JPL Horizons −1024 (Tier 2)</span>
    <span class="tech-badge">🌙 JPL Horizons 301/Moon (Live)</span>
    <span class="tech-badge">⚙️ Physics Sim (Tier 3)</span>
  </div>
  <div style="margin-top:1.2rem;display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;">
    <a href="https://www.nasa.gov/artemis-ii-multimedia/" target="_blank"
       style="color:rgba(0,212,255,0.6);font-size:.72rem;text-decoration:none;font-family:'Space Mono',monospace;">
      📸 Artemis II Multimedia
    </a>
    <a href="https://www.nasa.gov/mission/artemis-iii/" target="_blank"
       style="color:rgba(255,215,0,0.6);font-size:.72rem;text-decoration:none;font-family:'Space Mono',monospace;">
      🌕 Artemis III Mission
    </a>
  </div>
  <p style="margin-top:1.5rem;font-size:.72rem;color:rgba(255,255,255,0.35);">
    Developed by Deep Rushil · GODSPEED ARTEMIS II · April 2026
  </p>
</div>
""", unsafe_allow_html=True)
