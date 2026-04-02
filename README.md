# Artemis II: Real-Time Mission Intelligence Dashboard

A professional-grade Data Science portfolio project showcasing advanced ML/AI techniques through a real-time mission monitoring dashboard for NASA's Artemis II lunar mission.

## 🎯 Project Overview

This Streamlit application demonstrates:
- **Real-Time Mission Tracking**: Integration with NASA AROW and third-party telemetry APIs
- **Bayesian Inference**: Beta-Binomial conjugate priors for mission success prediction
- **Unsupervised Machine Learning**: Voting ensemble (Isolation Forest + One-Class SVM) for anomaly detection
- **Orbital Mechanics**: Physics-based simulation of Translunar Injection burns
- **Live Data Visualization**: Interactive Plotly charts with real-time telemetry updates
- **Production-Grade UI/UX**: Custom CSS with deep space theme and NASA-inspired design

## 🛰️ Real-Time Data Integration

### Data Sources (Priority Order)
1. **artemislivetracker.com API** - Community tracker using NASA telemetry
2. **artemislive.org API** - Alternative real-time tracking service
3. **Mission Profile Calculations** - Physics-based fallback using known mission timeline

### Available Telemetry Data
- Distance from Earth (km)
- Distance to Moon (km)
- Spacecraft velocity (km/s)
- Orbital altitude (km)
- Mission Elapsed Time (MET)

The application automatically attempts to fetch live data from available APIs. If APIs are unavailable, it falls back to calculated trajectories based on:
- Known Artemis II mission profile
- Orbital mechanics (Keplerian elements, vis-viva equation)
- Actual launch time (April 1, 2026, 22:35 UTC)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run artemis_ii_mission_intelligence.py
   ```

3. **Access the dashboard:**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

## 📊 Features Breakdown

### 1. Live Mission Success Prediction
- **Methodology**: Bayesian Beta-Binomial model
- **Prior**: α=95, β=5 (reflecting SLS/Orion heritage reliability)
- **Updates**: Real-time as mission milestones are confirmed
- **Output**: Success probability with 95% credible interval

### 2. Real-Time Telemetry Analysis
- **Data Source**: Attempts live APIs first, falls back to calculated trajectory
- **Visualization**: Interactive altitude/velocity trajectory plots
- **Metrics**: Altitude, velocity, Earth distance, Moon distance, flight status

### Enhancing Real-Time Data
To connect to additional NASA data sources:

1. **NASA API Key**: Get free API key at https://api.nasa.gov
2. **Add to Environment**: Create `.env` file with `NASA_API_KEY=your_key_here`
3. **Official AROW**: Monitor https://www.nasa.gov/trackartemis for state vector downloads
4. **State Vectors**: Download ephemeris files from AROW and parse for trajectory data

Example NASA API integration:
```python
import os
NASA_API_KEY = os.getenv('NASA_API_KEY', 'DEMO_KEY')

# Fetch data from NASA endpoints
response = requests.get(
    f"https://api.nasa.gov/planetary/some-endpoint?api_key={NASA_API_KEY}"
)
```

### 3. Anomaly Detection System
- **Algorithms**: 
  - Isolation Forest (contamination=0.05)
  - One-Class SVM (nu=0.05, RBF kernel)
- **Ensemble Method**: Voting (anomaly flagged when both algorithms agree)
- **Features**: Altitude (km) and Velocity (km/s)
- **Output**: Anomaly scores, binary predictions, visual highlighting

### 4. TLI Burn Simulator
- **Physics Model**: Keplerian orbital mechanics
- **Parameters**: 
  - Engine thrust: 80-105%
  - Burn duration: 250-450 seconds
  - Ignition vector: 0-360 degrees
- **Outcomes**: 
  - ✅ Successful Lunar Flyby
  - ❌ Failed Capture (insufficient velocity)
  - ❌ Deep Space Drift (excessive velocity)
  - ⚠️ Lunar Impact (trajectory intercept)

## 🎨 Design Philosophy

The interface follows a "Mission Control meets Deep Space" aesthetic:
- **Color Palette**: Deep space black (#0b0d17), neon cyan (#00d4ff), solar gold (#ffd700)
- **Typography**: Rajdhani (headers), Space Mono (monospace data)
- **Visual Effects**: Glowing elements, pulsing animations, high-contrast data displays
- **Inspiration**: NASA mission control interfaces, retro-futuristic design

## 📁 Project Structure

```
artemis_ii_mission_intelligence.py  # Main application
requirements.txt                     # Python dependencies
README.md                           # This file
```

## 🔧 Technical Implementation

### Bayesian Model
```python
class BayesianMissionSuccess:
    - Prior: Beta(α=95, β=5)
    - Update: Posterior = Beta(α + successes, β + failures)
    - Prediction: E[p] = α / (α + β)
```

### Anomaly Detection Pipeline
```python
1. Extract features: [altitude_km, velocity_km_s]
2. Isolation Forest → predictions, scores
3. One-Class SVM → predictions, scores
4. Voting ensemble → final anomaly flags
5. Normalize scores to [0, 1] range
```

### Orbital Propagator
```python
1. Calculate post-burn velocity: v = v0 + Δv
2. Compare to escape velocity: v_escape = √(2GM/r)
3. Propagate trajectory using Keplerian elements
4. Evaluate outcome based on final distance vs Moon position
```

## 📈 Data Science Techniques Demonstrated

| Technique | Application | Implementation |
|-----------|-------------|----------------|
| Bayesian Inference | Mission success prediction | Beta-Binomial conjugate priors |
| Isolation Forest | Outlier detection | Scikit-learn ensemble |
| One-Class SVM | Anomaly detection | RBF kernel, nu=0.05 |
| Ensemble Methods | Voting classifier | Binary consensus |
| Orbital Mechanics | Trajectory simulation | Keplerian elements |
| Real-Time Updates | Live MET counter | Datetime operations |

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- Advanced statistical modeling (Bayesian methods)
- Unsupervised machine learning (anomaly detection)
- Physics-based simulation (orbital mechanics)
- Data visualization (Plotly interactive charts)
- Production-grade software development (modular code, documentation)
- UI/UX design (custom CSS, theming)



## 📧 Contact

**Deep Rushil**
- [LIVE URL](https://artemis2.streamlit.app/)
-  [LinkedIn](https://www.linkedin.com/in/deeprushil/)

---

*Developed as part of Personal project, Please educate if there is any error, I am still learning. Thank you*
*April 2026, Bengaluru*
*GODSPEED ARTEMIS II* 

