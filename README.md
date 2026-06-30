# 🌬️ VayuDrishti AI v2.0
### Urban Air Quality Intelligence Platform
**ET AI Hackathon 2026 | PS5: AI-Powered Urban Air Quality Intelligence for Smart City Intervention**

---

## 🏆 What This Solves
India's air quality crisis kills **1.67 million people annually** (Lancet, 2024). Despite 900+ CPCB monitoring stations, only 31% of cities have actionable response protocols. VayuDrishti AI bridges this gap — from reactive dashboards to **proactive, evidence-based intervention**.

---

## 🤖 Architecture: 6-Agent Multi-AI System

```
CPCB AQI API ──────────┐
OpenWeatherMap ─────────┼──► Data Engine ──► Source Attribution Agent
Simulated CAAQMS ───────┘         │               ├── AQI Forecast Agent (24/48/72h + CI)
                                  │               ├── Health Advisory Agent (4 languages)
                                  │               ├── Enforcement Agent (routing + briefs)
                                  │               ├── Smart City Agent (interventions + what-if)
                                  │               └── Multilingual Agent (SMS/IVR/Push)
                                  └──► Executive Dashboard → City Admin
```

| Agent | Role | PS5 Alignment |
|---|---|---|
| 📡 AQI Forecast | Weather-aware GB, 24/48/72h + CI | Hyperlocal forecasting |
| 🔍 Source Attribution | PM ratio, NO₂, SO₂, zone, wind | Source attribution engine |
| 🏥 Health Advisory | 4 languages, per vulnerable group | Citizen health system |
| 🚨 Enforcement | Priority ranking, TSP routing | Enforcement intelligence |
| 🏙️ Smart City | Interventions + what-if simulation | Smart city planning |
| 🌐 Multilingual | SMS/IVR/Push/WhatsApp per channel | Multi-channel comms |

---

## ✅ PS5 Feature Coverage (100%)

| PS5 Requirement | Implementation |
|---|---|
| Geospatial Pollution Source Attribution | ✅ Multi-signal confidence scoring + source map |
| Hyperlocal AQI Forecasting 24-72h | ✅ Gradient Boosting + weather features + CI |
| Enforcement Intelligence | ✅ Priority scores + TSP routing + AI briefs |
| Multi-City Comparative Dashboard | ✅ 12-city benchmarking + trend analysis |
| Citizen Health Advisory (multilingual) | ✅ Telugu / Tamil / Hindi / English |
| Weather-aware atmospheric modelling | ✅ OpenWeatherMap integration |
| CAAQMS IoT sensor integration | ✅ CPCB API + realistic fallback |
| Digital Twin + What-if simulation | ✅ 7 scenarios with gauge visualization |
| Multi-Agent AI Architecture | ✅ 6 independent specialized agents |
| Ward-level heatmaps + risk scoring | ✅ Folium heatmap + risk score per station |
| Inspection route optimization | ✅ Nearest-neighbor TSP routing |
| Multi-channel advisory dispatch | ✅ SMS / IVR / Push / WhatsApp / Display |

---

## 🗂️ Folder Structure

```
vayu_drishti_ai/
├── app.py                     # Main Streamlit app (9 tabs)
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   ├── __init__.py
│   └── constants.py           # Cities, AQI scale, sources, scenarios
├── agents/
│   ├── __init__.py
│   └── agent_system.py        # All 6 AI agents
├── utils/
│   ├── __init__.py
│   ├── data_engine.py         # CPCB, OWM, synthetic data
│   └── map_builder.py         # Folium map constructors
└── data/                      # Local cache (auto-created)
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/vayu-drishti-ai.git
cd vayu-drishti-ai

# 2. Install
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your keys

# 4. Run
streamlit run app.py
```

---

## 🔑 API Keys (All Free Tier)

| Key | Where to Get | Required? |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | For AI alerts/explanations |
| `OWM_API_KEY` | [openweathermap.org](https://openweathermap.org/api) | For live weather |
| `CPCB_API_KEY` | [data.gov.in](https://data.gov.in) | For live AQI (realistic fallback if absent) |

> **All features work without API keys** — the app uses realistic simulated data seeded from actual CPCB 2024 patterns.

---

## ☁️ Deploy to Streamlit Cloud (Free)

```bash
# 1. Push to GitHub
git add . && git commit -m "VayuDrishti AI v2.0" && git push

# 2. Go to share.streamlit.io
# 3. Connect repo → select app.py
# 4. Add secrets:
#    GROQ_API_KEY = "your_key"
#    OWM_API_KEY  = "your_key"
# 5. Deploy → get live URL in 60 seconds
```

---

## 📊 Judging Criteria Coverage

| Criteria | Weight | Features Targeting It |
|---|---|---|
| **Innovation** | 25% | Multi-agent, Digital Twin, What-if, Multilingual |
| **Business Impact** | 25% | Source attribution, Enforcement routing, City interventions, Lives saved metric |
| **Technical Excellence** | 20% | GB forecast + CI, TSP routing, Attribution engine, CPCB + OWM integration |
| **Scalability** | 15% | 12-city support, Agent modularity, Streamlit Cloud deploy |
| **User Experience** | 15% | 9-tab app, KPI cards, Executive dashboard, Mobile-friendly |

---


---

## 🛠️ Tech Stack

- **Frontend:** Streamlit 1.35 + custom CSS
- **Maps:** Folium + streamlit-folium (station, heatmap, source, route maps)
- **ML:** scikit-learn GradientBoostingRegressor + weather features + confidence intervals
- **AI:** Groq Llama 3.3 70B (6 specialized agent prompts)
- **Data:** CPCB data.gov.in API + OpenWeatherMap API + realistic CPCB-pattern fallback
- **Charts:** Plotly Express + Graph Objects
- **Deploy:** Streamlit Community Cloud (free)

---


ET AI Hackathon 2026 — PS5: Urban Air Quality Intelligence
