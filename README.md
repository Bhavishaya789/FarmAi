# 🌾 FarmAI — Smart Crop & Disease Prediction System

FarmAI is an AI-powered agricultural assistant that helps farmers make data-driven decisions. It recommends the best crops based on soil and climate conditions, detects plant diseases from leaf images, and suggests fertilizers — all through a clean web interface.

---

## Features

- Crop Recommendation — predicts the best crop based on N, P, K, temperature, humidity, pH, and rainfall
- Disease Detection — analyzes plant leaf images and suggests treatment
- Fertilizer Suggestion — recommends fertilizers based on soil nutrient levels
- User Authentication — register, login, change password, and manage profile
- History Tracking — saves crop and disease prediction history per user
- PWA Support — installable as an Android app via the included manifest

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| ML Model | Random Forest (scikit-learn) |
| Database | SQLite via SQLAlchemy |
| Auth | Passlib + bcrypt |
| Frontend | HTML, CSS, JavaScript |
| Server | Uvicorn (ASGI) |

---

## Project Structure

```
FarmAi/
├── crop.py                  # Main FastAPI backend
├── crop_model.joblib        # Trained ML model
├── Crop_recommendation/     # Training dataset (CSV)
├── logo/                    # App images and backgrounds
├── static/uploads/          # User profile pictures
├── android_app/             # PWA version for Android
├── android_native/          # Native Android wrapper
├── *.html                   # Frontend pages
├── requirements.txt         # Python dependencies
└── start_farmai.bat         # One-click Windows launcher
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Bhavishaya789/FarmAi.git
cd FarmAi

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn crop:app --reload --host 127.0.0.1 --port 8000
```

Then open your browser at: **http://127.0.0.1:8000**

### Windows Quick Start
Just double-click `start_farmai.bat` — it handles everything automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Crop recommendation |
| POST | `/predict-disease` | Plant disease detection |
| POST | `/predict-fertilizer` | Fertilizer suggestion |
| POST | `/register` | User registration |
| POST | `/login` | User login |
| POST | `/change-password` | Change password |
| GET | `/get-user-history/{username}` | Fetch prediction history |

---

## ML Model

- Algorithm: Random Forest Classifier
- Dataset: [Crop Recommendation Dataset](Crop_recommendation/Crop_recommendation.csv)
- Features: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Output: Top-N crop suggestions with confidence scores

---

## Team

| Name | Roll Number |
|------|-------------|
| Bhavishya Kumar | 0251BTCS042 |
| Atharv Pandey | 0251BTCS048 |
| Dipanshu | 0251BTCS140 |
| Aditya Singh | 0251BTCS081 |

---

## License

This project is built for academic purposes.
