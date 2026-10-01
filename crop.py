# FarmAI Backend - crop.py
# Bhavishaya kumar - 0251BTCS042
# Atharv pandey  - 0251BTCS048
# Dipanshu       - 0251BTCS140
# Aditya singh   - 0251BTCS081

import os, shutil, random, string
from dotenv import load_dotenv

# ✅ FIX 1: Call load_dotenv() BEFORE reading any env vars
load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests as http_requests
from typing import Optional, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt
from disease_model import predict_disease_from_image, predict_disease_multiple, _safe_float
from intelligent_assistant import generate_ai_explanation, estimate_cost, generate_voice_base64
from gradcam import generate_gradcam_overlay
from pymongo import MongoClient

# ----------------------------------------
# CONFIG
# ----------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/farmai")
JWT_SECRET  = os.getenv("JWT_SECRET", "changeme_secret_please_change")
JWT_ALGO    = "HS256"
JWT_EXPIRE  = 60 * 24  # minutes
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

# ----------------------------------------
# FASTAPI APP
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Safe static file mounts
try:
    _static_dir = os.path.join(BASE_DIR, "static")
    if os.path.isdir(_static_dir):
        app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    _logo_dir = os.path.join(BASE_DIR, "logo")
    if os.path.isdir(_logo_dir):
        app.mount("/logo", StaticFiles(directory=_logo_dir), name="logo")
except Exception:
    pass

# ----------------------------------------
# MONGODB
# ----------------------------------------
# ✅ FIX 3: Wrap MongoDB in try/except — avoids 10-second hang if URI isn't set at build time
try:
    mongo_client     = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db_mongo         = mongo_client["farmai"]
    users_col        = db_mongo["users"]
    crop_hist_col    = db_mongo["crop_history"]
    disease_hist_col = db_mongo["disease_history"]
    otp_col          = db_mongo["otp_store"]
except Exception as _mongo_err:
    print(f"[WARN] MongoDB connection failed at startup: {_mongo_err}")
    mongo_client = None
    db_mongo = None
    users_col = None
    crop_hist_col = None
    disease_hist_col = None
    otp_col = None

# ----------------------------------------
# AUTH HELPERS
# ----------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p): return pwd_context.hash(p)
def verify_password(p, h): return pwd_context.verify(p, h)

def create_token(username: str):
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE)
    return jwt.encode({"sub": username, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGO)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token or token expired")

def generate_otp():
    return "".join(random.choices(string.digits, k=6))

# ----------------------------------------
# ML MODEL
# ----------------------------------------
def train_and_persist_model():
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "Crop_recommendation", "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X, y = df[feature_cols], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_split=5, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    m.fit(X, y)
    model_save_path = os.path.join(BASE_DIR, "crop_model.joblib")
    joblib.dump({"model": m, "features": feature_cols, "accuracy": acc}, model_save_path)
    return m, feature_cols, acc

def load_model():
    import os
    import joblib

    model_path = os.path.join(BASE_DIR, "crop_model.joblib")

    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            if "accuracy" in data:
                return data["model"], data["features"], data["accuracy"]
        except Exception as _e:
            print("Model load warning:", _e)

    try:
        return train_and_persist_model()
    except Exception as _e:
        print("Training fallback failed:", _e)
        return None, ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], 0.0

model, feature_names, model_accuracy = load_model()

# ----------------------------------------
# CROP ECONOMICS DATA (India-specific averages per acre)
# ----------------------------------------
CROP_ECONOMICS = {
    "rice": {"yield": "2.5-3.5 tons", "profit": "₹35,000-₹50,000"},
    "maize": {"yield": "3.5-4.5 tons", "profit": "₹45,000-₹65,000"},
    "wheat": {"yield": "2.0-2.8 tons", "profit": "₹30,000-₹45,000"},
    "cotton": {"yield": "0.8-1.5 tons", "profit": "₹40,000-₹60,000"},
    "sugarcane": {"yield": "35-50 tons", "profit": "₹60,000-₹90,000"},
    "pomegranate": {"yield": "5-8 tons", "profit": "₹1,20,000-₹2,50,000"},
    "banana": {"yield": "15-25 tons", "profit": "₹1,00,000-₹2,00,000"},
    "mango": {"yield": "3-6 tons", "profit": "₹80,000-₹1,50,000"},
    "grapes": {"yield": "10-15 tons", "profit": "₹2,00,000-₹4,00,000"},
    "apple": {"yield": "6-10 tons", "profit": "₹1,50,000-₹3,00,000"},
    "orange": {"yield": "8-12 tons", "profit": "₹90,000-₹1,80,000"},
    "papaya": {"yield": "20-30 tons", "profit": "₹1,50,000-₹3,00,000"},
    "coconut": {"yield": "4,000-7,000 nuts", "profit": "₹80,000-₹1,40,000"},
    "coffee": {"yield": "0.5-1.2 tons", "profit": "₹70,000-₹1,30,000"},
    "tea": {"yield": "0.8-1.5 tons", "profit": "₹60,000-₹1,10,000"},
    "potato": {"yield": "12-18 tons", "profit": "₹70,000-₹1,20,000"},
    "onion": {"yield": "10-15 tons", "profit": "₹60,000-₹1,30,000"},
    "tomato": {"yield": "15-25 tons", "profit": "₹80,000-₹2,00,000"},
    "jute": {"yield": "1.0-1.5 tons", "profit": "₹25,000-₹40,000"},
    "chickpea": {"yield": "0.6-1.0 tons", "profit": "₹20,000-₹35,000"},
    "lentil": {"yield": "0.4-0.8 tons", "profit": "₹18,000-₹30,000"},
    "default": {"yield": "Varied (approx 2-5 tons)", "profit": "₹30,000-₹60,000"}
}

# Ensure uploads directory exists (use /tmp on serverless environments)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception:
    UPLOAD_DIR = "/tmp/uploads"
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception:
        pass

# ----------------------------------------
# PYDANTIC MODELS
# ----------------------------------------
class UserLogin(BaseModel):
    username: str
    password: str

class PasswordChange(BaseModel):
    username: str
    current_password: str
    new_password: str

class OTPVerify(BaseModel):
    username: str
    otp: str

class PredictRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_type: str = "Not specified"
    top_n: int = 5

class FertilizerRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    crop: str

class CropHistorySave(BaseModel):
    username: str
    crop_name: str
    confidence: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class DiseaseHistorySave(BaseModel):
    username: str
    disease_name: str
    confidence: float
    treatment: str

class UserUpdate(BaseModel):
    address: Optional[str] = None

# ----------------------------------------
# AUTH ROUTES  (plain def — FastAPI runs these in threadpool automatically)
# ----------------------------------------

@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    address: Optional[str] = Form(None),
    profile_picture: Optional[UploadFile] = File(None)
):
    if users_col.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if users_col.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    picture_path = None
    if profile_picture and profile_picture.filename:
        ext = profile_picture.filename.split(".")[-1]
        try:
            target_path = os.path.join(UPLOAD_DIR, fname)
            with open(target_path, "wb") as f:
                shutil.copyfileobj(profile_picture.file, f)
            picture_path = f"uploads/{fname}"
        except Exception:
            picture_path = None

    otp = generate_otp()
    otp_col.delete_many({"username": username})
    otp_col.insert_one({
        "username": username,
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=10)
    })

    users_col.insert_one({
        "username": username,
        "email": email,
        "hashed_password": hash_password(password),
        "address": address,
        "profile_picture": picture_path,
        "verified": False,
        "created_at": datetime.utcnow()
    })

    return {"status": "registered", "otp": otp, "email": email, "username": username}


@app.post("/verify-otp")
def verify_otp(req: OTPVerify):
    record = otp_col.find_one({"username": req.username})
    if not record:
        raise HTTPException(status_code=400, detail="No OTP found. Please register again.")
    if record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    if datetime.utcnow() > record["expires"]:
        raise HTTPException(status_code=400, detail="OTP expired. Please register again.")

    users_col.update_one({"username": req.username}, {"$set": {"verified": True}})
    otp_col.delete_many({"username": req.username})

    user = users_col.find_one({"username": req.username})
    token = create_token(req.username)
    return {
        "message": "Email verified",
        "access_token": token,
        "user": {
            "username": user["username"],
            "email": user["email"],
            "address": user.get("address"),
            "profile_picture": user.get("profile_picture")
        }
    }


@app.post("/resend-otp")
def resend_otp(data: dict):
    username = data.get("username")
    user = users_col.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    otp = generate_otp()
    otp_col.delete_many({"username": username})
    otp_col.insert_one({
        "username": username,
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=10)
    })
    return {"otp": otp, "email": user["email"], "username": username}


@app.post("/login")
def login(user: UserLogin):
    db_user = users_col.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not db_user.get("verified", False):
        raise HTTPException(status_code=403, detail="Email not verified. Please verify your account.")
    token = create_token(user.username)
    return {
        "message": "Login successful",
        "access_token": token,
        "user": {
            "username": db_user["username"],
            "email": db_user.get("email"),
            "address": db_user.get("address"),
            "profile_picture": db_user.get("profile_picture")
        }
    }


@app.post("/change-password")
def change_password(req: PasswordChange):
    db_user = users_col.find_one({"username": req.username})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, db_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    users_col.update_one(
        {"username": req.username},
        {"$set": {"hashed_password": hash_password(req.new_password)}}
    )
    return {"message": "Password updated successfully"}


@app.put("/users/{username}")
def update_profile(username: str, req: UserUpdate):
    users_col.update_one({"username": username}, {"$set": {"address": req.address}})
    user = users_col.find_one({"username": username})
    return {
        "username": user["username"],
        "address": user.get("address"),
        "profile_picture": user.get("profile_picture")
    }


@app.post("/users/{username}/profile-picture")
def update_profile_picture(username: str, file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    fname = f"{username}_profile.{ext}"
    os.makedirs("static/uploads", exist_ok=True)
    with open(f"static/uploads/{fname}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    users_col.update_one({"username": username}, {"$set": {"profile_picture": f"uploads/{fname}"}})
    return {"profile_picture": f"uploads/{fname}"}

# ----------------------------------------
# HISTORY ROUTES
# ----------------------------------------

@app.post("/save-crop-history")
def save_crop_history(req: CropHistorySave):
    crop_hist_col.insert_one({
        "username": req.username, "crop_name": req.crop_name,
        "confidence": req.confidence, "temperature": req.temperature,
        "humidity": req.humidity, "ph": req.ph, "rainfall": req.rainfall,
        "timestamp": datetime.utcnow()
    })
    return {"status": "success"}


@app.post("/save-disease-history")
def save_disease_history(req: DiseaseHistorySave):
    disease_hist_col.insert_one({
        "username": req.username, "disease_name": req.disease_name,
        "confidence": req.confidence, "treatment": req.treatment,
        "timestamp": datetime.utcnow()
    })
    return {"status": "success"}


@app.get("/get-user-history/{username}")
def get_user_history(username: str):
    crops = list(crop_hist_col.find({"username": username}, {"_id": 0}).sort("timestamp", -1).limit(50))
    diseases = list(disease_hist_col.find({"username": username}, {"_id": 0}).sort("timestamp", -1).limit(50))
    return {"crops": crops, "diseases": diseases}

# ----------------------------------------
# ML ROUTES
# ----------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
def safe_generate_summary(crop, prob, req):
    try:
        exp = generate_ai_explanation(
            crop,
            float(prob),
            "moderate",
            "en"
        )
        return exp.get("summary", "No explanation available")
    except Exception as e:
        print("SUMMARY ERROR:", e)
        return "Explanation unavailable"



@app.post("/predict")
def predict(req: PredictRequest, current_user: str = Depends(get_current_user)):
    warnings = []
    if req.temperature > 60:
        warnings.append("Extreme heat detected.")
    if req.ph < 0 or req.ph > 14:
        warnings.append("pH must be 0-14.")
    row = {
        "N": req.nitrogen, "P": req.phosphorus, "K": req.potassium,
        "temperature": req.temperature, "humidity": req.humidity,
        "ph": req.ph, "rainfall": req.rainfall
    }
    X = pd.DataFrame([row])
    X = X[feature_names]
    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
    except Exception:
        pred = model.predict(X)
        classes = pred
        probs = [1.0]
    idx = np.argsort(probs)[::-1][:req.top_n]
    
    # ----------------------------------------------------
    # EXPLAINABILITY (LIGHTWEIGHT FEATURE IMPORTANCE)
    # ----------------------------------------------------
    explanation = []
    try:
        # Instead of heavy SHAP, use model's internal feature_importances_
        # This is instant and uses NO extra memory or libraries
        try:
            importances = model.feature_importances_
        except AttributeError:
            importances = [0.1] * len(feature_names)
        for i, fname in enumerate(feature_names):
            # Scale importance by the input relative to standard to give 'reasoning'
            val = float(importances[i])
            explanation.append({
                "feature": fname, 
                "contribution": val
            })
        explanation = sorted(explanation, key=lambda x: x["contribution"], reverse=True)
    except Exception:
        explanation = [{"field": "System", "contribution": 0.1, "note": "Balanced Reasoning"}]

    return {
        "status": "success",
        "metadata": {
            "accuracy": model_accuracy, 
            "soil_type": req.soil_type, 
            "warnings": warnings or None,
            "explainability": {
                "top_crop_explained": classes[idx[0]],
                "feature_contributions": explanation
            }
        },
        "suggestions": [
    {
        "crop": classes[i],
        "probability": float(probs[i]),
        "economics": CROP_ECONOMICS.get(classes[i].lower(), CROP_ECONOMICS["default"]),
        "summary": safe_generate_summary(classes[i], probs[i], req)
    } for i in idx
]
    }


@app.post("/predict-fertilizer")
def predict_fertilizer(req: FertilizerRequest, current_user: str = Depends(get_current_user)):
    ideal = {"N": 40, "P": 40, "K": 40}
    n_diff = ideal["N"] - req.nitrogen
    p_diff = ideal["P"] - req.phosphorus
    k_diff = ideal["K"] - req.potassium
    if n_diff > 10:
        rec = "Urea or Ammonium Nitrate to boost Nitrogen."
    elif p_diff > 10:
        rec = "DAP or Single Superphosphate for Phosphorus."
    elif k_diff > 10:
        rec = "MOP or Potassium Sulfate to increase Potassium."
    else:
        rec = "Soil is optimal. Use balanced 10-10-10 compost."
    return {"crop": req.crop, "recommendation": rec, "status": "Verified"}


@app.post("/predict-disease")
async def predict_disease(
    files: List[UploadFile] = File(...),
    crop: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    current_user: str = Depends(get_current_user)
):
    try:
        image_list = []
        for file in files:
            contents = await file.read()
            image_list.append(contents)
        
        if len(image_list) == 1:
            result = predict_disease_from_image(image_list[0], crop=crop or "", lat=lat, lng=lng)
        else:
            result = predict_disease_multiple(image_list, crop=crop or "", lat=lat, lng=lng)

        if "error" in result:
            return result
        return {
            "disease":    result.get("disease", "Unknown"),
            "confidence": _safe_float(result.get("confidence"), 0.0),
            "severity":   result.get("severity", "Medium"),
            "treatment":  result.get("treatment", "No treatment available."),
            "fertilizer": result.get("fertilizer", "No fertilizer recommendation."),
            "safety":     result.get("safety", ""),
            "cost_estimate": result.get("cost_estimate", ""),
            "location_name": result.get("location_name", ""),
            "reason":     result.get("reason", "Diagnosis based on AI visual analysis."),
            "method":     result.get("method", ""),
            "wrong_inputs": result.get("wrong_inputs", 0),
            "total_inputs": result.get("total_inputs", len(image_list)),
            "gradcam_url": generate_gradcam_overlay(image_list[0]) if image_list else ""
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "disease": "System Overload / Error",
            "confidence": 0.01,
            "severity": "Unknown",
            "treatment": "Connection interrupted. Please check network and retry.",
            "fertilizer": "N/A",
            "safety": "Check equipment safety.",
            "cost_estimate": "N/A",
            "reason": f"Backend Exception: {str(e)[:50]}",
            "method": "System Error"
        }

# ----------------------------------------
# CHATBOT ROUTE (Grok API)
# ----------------------------------------

class ChatMessage(BaseModel):
    message: str
    history: list = []

SYSTEM_PROMPT = """You are FarmAI Assistant, an expert agricultural advisor built into the FarmAI platform.

CRITICAL FACTS — NEVER get these wrong:
- FarmAI's crop recommendation model supports EXACTLY 101 crops (not 3, not 5, not any other number)
- The model recommends the TOP 5 best crops from these 101 based on soil and climate inputs
- The complete list of 101 supported crops is:
  Almond, Apple, Apricot, Artichoke, Asparagus, Banana, Barley, Beetroot, Bell Pepper, Black Pepper, Blackberry, Blackgram, Blueberry, Broccoli, Buckwheat, Cabbage, Canola, Cardamom, Carrot, Cashew, Cassava, Cauliflower, Celery, Cherry, Chickpea, Chili, Cinnamon, Clove, Cocoa, Coconut, Coffee, Cotton, Cranberry, Cucumber, Dates, Eggplant, Fava Beans, Fig, Flax, Garlic, Grapes, Green Beans, Guava, Hazelnut, Jute, Kale, Kidneybeans, Kiwi, Lentil, Lettuce, Lima Beans, Macadamia, Maize, Mango, Millet, Mothbeans, Mungbean, Muskmelon, Mustard, Nectarine, Nutmeg, Oats, Okra, Olive, Onion, Orange, Papaya, Peach, Peanuts, Pear, Peas, Pecan, Pigeonpeas, Pineapple, Pistachio, Plum, Pomegranate, Potato, Pumpkin, Quinoa, Radish, Raspberry, Rice, Rubber, Safflower, Sesame, Sorghum, Soybeans, Spinach, Squash, Strawberry, Sugarcane, Sunflower, Sweet Potato, Tea, Turnip, Walnut, Watermelon, Wheat, Yam, Zucchini

If anyone asks "how many crops does FarmAI support/recommend/suggest" — answer: 101 crops.
If anyone asks to list the crops — list all 101 names above.
If anyone asks what the model predicts — it picks the best 5 from 101 crops.

IDEAL CONDITIONS FOR ALL CROPS:
  - Almond: N:100, P:51, K:71, Temp:22.5C, Hum:49%, pH:6.7, Rain:64mm
  - Apple: N:21, P:134, K:200, Temp:22.6C, Hum:92%, pH:5.9, Rain:113mm
  - Apricot: N:100, P:50, K:70, Temp:22.3C, Hum:61%, pH:6.5, Rain:64mm
  - Artichoke: N:81, P:39, K:49, Temp:19.6C, Hum:70%, pH:6.5, Rain:65mm
  - Asparagus: N:80, P:39, K:50, Temp:20.0C, Hum:70%, pH:7.0, Rain:64mm
  - Banana: N:100, P:82, K:50, Temp:27.4C, Hum:80%, pH:6.0, Rain:105mm
  - Barley: N:65, P:46, K:41, Temp:17.0C, Hum:59%, pH:6.8, Rain:59mm
  - Beetroot: N:59, P:40, K:80, Temp:19.8C, Hum:69%, pH:6.8, Rain:66mm
  - Bell Pepper: N:101, P:51, K:50, Temp:25.2C, Hum:69%, pH:6.5, Rain:103mm
  - Black Pepper: N:78, P:40, K:49, Temp:30.5C, Hum:80%, pH:6.0, Rain:247mm
  - Blackberry: N:79, P:40, K:50, Temp:19.7C, Hum:70%, pH:6.0, Rain:101mm
  - Blackgram: N:40, P:67, K:19, Temp:30.0C, Hum:65%, pH:7.1, Rain:68mm
  - Blueberry: N:59, P:30, K:31, Temp:19.9C, Hum:70%, pH:5.0, Rain:98mm
  - Broccoli: N:101, P:50, K:70, Temp:19.9C, Hum:70%, pH:6.8, Rain:101mm
  - Buckwheat: N:45, P:30, K:30, Temp:20.0C, Hum:51%, pH:6.0, Rain:60mm
  - Cabbage: N:101, P:50, K:70, Temp:20.2C, Hum:70%, pH:6.7, Rain:99mm
  - Canola: N:98, P:50, K:51, Temp:15.0C, Hum:60%, pH:6.5, Rain:60mm
  - Cardamom: N:82, P:39, K:51, Temp:25.1C, Hum:79%, pH:6.0, Rain:206mm
  - Carrot: N:59, P:50, K:50, Temp:19.6C, Hum:70%, pH:6.2, Rain:66mm
  - Cashew: N:79, P:40, K:50, Temp:29.9C, Hum:70%, pH:6.1, Rain:126mm
  - Cassava: N:59, P:40, K:80, Temp:29.8C, Hum:65%, pH:6.0, Rain:156mm
  - Cauliflower: N:100, P:50, K:70, Temp:19.7C, Hum:70%, pH:6.8, Rain:100mm
  - Celery: N:101, P:49, K:69, Temp:19.9C, Hum:80%, pH:6.5, Rain:99mm
  - Cherry: N:103, P:50, K:70, Temp:20.4C, Hum:60%, pH:6.5, Rain:80mm
  - Chickpea: N:40, P:68, K:80, Temp:18.9C, Hum:17%, pH:7.3, Rain:80mm
  - Chili: N:82, P:50, K:50, Temp:24.6C, Hum:71%, pH:6.5, Rain:79mm
  - Cinnamon: N:81, P:40, K:49, Temp:30.5C, Hum:79%, pH:5.8, Rain:206mm
  - Clove: N:79, P:40, K:51, Temp:29.9C, Hum:81%, pH:6.0, Rain:204mm
  - Cocoa: N:99, P:50, K:69, Temp:24.5C, Hum:79%, pH:5.8, Rain:195mm
  - Coconut: N:22, P:17, K:31, Temp:27.4C, Hum:95%, pH:6.0, Rain:176mm
  - Coffee: N:101, P:29, K:30, Temp:25.5C, Hum:59%, pH:6.8, Rain:158mm
  - Cotton: N:118, P:46, K:20, Temp:24.0C, Hum:80%, pH:6.9, Rain:80mm
  - Cranberry: N:62, P:30, K:29, Temp:20.3C, Hum:80%, pH:4.7, Rain:127mm
  - Cucumber: N:99, P:50, K:70, Temp:24.9C, Hum:70%, pH:6.5, Rain:101mm
  - Dates: N:79, P:41, K:50, Temp:37.8C, Hum:29%, pH:7.2, Rain:19mm
  - Eggplant: N:99, P:50, K:50, Temp:24.8C, Hum:70%, pH:6.0, Rain:101mm
  - Fava Beans: N:35, P:50, K:50, Temp:20.1C, Hum:60%, pH:6.8, Rain:67mm
  - Fig: N:81, P:39, K:50, Temp:27.3C, Hum:50%, pH:6.7, Rain:66mm
  - Flax: N:59, P:40, K:41, Temp:19.7C, Hum:59%, pH:6.5, Rain:61mm
  - Garlic: N:78, P:51, K:50, Temp:19.9C, Hum:70%, pH:6.8, Rain:64mm
  - Grapes: N:23, P:133, K:200, Temp:23.8C, Hum:82%, pH:6.0, Rain:70mm
  - Green Beans: N:33, P:50, K:50, Temp:25.6C, Hum:60%, pH:6.8, Rain:66mm
  - Guava: N:101, P:50, K:69, Temp:25.1C, Hum:70%, pH:6.2, Rain:124mm
  - Hazelnut: N:81, P:41, K:51, Temp:19.6C, Hum:59%, pH:6.8, Rain:99mm
  - Jute: N:78, P:47, K:40, Temp:25.0C, Hum:80%, pH:6.7, Rain:175mm
  - Kale: N:79, P:30, K:50, Temp:15.0C, Hum:70%, pH:6.7, Rain:65mm
  - Kidneybeans: N:21, P:68, K:20, Temp:20.1C, Hum:22%, pH:5.7, Rain:106mm
  - Kiwi: N:101, P:50, K:70, Temp:20.0C, Hum:71%, pH:6.5, Rain:125mm
  - Lentil: N:19, P:68, K:19, Temp:24.5C, Hum:65%, pH:6.9, Rain:46mm
  - Lettuce: N:80, P:29, K:49, Temp:17.6C, Hum:71%, pH:6.4, Rain:65mm
  - Lima Beans: N:35, P:50, K:50, Temp:24.6C, Hum:61%, pH:6.7, Rain:64mm
  - Macadamia: N:80, P:39, K:50, Temp:25.0C, Hum:70%, pH:6.2, Rain:126mm
  - Maize: N:78, P:48, K:20, Temp:22.4C, Hum:65%, pH:6.2, Rain:85mm
  - Mango: N:20, P:27, K:30, Temp:31.2C, Hum:50%, pH:5.8, Rain:95mm
  - Millet: N:59, P:30, K:30, Temp:29.8C, Hum:40%, pH:6.5, Rain:45mm
  - Mothbeans: N:21, P:48, K:20, Temp:28.2C, Hum:53%, pH:6.8, Rain:51mm
  - Mungbean: N:21, P:47, K:20, Temp:28.5C, Hum:85%, pH:6.7, Rain:48mm
  - Muskmelon: N:100, P:18, K:50, Temp:28.7C, Hum:92%, pH:6.4, Rain:25mm
  - Mustard: N:59, P:40, K:40, Temp:20.6C, Hum:60%, pH:6.8, Rain:59mm
  - Nectarine: N:98, P:50, K:71, Temp:22.3C, Hum:60%, pH:6.5, Rain:80mm
  - Nutmeg: N:81, P:40, K:50, Temp:29.7C, Hum:79%, pH:6.0, Rain:201mm
  - Oats: N:57, P:39, K:40, Temp:15.0C, Hum:59%, pH:6.2, Rain:70mm
  - Okra: N:80, P:50, K:50, Temp:30.0C, Hum:70%, pH:6.7, Rain:81mm
  - Olive: N:80, P:51, K:50, Temp:22.8C, Hum:49%, pH:7.6, Rain:61mm
  - Onion: N:99, P:51, K:51, Temp:20.0C, Hum:71%, pH:6.4, Rain:66mm
  - Orange: N:20, P:17, K:10, Temp:22.8C, Hum:92%, pH:7.0, Rain:110mm
  - Papaya: N:50, P:59, K:50, Temp:33.7C, Hum:92%, pH:6.7, Rain:143mm
  - Peach: N:101, P:50, K:70, Temp:23.2C, Hum:59%, pH:6.5, Rain:79mm
  - Peanuts: N:36, P:46, K:44, Temp:29.7C, Hum:50%, pH:6.3, Rain:74mm
  - Pear: N:101, P:50, K:70, Temp:20.1C, Hum:60%, pH:6.5, Rain:80mm
  - Peas: N:34, P:51, K:50, Temp:19.9C, Hum:61%, pH:6.7, Rain:66mm
  - Pecan: N:99, P:50, K:70, Temp:24.7C, Hum:60%, pH:6.7, Rain:98mm
  - Pigeonpeas: N:21, P:68, K:20, Temp:27.7C, Hum:48%, pH:5.8, Rain:149mm
  - Pineapple: N:100, P:51, K:70, Temp:24.5C, Hum:71%, pH:6.0, Rain:125mm
  - Pistachio: N:76, P:39, K:49, Temp:28.2C, Hum:40%, pH:7.7, Rain:44mm
  - Plum: N:101, P:50, K:70, Temp:22.6C, Hum:60%, pH:6.5, Rain:79mm
  - Pomegranate: N:19, P:19, K:40, Temp:21.8C, Hum:90%, pH:6.4, Rain:108mm
  - Potato: N:101, P:50, K:50, Temp:17.4C, Hum:85%, pH:6.0, Rain:88mm
  - Pumpkin: N:79, P:50, K:70, Temp:25.3C, Hum:71%, pH:6.7, Rain:100mm
  - Quinoa: N:82, P:51, K:50, Temp:19.8C, Hum:50%, pH:7.4, Rain:41mm
  - Radish: N:57, P:41, K:49, Temp:17.5C, Hum:70%, pH:6.5, Rain:65mm
  - Raspberry: N:81, P:39, K:49, Temp:20.2C, Hum:70%, pH:6.0, Rain:99mm
  - Rice: N:80, P:48, K:40, Temp:23.7C, Hum:82%, pH:6.4, Rain:236mm
  - Rubber: N:81, P:40, K:50, Temp:30.0C, Hum:80%, pH:5.2, Rain:250mm
  - Safflower: N:59, P:41, K:41, Temp:28.1C, Hum:39%, pH:6.7, Rain:44mm
  - Sesame: N:61, P:40, K:39, Temp:29.8C, Hum:49%, pH:6.8, Rain:55mm
  - Sorghum: N:100, P:49, K:49, Temp:27.1C, Hum:50%, pH:6.9, Rain:57mm
  - Soybeans: N:41, P:60, K:60, Temp:25.0C, Hum:61%, pH:6.5, Rain:76mm
  - Spinach: N:81, P:31, K:50, Temp:14.8C, Hum:71%, pH:6.7, Rain:64mm
  - Squash: N:80, P:50, K:70, Temp:24.9C, Hum:70%, pH:6.7, Rain:99mm
  - Strawberry: N:100, P:50, K:70, Temp:20.0C, Hum:70%, pH:6.0, Rain:99mm
  - Sugarcane: N:124, P:65, K:50, Temp:27.6C, Hum:78%, pH:6.9, Rain:201mm
  - Sunflower: N:80, P:50, K:50, Temp:25.0C, Hum:50%, pH:6.7, Rain:65mm
  - Sweet Potato: N:61, P:50, K:78, Temp:25.0C, Hum:69%, pH:6.0, Rain:101mm
  - Tea: N:100, P:51, K:51, Temp:20.5C, Hum:81%, pH:5.0, Rain:196mm
  - Turnip: N:62, P:41, K:49, Temp:17.4C, Hum:70%, pH:6.2, Rain:63mm
  - Walnut: N:98, P:50, K:70, Temp:20.3C, Hum:60%, pH:6.7, Rain:80mm
  - Watermelon: N:99, P:17, K:50, Temp:25.6C, Hum:85%, pH:6.5, Rain:51mm
  - Wheat: N:92, P:50, K:49, Temp:19.8C, Hum:60%, pH:6.8, Rain:76mm
  - Yam: N:81, P:51, K:79, Temp:29.7C, Hum:70%, pH:6.0, Rain:121mm
  - Zucchini: N:79, P:50, K:69, Temp:24.5C, Hum:71%, pH:6.7, Rain:100mm

ABOUT FARMAI:
- Uses Random Forest ML model trained on soil & climate data
- Inputs: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Model accuracy: ~99%
- Also has: Plant Disease Detection (Gemini AI), Fertilizer Advisor, History Tracking

You also help farmers with crop selection, disease treatment, fertilizer advice,
irrigation, pest control, harvesting tips, and any agriculture-related knowledge.

CRITICAL FIELD ENFORCEMENT:
If the user's question, uploaded text document, or uploaded image is NOT related to agriculture, farming, crops, soil, plants, or botany, you MUST refuse to answer or analyze it. You MUST explicitly reply with exactly: "The file is not related to the chatbot field." and say nothing else.

Be friendly, concise, practical. DO NOT cut off lists. If asked to list crops, list all 101 crops entirely without summarizing or stopping early."""

@app.post("/chat")
def chat(req: ChatMessage):
    # Support comma-separated keys; pick the first one
    grok_pool = [k.strip() for k in GROK_API_KEY.split(",") if k.strip()]
    if not grok_pool:
        return {"reply": "Chatbot is not configured. Please add GROK_API_KEY to .env file."}
    active_key = grok_pool[0]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 1500, "temperature": 0.7},
            timeout=20
        )
        if resp.status_code == 400:
            return {"reply": f"API Error: {resp.json().get('error', {}).get('message', 'Bad request.')}"}
        if resp.status_code == 401:
            return {"reply": "Invalid API key. Please update GROK_API_KEY in your .env file."}
        resp.raise_for_status()
        return {"reply": resp.json()["choices"][0]["message"]["content"]}
    except http_requests.exceptions.Timeout:
        return {"reply": "Sorry, the response took too long. Please try again."}
    except Exception as e:
        return {"reply": f"Connection error: {str(e)[:80]}"}


@app.post("/chat-file")
async def chat_file(
    file: UploadFile = File(...),
    message: str = Form(default="Analyze this and give agricultural advice.")
):
    import base64
    contents = await file.read()
    fname = file.filename.lower()
    ext = fname.rsplit(".", 1)[-1] if "." in fname else ""

    # ---- IMAGE: use Gemini Vision ----
    if ext in ("jpg", "jpeg", "png", "webp", "gif"):
        # Support multiple keys
        gem_pool = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
        if not gem_pool:
            return {"reply": "Gemini API key not configured for image analysis."}
        active_gemini_key = gem_pool[0]

        b64 = base64.b64encode(contents).decode("utf-8")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        prompt = f"""{SYSTEM_PROMPT}

User uploaded an image and asks: "{message}"
First, verify if this image is related to agriculture, farming, crops, soil, or plants.
If it is NOT related, you MUST reply with exactly: "The file is not related to the chatbot field."
If it IS related, analyze the image from an agricultural perspective. Identify:
- What crop or plant is shown
- Any visible diseases, pests, or deficiencies
- Health status
- Recommended treatment or action
Be specific and practical."""

        try:
            # Use stable v1beta endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={active_gemini_key}"
            resp = http_requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime, "data": b64}}]}],
                      "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}},
                timeout=30
            )
            if resp.status_code == 429:
                return {"reply": "The analysis engine is currently busy (Rate Limit). Please wait a moment and try again."}
            resp.raise_for_status()
            reply = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {"reply": reply}
        except Exception as e:
            return {"reply": f"Image analysis failed: {str(e)[:100]}"}

    # ---- PDF: extract text then send to Groq ----
    elif ext == "pdf":
        try:
            import pypdf
            import io as _io
            reader = pypdf.PdfReader(_io.BytesIO(contents))
            text = "\n".join(p.extract_text() or "" for p in reader.pages[:5]).strip()
            if not text:
                return {"reply": "Could not extract text from this PDF. It may be a scanned image PDF."}
            text = text[:3000]  # limit tokens
        except Exception as e:
            return {"reply": f"Could not read PDF: {str(e)[:80]}"}

        user_msg = f'User uploaded a document and asks: "{message}"\n\nDocument content:\n{text}'

    # ---- TXT / CSV ----
    elif ext in ("txt", "csv"):
        try:
            text = contents.decode("utf-8", errors="ignore")[:3000]
        except:
            return {"reply": "Could not read this file."}
        user_msg = f'User uploaded a text file and asks: "{message}"\n\nFile content:\n{text}'

    else:
        return {"reply": f"Unsupported file type '.{ext}'. Please upload an image (JPG/PNG), PDF, or TXT file."}

    # Send document text to Groq
    # Support multiple keys for Groq
    grok_pool = [k.strip() for k in GROK_API_KEY.split(",") if k.strip()]
    if not grok_pool:
        return {"reply": "Chatbot API key not configured."}
    active_grok_key = grok_pool[0]

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {active_grok_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                  "max_tokens": 1500, "temperature": 0.5},
            timeout=20
        )
        resp.raise_for_status()
        return {"reply": resp.json()["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"reply": f"Analysis failed: {str(e)[:80]}"}


# ----------------------------------------
# HTML SERVING
# ----------------------------------------

@app.get("/")
def serve_home(): return FileResponse(os.path.join(BASE_DIR, "crop ui.html"))

@app.get("/login")
@app.get("/login-page")
@app.get("/login.html")
def serve_login(): return FileResponse(os.path.join(BASE_DIR, "login.html"))

@app.get("/register")
@app.get("/register-page")
@app.get("/register.html")
def serve_register(): return FileResponse(os.path.join(BASE_DIR, "register.html"))

@app.get("/verify")
@app.get("/verify-page")
@app.get("/verify.html")
def serve_verify(): return FileResponse(os.path.join(BASE_DIR, "verify.html"))

@app.get("/profile")
@app.get("/profile-page")
@app.get("/profile.html")
def serve_profile(): return FileResponse(os.path.join(BASE_DIR, "profile.html"))

@app.get("/results")
@app.get("/results-page")
@app.get("/results.html")
def serve_results(): return FileResponse(os.path.join(BASE_DIR, "results.html"))

@app.get("/change-password")
@app.get("/change-password-page")
@app.get("/change_password.html")
def serve_change_password(): return FileResponse(os.path.join(BASE_DIR, "change_password.html"))

@app.get("/disease")
@app.get("/disease-page")
@app.get("/disease.html")
def serve_disease(): return FileResponse(os.path.join(BASE_DIR, "disease.html"))

@app.get("/history")
@app.get("/history-page")
@app.get("/history.html")
def serve_history(): return FileResponse(os.path.join(BASE_DIR, "history.html"))

@app.get("/fertilizer")
@app.get("/fertilizer-page")
@app.get("/fertilizer.html")
def serve_fertilizer(): return FileResponse(os.path.join(BASE_DIR, "fertilizer.html"))

@app.get("/chatbot.html")
def serve_chatbot(): return FileResponse(os.path.join(BASE_DIR, "chatbot.html"))

@app.get("/googlebf6a1bb40761389f.html")
def serve_google_verification(): return FileResponse(os.path.join(BASE_DIR, "googlebf6a1bb40761389f.html"))

from typing import Dict, Any
from explanation_engine import generate_farmer_explanation

@app.post("/explain")
async def explain_disease(data: Dict[str, Any]):
    return generate_farmer_explanation(
        disease=data.get("disease", ""),
        confidence=data.get("confidence", 0.0),
        stage=data.get("stage") or data.get("severity", "Moderate")
    )
from gtts import gTTS
import base64
import io

def generate_voice_base64(text, language="en"):
    try:
        # fallback language
        if language not in ["en", "hi"]:
            language = "en"

        tts = gTTS(text=text, lang=language)
        buf = io.BytesIO()
        tts.write_to_fp(buf)

        audio_bytes = buf.getvalue()
        return "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode()

    except Exception as e:
        print("TTS ERROR:", e)
        return ""
@app.post("/full-explain")
async def full_explain_disease(data: Dict[str, Any]):
    disease = data.get("disease", "")
    confidence = float(data.get("confidence", 1.0))
    stage = data.get("stage") or data.get("severity", "Moderate")
    area = float(data.get("area", 1.0))
    language = data.get("language", "hi").lower() # Hindi default!
    
    try:
        # 1. Fetch LLM explanation
        exp = generate_ai_explanation(disease, confidence, stage, language)
        
        # 2. Dynamic Cost Prediction
        cost_data = estimate_cost(disease, stage, area)
        exp["cost"] = cost_data["estimated_cost"]
        exp["total_cost"] = cost_data["total_cost"]
        
        # 3. Voice Explanation (TTS) - create a sensible script combining summary and steps
        actions_txt = ", ".join(exp.get("actions", []))
        tts_script = f"{exp.get('title', '')}. {exp.get('summary', '')}. Actions you must take: {actions_txt}"
        exp["audio_url"] = generate_voice_base64(tts_script, language)
        
        return exp
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi.responses import FileResponse

@app.get("/favicon.ico")
def favicon():
    return FileResponse(os.path.join(BASE_DIR, "favicon.ico"))
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)