import streamlit as st
import cv2
import numpy as np
import requests
from datetime import datetime
import os
from google import genai

# Config Halaman Streamlit
st.set_page_config(page_title="JARVIS Virtual Visor", layout="wide")

st.title("🛡️ JARVIS Virtual Helmet Visor")

# API Setup
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
KOTA = st.sidebar.text_input("Kota untuk Cuaca", value="Jakarta")

# Load Cascade Classifier
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

# Fungsi Cuaca
def get_weather(kota):
    try:
        url = f"https://wttr.in/{kota}?format=%t+%C"
        res = requests.get(url, timeout=3).text
        return res.strip()
    except Exception:
        return "27 C - Operational"

# Fitur Input Kamera Web
picture = st.camera_input("Ambil foto dari Visor Helm")

if picture:
    # Konversi gambar Streamlit ke format OpenCV
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi Wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    # Draw HUD Overlay
    h, w, _ = img.shape
    weather_info = get_weather(KOTA)
    now = datetime.now().strftime("%H:%M:%S | %d-%m-%Y")
    
    # Header HUD Box
    cv2.rectangle(img, (10, 10), (w - 10, 85), (20, 20, 20), -1)
    cv2.rectangle(img, (10, 10), (w - 10, 85), (255, 255, 0), 1)
    cv2.putText(img, "HELM VISOR JARVIS ONLINE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"WAKTU : {now}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"CUACA : {KOTA} ({weather_info})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    # Tracking Box Wajah
    for (x, y, fw, fh) in faces:
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (255, 255, 0), 2)
        cx, cy = x + fw // 2, y + fh // 2
        cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
        cv2.putText(img, "TARGET DETECTED: PILOT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
    # Tampilkan Hasil ke Streamlit Interface
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Hasil Output HUD Visor", use_container_width=True)

# Tanya JARVIS AI
st.write("---")
query = st.text_input("Tanya JARVIS AI:")
if query and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Jawab ringkas dalam 1-2 kalimat pendek ala asisten JARVIS Iron Man: {query}"
        )
        st.info(f"**JARVIS:** {response.text}")
    except Exception as e:
        st.error(f"Gagal menghubungi JARVIS AI: {e}")
