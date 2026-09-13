import streamlit as st
import cv2
import numpy as np
import requests
from datetime import datetime
import os
import urllib.request
from google import genai

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="JARVIS Virtual Visor", layout="wide")
st.title("🛡️ JARVIS Virtual Helmet Visor")

# Sidebar untuk API Keys dan Pengaturan
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
KOTA = st.sidebar.text_input("Kota untuk Cuaca", value="Jakarta")

# --- FUNGSI LOAD HAAR CASCADE AMAN (DOWNLOADING AUTOMATIC) ---
@st.cache_resource
def load_face_cascade():
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, xml_filename)
        except Exception as e:
            st.error(f"Gagal mengunduh file classifier: {e}")
            return None
    return cv2.CascadeClassifier(xml_filename)

face_cascade = load_face_cascade()

# --- FUNGSI INFORMASI CUACA ---
def get_weather(kota):
    try:
        url = f"https://wttr.in/{kota}?format=%t+%C"
        res = requests.get(url, timeout=3).text
        return res.strip()
    except Exception:
        return "27 C - Operational"

# --- INTERFACE KAMERA WEBCAM ---
picture = st.camera_input("Ambil foto dari Visor Helm")

if picture and face_cascade is not None:
    # Konversi gambar dari Streamlit ke OpenCV
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi Wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    # Desain Overlays HUD
    h, w, _ = img.shape
    weather_info = get_weather(KOTA)
    now = datetime.now().strftime("%H:%M:%S | %d-%m-%Y")
    
    # Render Box Header HUD
    cv2.rectangle(img, (10, 10), (w - 10, 85), (20, 20, 20), -1)
    cv2.rectangle(img, (10, 10), (w - 10, 85), (255, 255, 0), 1)
    cv2.putText(img, "HELM VISOR JARVIS ONLINE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"WAKTU : {now}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"CUACA : {KOTA} ({weather_info})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    # Render Boks Deteksi Wajah
    for (x, y, fw, fh) in faces:
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (255, 255, 0), 2)
        cx, cy = x + fw // 2, y + fh // 2
        cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
        cv2.putText(img, "TARGET DETECTED: PILOT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
    # Tampilkan Gambar HUD ke Layar Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Hasil Visual Visor HUD", use_container_width=True)

# --- INTEGRASI ASISTEN AI GEMINI ---
st.write("---")
query = st.text_input("Tanya JARVIS AI:")
if query:
    if not GEMINI_API_KEY:
        st.warning("Masukkan Gemini API Key pada sidebar kiri terlebih dahulu.")
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Jawab ringkas dalam 1-2 kalimat pendek ala asisten JARVIS Iron Man: {query}"
            )
            st.info(f"**JARVIS:** {response.text}")
        except Exception as e:
            st.error(f"Gagal memproses respon AI: {e}")
