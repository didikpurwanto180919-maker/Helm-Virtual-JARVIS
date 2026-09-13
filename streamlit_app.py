import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import requests
import urllib.request
from google import genai
import time

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="H.E.L.M. Visor - Autonomous Rider Assistant",
    layout="wide",
    page_icon="🪖"
)

# Custom CSS untuk gaya minimalis futuristik
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #111;
        color: white;
    }
    h1, h2, h3 {
        color: #0ff; /* Cyan */
    }
    .stCameraInput {
        border: 2px solid #0ff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    .stInfo {
        background-color: rgba(0, 255, 255, 0.1);
        color: #0ff;
        border: 1px solid #0ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ H.E.L.M. Visor System")
st.caption("Heuristic Electronic Location & Monitoring Visor")

# --- INISIALISASI STATE ---
if "face_cascade" not in st.session_state:
    st.session_state.face_cascade = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR KONFIGURASI ---
with st.sidebar:
    st.header("⚙️ System Config")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    LOCATION = st.text_input("Current Location", value="Jakarta")
    st.write("---")
    st.write("H.E.L.M. Core v1.0 | Status: Online")

# --- FUNGSI CORE ---

# 1. Pemuatan Haar Cascade untuk Deteksi Wajah
def load_cascade():
    if st.session_state.face_cascade is not None:
        return st.session_state.face_cascade
    
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        with st.spinner("Downloading H.E.L.M. Visual Database..."):
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, xml_filename)
            except Exception as e:
                st.error(f"Visual Database Sync Failed: {e}")
                return None
    
    # Memastikan modul CV2 memiliki atribut CascadeClassifier
    # Error libGL.so.1 diperbaiki dengan menggunakan opencv-python-headless di requirements.txt
    if not hasattr(cv2, 'CascadeClassifier'):
         st.error("OpenCV module is incomplete. Verify install of opencv-python-headless.")
         return None

    cascade = cv2.CascadeClassifier(xml_filename)
    if cascade.empty():
        st.error(f"Failed to load Visual Database from {xml_filename}.")
        return None
        
    st.session_state.face_cascade = cascade
    return cascade

face_cascade = load_cascade()

# 2. Pengambilan Data Cuaca (API-less fallback)
def fetch_weather(location):
    """Fallback weather check using wttr.in, which requires no API key."""
    try:
        url = f"https://wttr.in/{location}?format=%t+%C"
        response = requests.get(url, timeout=3).text
        return response.strip()
    except Exception:
        return "Operational" # Default status if request fails

# 3. Komunikasi Asisten AI (Gemini)
def get_rider_asist(query, api_key):
    """Sends visual/text data to Gemini for analysis."""
    if not api_key:
        return "⚠️ H.E.L.M. Core AI Offline: Gemini API Key missing."
    try:
        client = genai.Client(api_key=api_key)
        # Prompt singkat agar respon cepat
        prompt = (
            "Response as H.E.L.M., a practical and focused rider assistant AI inside a helmet visor. "
            "Keep the response in Indonesian, very brief, clear, and centered on rider safety or info. "
            f"Question: {query}"
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', # Menggunakan flash untuk kecepatan
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Protokol komunikasi AI terganggu: {e}"

# --- MAIN INTERFACE: KAMERA & HUD ---
st.subheader("📷 Visor Output")

if face_cascade is not None:
    # Kamera Input
    camera_image = st.camera_input("Rider Safety Scan (Tap to Analyze)")

    if camera_image:
        # Konversi data input ke format OpenCV
        image_bytes = camera_image.getvalue()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Proses deteksi wajah (targeting)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        # --- RENDER HUD OVERLAY (OpenCV) ---
        h, w, _ = img.shape
        weather_info = fetch_weather(LOCATION)
        now_time = datetime.now().strftime("%H:%M:%S")
        now_date = datetime.now().strftime("%d/%m/%Y")
        
        color_cyan = (255, 255, 0) # OpenCV uses BGR
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)
        
        # 1. Header Box & Status
        cv2.rectangle(img, (10, 10), (w - 10, 85), (20, 20, 20), -1) # Dark fill
        cv2.rectangle(img, (10, 10), (w - 10, 85), color_cyan, 1) # Cyan border
        
        cv2.putText(img, "H.E.L.M. VISOR - ACTIVE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_cyan, 2)
        cv2.putText(img, f"TIME: {now_time}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"LOC: {LOCATION} ({weather_info})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_green, 1)
        
        # 2. Targeting Wajah
        if len(faces) == 0:
            cv2.putText(img, "NO RIDER DETECTED", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 1)
        
        for (x, y, fw, fh) in faces:
            # Kotak target
            cv2.rectangle(img, (x, y), (x + fw, y + fh), color_cyan, 2)
            
            # Crosshair tengah
            cx, cy = x + fw // 2, y + fh // 2
            cv2.line(img, (cx - 15, cy), (cx + 15, cy), color_red, 2)
            cv2.line(img, (cx, cy - 15), (cx, cy + 15), color_red, 2)
            
            # Label
            cv2.putText(img, "TARGET: RIDER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_cyan, 1)
            cv2.putText(img, "SCANNING OK", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)

        # 3. Render Output ke Streamlit (RGB conversion)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"H.E.L.M. Scan at {now_time}", use_container_width=True)

else:
    st.error("⚠️ Visual Database not initialized. Scan function offline.")

# --- INTERFACE CHAT: RIDER COMMUNICATION ---
st.write("---")
st.subheader("🗣️ Rider Comms (H.E.L.M. Core)")

# Container untuk history chat
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input untuk rider
rider_query = st.chat_input("H.E.L.M., any safety issues ahead?")

if rider_query:
    # 1. Simpan history user
    st.session_state.chat_history.append({"role": "user", "content": rider_query})
    with st.chat_message("user"):
        st.markdown(rider_query)
    
    # 2. Proses AI Assistant
    with st.spinner("Analyzing Comms Protocol..."):
        ai_response = get_rider_asist(rider_query, GEMINI_API_KEY)
        # Menambahkan delay sedikit agar seperti berpikir
        time.sleep(0.5)

    # 3. Simpan dan tampilkan history AI
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
