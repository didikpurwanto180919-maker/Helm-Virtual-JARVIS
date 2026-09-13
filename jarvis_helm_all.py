import streamlit as st
import cv2
import numpy as np
import requests
from datetime import datetime
import os
import urllib.request
from google import genai

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="JARVIS Virtual Visor", 
    layout="wide", 
    page_icon="🛡️"
)

# Custom CSS untuk tampilan lebih Sci-Fi
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
        color: #c9d1d9;
    }
    .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #58a6ff;
        text-shadow: 0 0 10px #58a6ff;
    }
    .stCameraInput > div > div {
        border: 2px solid #58a6ff;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ JARVIS Virtual Helmet Visor")

# --- 2. SIDEBAR SETUP (API KEYS) ---
with st.sidebar:
    st.header("⚙️ Sistem Pengaturan")
    # Disarankan menggunakan st.secrets untuk deployment produk
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    KOTA = st.text_input("Kota untuk Cuaca", value="Jakarta")
    st.write("---")
    st.write("Visor JARVIS: v2.8 - Status: Operational")

# --- 3. PEMUATAN HAAR CASCADE (SANGAT STABIL) ---
@st.cache_resource
def load_face_cascade():
    """
    Memuat Haar Cascade. Jika tidak ada secara lokal, sistem akan mengunduhnya.
    Namun, disarankan memasukkan file XML langsung ke repo GitHub Anda.
    """
    xml_filename = "haarcascade_frontalface_default.xml"
    
    # Cek apakah modul cv2 rusak (sering terjadi jika headless tidak lengkap)
    if not hasattr(cv2, 'CascadeClassifier'):
        st.error("FATAL ERROR: Modul cv2 terdeteksi rusak (tanpa CascadeClassifier). Periksa requirements.txt Anda.")
        return None

    # Cek apakah file lokal ada
    if not os.path.exists(xml_filename):
        # Fallback: Unduh jika file lokal tidak ditemukan
        with st.spinner(f"Unduh file model {xml_filename}..."):
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, xml_filename)
            except Exception as e:
                st.error(f"Gagal mengunduh file classifier: {e}")
                return None
    
    cascade = cv2.CascadeClassifier(xml_filename)
    if cascade.empty():
        st.error(f"Gagal memuat Cascade Classifier dari {xml_filename}.")
        return None
    return cascade

# Coba muat cascade secara global
face_cascade = load_face_cascade()

# --- 4. FUNGSI LAYANAN (CUACA & AI) ---
def get_weather(kota):
    """Mengambil informasi cuaca dari wttr.in (gratis dan tanpa API key)"""
    try:
        url = f"https://wttr.in/{kota}?format=%t+%C"
        res = requests.get(url, timeout=3).text
        return res.strip()
    except Exception:
        return "27 C - Operational"

def talk_to_jarvis(query, api_key):
    """Menghubungi protokol AI Gemini untuk mendapatkan respon singkat ala JARVIS"""
    if not api_key or "MASUKKAN" in api_key:
        return "⚠️ Masukkan API Key untuk mengaktifkan AI."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Jawab ringkas dalam 1-2 kalimat pendek ala asisten JARVIS Iron Man: {query}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Protokol komunikasi AI gagal: {e}"

# --- 5. INTERFACE HUD KAMERA ---
st.subheader("📸 Visor Interface")

# Cek kondisi sebelum menampilkan kamera
if face_cascade is not None and not face_cascade.empty():
    picture = st.camera_input("Ambil foto dari Visor Helm")

    if picture:
        # Konversi gambar dari Streamlit ke OpenCV
        bytes_data = picture.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        # --- DESAIN OVERLAYS HUD ---
        h, w, _ = img.shape
        color_cyan = (255, 255, 0)
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)
        
        weather_info = get_weather(KOTA)
        now = datetime.now().strftime("%H:%M:%S | %d-%m-%Y")
        
        # Render Box Header HUD
        cv2.rectangle(img, (10, 10), (w - 10, 85), (20, 20, 20), -1) # Background box
        cv2.rectangle(img, (10, 10), (w - 10, 85), color_cyan, 1) # Border box
        
        # Text Header HUD
        cv2.putText(img, "HELM VISOR JARVIS - SYSTEM ONLINE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_cyan, 2)
        cv2.putText(img, f"WAKTU : {now}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"CUACA : {KOTA} ({weather_info})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_green, 1)
        
        # Render Boks Deteksi Wajah & Reticle
        if len(faces) == 0:
            st.toast("Pilot Tidak Terdeteksi", icon="ℹ️")
        
        for (x, y, fw, fh) in faces:
            # Box Targeting
            cv2.rectangle(img, (x, y), (x + fw, y + fh), color_cyan, 2)
            
            # Center Reticle (Silang Merah)
            cx, cy = x + fw // 2, y + fh // 2
            cv2.line(img, (cx - 15, cy), (cx + 15, cy), color_red, 2)
            cv2.line(img, (cx, cy - 15), (cx, cy + 15), color_red, 2)
            
            # Label Target
            cv2.putText(img, "TARGET DETECTED: PILOT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_cyan, 1)
            st.toast("Wajah Pilot Dikenali", icon="✅")
            
        # Tampilkan Gambar HUD ke Layar Streamlit
        # Konversi ke RGB karena Streamlit membutuhkan RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Output Visual HUD Visor", use_container_width=True)
else:
    st.error("⚠️ Sistem Deteksi Wajah gagal diaktifkan. Periksa log atau requirements.txt Anda.")
    st.camera_input("Ambil foto (Hanya Cuaca)")

# --- 6. INTEGRASI ASISTEN AI GEMINI ---
st.write("---")
st.subheader("🗣️ Tanya JARVIS AI")
query = st.text_input("Tanya Protokol AI:", placeholder="JARVIS, apa cuaca hari ini?")

if query:
    if not GEMINI_API_KEY:
        st.warning("Masukkan Gemini API Key pada sidebar kiri terlebih dahulu.")
    else:
        with st.spinner("Menghubungi protokol JARVIS..."):
            response = talk_to_jarvis(query, GEMINI_API_KEY)
            if "⚠️" in response:
                st.error(response)
            else:
                st.info(f"🤖 **JARVIS:** {response}")
