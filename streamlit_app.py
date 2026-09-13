"""
H.E.L.M. Visor System - Virtual Helm JARVIS Edition
===================================================
Aplikasi Streamlit dengan integrasi OpenCV (AR Overlay Iron Man)
dan Google Gemini AI untuk asisten berkendara.
"""

import os
import time
import json
import urllib.request
import numpy as np
import cv2
import requests
import streamlit as st
from google import genai
from google.genai import types

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="H.E.L.M. Visor System",
    page_icon="🪖",
    layout="wide"
)

# Style Futuristik
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .stCameraInput {
        border: 2px solid #238636;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪖 H.E.L.M. Visor System")
st.caption("Heuristic Electronic Location & Monitoring Visor — Powered by JARVIS AI")

# --- INISIALISASI SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# --- SIDEBAR KONFIGURASI ---
with st.sidebar:
    st.header("⚙️ System Config")
    
    # Kunci API Gemini
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Masukkan API Key Gemini Anda"
    )
    
    location_input = st.text_input("Current Location", value="Jakarta")
    
    model_name = st.selectbox(
        "Model Gemini AI",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0
    )
    
    st.markdown("---")
    pasang_helm_ar = st.checkbox("🪖 Pasang Helm Iron Man (AR Overlay)", value=True)
    
    st.markdown("---")
    st.write("H.E.L.M. Core v2.0 | Status: **ONLINE**")


# --- FUNGSI DETEKSI WAJAH (HAAR CASCADE) ---
@st.cache_resource
def load_face_cascade():
    """Mengunduh dan memuat Haar Cascade Classifier secara aman."""
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, xml_filename)
        except Exception as e:
            return None, f"Gagal mengunduh cascade: {e}"
            
    try:
        cascade = cv2.CascadeClassifier(xml_filename)
        if cascade.empty():
            return None, "File cascade kosong atau tidak valid."
        return cascade, None
    except Exception as e:
        return None, f"Error memuat OpenCV Cascade: {e}"

face_cascade, cascade_error = load_face_cascade()


# --- FUNGSI MODEL HELM VIRTUAL IRON MAN (AR) ---
def buat_helm_ironman(lebar: int, tinggi: int) -> np.ndarray:
    """Menggambar geometri Helm Iron Man futuristik (BGRA) secara prosedural."""
    kanvas = np.zeros((tinggi, lebar, 4), dtype=np.uint8)

    pusat_x, pusat_y = lebar // 2, int(tinggi * 0.5)
    sumbu_x, sumbu_y = int(lebar * 0.44), int(tinggi * 0.46)

    # Warna khas Iron Man (BGR + Alpha)
    warna_merah = (20, 15, 180, 255)       # Red metallic
    warna_emas = (40, 200, 240, 255)       # Gold faceplate
    warna_mata = (255, 240, 200, 255)      # Cyan/white glow
    warna_garis = (10, 10, 100, 255)       # Contour lines

    # 1. Tempurung Luar Helm (Merah)
    cv2.ellipse(kanvas, (pusat_x, pusat_y), (sumbu_x, sumbu_y), 0, 0, 360, warna_merah, -1)

    # 2. Plat Wajah Depan (Emas/Gold Faceplate)
    titik_plat_emas = np.array([
        [pusat_x - int(sumbu_x * 0.65), pusat_y - int(sumbu_y * 0.60)],
        [pusat_x + int(sumbu_x * 0.65), pusat_y - int(sumbu_y * 0.60)],
        [pusat_x + int(sumbu_x * 0.75), pusat_y - int(sumbu_y * 0.10)],
        [pusat_x + int(sumbu_x * 0.55), pusat_y + int(sumbu_y * 0.45)],
        [pusat_x + int(sumbu_x * 0.35), pusat_y + int(sumbu_y * 0.85)],
        [pusat_x - int(sumbu_x * 0.35), pusat_y + int(sumbu_y * 0.85)],
        [pusat_x - int(sumbu_x * 0.55), pusat_y + int(sumbu_y * 0.45)],
        [pusat_x - int(sumbu_x * 0.75), pusat_y - int(sumbu_y * 0.10)],
    ], dtype=np.int32)
    cv2.fillPoly(kanvas, [titik_plat_emas], warna_emas)

    # 3. Slot Mata menyala (Glowing LED)
    tinggi_mata = int(sumbu_y * 0.08)
    pos_y_mata = pusat_y - int(sumbu_y * 0.20)

    mata_kiri = np.array([
        [pusat_x - int(sumbu_x * 0.60), pos_y_mata],
        [pusat_x - int(sumbu_x * 0.15), pos_y_mata + int(tinggi_mata * 0.4)],
        [pusat_x - int(sumbu_x * 0.20), pos_y_mata + tinggi_mata],
        [pusat_x - int(sumbu_x * 0.55), pos_y_mata + int(tinggi_mata * 0.7)],
    ], dtype=np.int32)
    
    mata_kanan = np.array([
        [pusat_x + int(sumbu_x * 0.15), pos_y_mata + int(tinggi_mata * 0.4)],
        [pusat_x + int(sumbu_x * 0.60), pos_y_mata],
        [pusat_x + int(sumbu_x * 0.55), pos_y_mata + int(tinggi_mata * 0.7)],
        [pusat_x + int(sumbu_x * 0.20), pos_y_mata + tinggi_mata],
    ], dtype=np.int32)

    cv2.fillPoly(kanvas, [mata_kiri], warna_mata)
    cv2.fillPoly(kanvas, [mata_kanan], warna_mata)

    # 4. Garis Panel & Detail Mulut
    cv2.polylines(kanvas, [titik_plat_emas], isClosed=True, color=warna_garis, thickness=max(2, lebar // 100))
    y_mulut = pusat_y + int(sumbu_y * 0.55)
    cv2.line(kanvas, (pusat_x - int(sumbu_x * 0.30), y_mulut), (pusat_x + int(sumbu_x * 0.30), y_mulut), warna_garis, thickness=max(2, lebar // 90))

    # Blur halus untuk transparansi tepi
    alpha = kanvas[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    kanvas[:, :, 3] = (alpha * 255).astype(np.uint8)

    return kanvas


def tempel_overlay(frame_bgr, overlay_bgra, x, y):
    """Penggabungan Alpha Blending gambar BGRA di atas BGR."""
    h_ov, w_ov = overlay_bgra.shape[:2]
    h_f, w_f = frame_bgr.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_ov, w_f), min(y + h_ov, h_f)
    if x1 >= x2 or y1 >= y2:
        return frame_bgr

    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    bagian_overlay = overlay_bgra[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha = bagian_overlay[:, :, 3:4].astype(np.float32) / 255.0
    warna_ov = bagian_overlay[:, :, :3].astype(np.float32)

    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    hasil = warna_ov * alpha + roi * (1 - alpha)
    frame_bgr[y1:y2, x1:x2] = hasil.astype(np.uint8)
    return frame_bgr


def render_visor_hud(frame_bgr, face_cascade_obj, aktifkan_ar=True):
    """Memproses frame dengan Deteksi Wajah, Overlay Helm, dan Indikator HUD."""
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    hasil = frame_bgr.copy()
    
    # Deteksi Wajah
    wajah_terdeteksi = []
    if face_cascade_obj is not None:
        wajah_terdeteksi = face_cascade_obj.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (fx, fy, fw, fh) in wajah_terdeteksi:
        if aktifkan_ar:
            # Pemasangan Helm Iron Man
            lebar_helm = int(fw * 1.6)
            tinggi_helm = int(fh * 1.9)
            helm = buat_helm_ironman(lebar_helm, tinggi_helm)

            pos_x = fx - int((lebar_helm - fw) / 2)
            pos_y = fy - int(tinggi_helm * 0.25)
            hasil = tempel_overlay(hasil, helm, pos_x, pos_y)
        
        # Bounding Box HUD Visual Target
        cv2.rectangle(hasil, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        cv2.putText(hasil, "TARGET: RIDER", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Top HUD Bar
    cv2.rectangle(hasil, (10, 10), (w - 10, 50), (15, 15, 15), -1)
    cv2.rectangle(hasil, (10, 10), (w - 10, 50), (255, 255, 0), 1)
    waktu_str = time.strftime("%H:%M:%S")
    cv2.putText(hasil, f"H.E.L.M. HUD | TIME: {waktu_str} | LOC: {location_input.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return hasil


# --- INTEGRASI GEMINI AI ---
def analisis_visor_ai(api_key, frame_bgr, prompt_user, model):
    """Mengirim gambar frame visor dan pertanyaan ke AI Gemini."""
    if not api_key:
        return "⚠️ Kunci API Gemini belum diisi di menu sidebar."
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Encode frame BGR ke JPEG bytes
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        image_bytes = buffer.tobytes()
        
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        
        system_prompt = (
            "Kamu adalah JARVIS, AI asisten di dalam helm pengemudi (H.E.L.M. Visor). "
            "Jawab singkat, presisi, ringkas, dan utamakan keselamatan jalan dalam Bahasa Indonesia.\n"
            f"Pertanyaan Pengendara: {prompt_user if prompt_user else 'Analisis bahaya jalan di depan.'}"
        )
        
        res = client.models.generate_content(
            model=model,
            contents=[system_prompt, image_part]
        )
        return res.text
    except Exception as e:
        return f"Gagal menghubungkan ke H.E.L.M. Core AI: {e}"


# --- LAYOUT ANTARMUKA UTAMA ---
col_kamera, col_chat = st.columns([1.1, 0.9])

with col_kamera:
    st.subheader("📷 Visor Output")
    
    if cascade_error:
        st.warning(f"⚠️ Warning Sistem Visual: {cascade_error}")
        
    camera_input = st.camera_input("Ambil gambar snapshot visor")
    
    if camera_input:
        bytes_data = camera_input.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Simpan frame asli ke session
        st.session_state.last_frame = frame_bgr
        
        # Render dengan Overlay AR
        frame_hud = render_visor_hud(frame_bgr, face_cascade, aktifkan_ar=pasang_helm_ar)
        
        # Tampilkan di Streamlit (Ubah BGR ke RGB)
        st.image(cv2.cvtColor(frame_hud, cv2.COLOR_BGR2RGB), caption="Visor HUD Stream", use_container_width=True)
    else:
        st.info("Kamera siap. Silahkan ambil gambar untuk memulai pemindaian HUD.")

with col_chat:
    st.subheader("🗣️ Rider Comms (H.E.L.M. Core)")
    
    # Tampilan Riwayat Chat
    chat_box = st.container(height=380)
    with chat_box:
        for chat in st.session_state.chat_history:
            role = chat["role"]
            avatar = "🪖" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.write(chat["content"])

    # Input Teks Pertanyaan
    user_query = st.chat_input("H.E.L.M., ada bahaya di depan?")
    
    if user_query:
        # Tampilkan input user
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("JARVIS menganalisis input..."):
            if st.session_state.last_frame is not None:
                jawaban_ai = analisis_visor_ai(api_key_input, st.session_state.last_frame, user_query, model_name)
            else:
                # Fallback tanya jawab tanpa gambar jika belum ambil snapshot
                if api_key_input:
                    try:
                        c = genai.Client(api_key=api_key_input)
                        resp = c.models.generate_content(
                            model=model_name,
                            contents=f"Kamu adalah JARVIS asisten helm. Jawab singkat: {user_query}"
                        )
                        jawaban_ai = resp.text
                    except Exception as err:
                        jawaban_ai = f"Error: {err}"
                else:
                    jawaban_ai = "⚠️ Masukkan API Key Gemini di sidebar untuk berkomunikasi."
        
        st.session_state.chat_history.append({"role": "assistant", "content": jawaban_ai})
        st.rerun()
