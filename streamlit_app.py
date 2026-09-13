"""
H.E.L.M. Visor System - Fixing Virtual Helmet Overlay
=====================================================
Aplikasi Streamlit untuk memasang Helm Virtual pada foto/kamera secara otomatis.
"""

import os
import time
import urllib.request
import numpy as np
import cv2
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

st.title("🪖 H.E.L.M. Visor System - Virtual Helmet AR")
st.caption("Auto-Detect Face & Fit Virtual Helmet Overlay")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Masukkan API Key Gemini Anda"
    )
    
    model_name = st.selectbox(
        "Model Gemini AI",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0
    )
    
    st.markdown("---")
    mode_input = st.radio("Sumber Foto:", ["Ambil Foto (Kamera)", "Upload Foto dari Perangkat"])
    pasang_helm_ar = st.checkbox("🪖 Paksa Tampil Helm Virtual", value=True)
    
    st.markdown("---")
    st.write("H.E.L.M. Core v3.0 | Status: **ONLINE**")


# --- DOWLOAD & LOAD HAAR CASCADE ---
@st.cache_resource
def load_face_cascade():
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
            return None, "File cascade kosong."
        return cascade, None
    except Exception as e:
        return None, f"Error memuat Cascade: {e}"

face_cascade, cascade_error = load_face_cascade()


# --- FUNGSI MENGGAMBAR HELM IRON MAN (BGRA) ---
def buat_helm_ironman(lebar: int, tinggi: int) -> np.ndarray:
    """Menggambar Geometri Helm Iron Man dengan alpha channel transparansi."""
    kanvas = np.zeros((tinggi, lebar, 4), dtype=np.uint8)

    pusat_x, pusat_y = lebar // 2, int(tinggi * 0.48)
    sumbu_x, sumbu_y = int(lebar * 0.45), int(tinggi * 0.45)

    # Warna BGR + Alpha
    warna_merah = (20, 15, 180, 255)
    warna_emas = (40, 200, 240, 255)
    warna_mata = (255, 240, 200, 255)
    warna_garis = (10, 10, 100, 255)

    # 1. Tempurung Luar Merah
    cv2.ellipse(kanvas, (pusat_x, pusat_y), (sumbu_x, sumbu_y), 0, 0, 360, warna_merah, -1)

    # 2. Plat Topeng Emas
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

    # 3. Mata LED Menyala
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

    # 4. Garis Kontur & Garis Mulut
    cv2.polylines(kanvas, [titik_plat_emas], isClosed=True, color=warna_garis, thickness=max(2, lebar // 90))
    y_mulut = pusat_y + int(sumbu_y * 0.55)
    cv2.line(kanvas, (pusat_x - int(sumbu_x * 0.30), y_mulut), (pusat_x + int(sumbu_x * 0.30), y_mulut), warna_garis, thickness=max(2, lebar // 80))

    return kanvas


def tempel_overlay(frame_bgr, overlay_bgra, x, y):
    """Menempelkan gambar BGRA di atas gambar BGR menggunakan Alpha Blending."""
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
    hasil = warna_ov * alpha + roi * (1.0 - alpha)
    frame_bgr[y1:y2, x1:x2] = hasil.astype(np.uint8)
    return frame_bgr


def tempel_helm_otomatis(frame_bgr, face_cascade_obj):
    """Mendeteksi wajah dan menempelkan helm. Jika tidak terdeteksi, helm tetap ditempel di tengah."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hasil = frame_bgr.copy()
    h_frame, w_frame = frame_bgr.shape[:2]
    
    wajah_terdeteksi = []
    if face_cascade_obj is not None:
        # Parameter dibuat lebih sensitif agar wajah lebih mudah terdeteksi
        wajah_terdeteksi = face_cascade_obj.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(30, 30)
        )

    if len(wajah_terdeteksi) > 0:
        for (fx, fy, fw, fh) in wajah_terdeteksi:
            lebar_helm = int(fw * 1.8)
            tinggi_helm = int(fh * 2.1)
            helm = buat_helm_ironman(lebar_helm, tinggi_helm)

            pos_x = fx - int((lebar_helm - fw) / 2)
            pos_y = fy - int(tinggi_helm * 0.32)
            
            hasil = tempel_overlay(hasil, helm, pos_x, pos_y)
        status_msg = f"✅ Helm terdeteksi dan terpasang pada {len(wajah_terdeteksi)} wajah!"
    else:
        # Fallback: Jika wajah gagal terdeteksi, helm tetap ditempel di tengah foto
        lebar_helm = int(w_frame * 0.5)
        tinggi_helm = int(h_frame * 0.6)
        helm = buat_helm_ironman(lebar_helm, tinggi_helm)

        pos_x = (w_frame - lebar_helm) // 2
        pos_y = (h_frame - tinggi_helm) // 2
        hasil = tempel_overlay(hasil, helm, pos_x, pos_y)
        status_msg = "ℹ️ Wajah tidak terdeteksi otomatis. Helm ditayangkan di posisi tengah (Fallback Mode)."

    return hasil, status_msg


# --- LAYOUT DUA KOLOM ---
col_kamera, col_chat = st.columns([1.1, 0.9])

with col_kamera:
    st.subheader("📷 Visual Output")
    
    frame_bgr = None
    
    if mode_input == "Ambil Foto (Kamera)":
        camera_input = st.camera_input("Ambil snapshot foto")
        if camera_input:
            bytes_data = camera_input.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        uploaded_file = st.file_uploader("Upload Foto (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            bytes_data = uploaded_file.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
    if frame_bgr is not None:
        st.session_state.last_frame = frame_bgr
        
        if pasang_helm_ar:
            frame_hasil, status_msg = tempel_helm_otomatis(frame_bgr, face_cascade)
            st.info(status_msg)
        else:
            frame_hasil = frame_bgr

        # Tampilkan Foto Hasil
        st.image(cv2.cvtColor(frame_hasil, cv2.COLOR_BGR2RGB), caption="Hasil Helm Virtual Overlay", use_container_width=True)
    else:
        st.info("Silakan ambil foto lewat kamera atau unggah foto untuk melihat tampilan helm.")

with col_chat:
    st.subheader("💬 Tanya JARVIS AI")
    
    chat_box = st.container(height=380)
    with chat_box:
        for chat in st.session_state.chat_history:
            role = chat["role"]
            avatar = "🪖" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.write(chat["content"])

    user_query = st.chat_input("Tanya JARVIS tentang foto ini...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("JARVIS menganalisis foto..."):
            if st.session_state.last_frame is not None and api_key_input:
                try:
                    client = genai.Client(api_key=api_key_input)
                    _, buffer = cv2.imencode('.jpg', st.session_state.last_frame)
                    image_part = types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
                    
                    prompt = f"Kamu adalah JARVIS. Jawab singkat dalam Bahasa Indonesia: {user_query}"
                    res = client.models.generate_content(model=model_name, contents=[prompt, image_part])
                    jawaban_ai = res.text
                except Exception as e:
                    jawaban_ai = f"Error: {e}"
            else:
                jawaban_ai = "⚠️ Masukkan API Key Gemini di sidebar dan ambil/upload foto terlebih dahulu."
        
        st.session_state.chat_history.append({"role": "assistant", "content": jawaban_ai})
        st.rerun()
