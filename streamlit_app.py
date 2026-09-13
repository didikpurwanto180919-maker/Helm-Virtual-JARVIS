"""
H.E.L.M. Visor System - Virtual Helm AR Edition
===================================================
Aplikasi Streamlit untuk memasang Helm Virtual pada foto/kamera
dan menganalisis tampilan menggunakan Google Gemini AI.
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

st.title("🪖 H.E.L.M. Visor - Fitur Pasang Helm AR")
st.caption("Deteksi Wajah Otomatis & Penempelan Helm Virtual Futuristik")

# --- INISIALISASI SESSION STATE ---
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
    pasang_helm_ar = st.checkbox("🪖 Aktifkan Helm Virtual", value=True)
    
    st.markdown("---")
    st.write("H.E.L.M. AR Core v2.5 | Status: **ONLINE**")


# --- DETEKSI WAJAH (HAAR CASCADE) ---
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
            return None, "File cascade tidak valid."
        return cascade, None
    except Exception as e:
        return None, f"Error memuat OpenCV Cascade: {e}"

face_cascade, cascade_error = load_face_cascade()


# --- FUNGSI MENGGAMBAR HELM VIRTUAL (BGRA) ---
def buat_helm_ironman(lebar: int, tinggi: int) -> np.ndarray:
    """Menggambar Helm Virtual Futuristik warna Merah & Emas dengan Transparansi."""
    kanvas = np.zeros((tinggi, lebar, 4), dtype=np.uint8)

    pusat_x, pusat_y = lebar // 2, int(tinggi * 0.5)
    sumbu_x, sumbu_y = int(lebar * 0.44), int(tinggi * 0.46)

    warna_merah = (20, 15, 180, 255)       # Red Metallic (BGR)
    warna_emas = (40, 200, 240, 255)       # Gold Faceplate
    warna_mata = (255, 240, 200, 255)      # Cyan Glow
    warna_garis = (10, 10, 100, 255)       # Contour lines

    # 1. Tempurung Helm (Merah)
    cv2.ellipse(kanvas, (pusat_x, pusat_y), (sumbu_x, sumbu_y), 0, 0, 360, warna_merah, -1)

    # 2. Topeng Depan (Emas)
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

    # 4. Garis Panel & Detail Mulut
    cv2.polylines(kanvas, [titik_plat_emas], isClosed=True, color=warna_garis, thickness=max(2, lebar // 100))
    y_mulut = pusat_y + int(sumbu_y * 0.55)
    cv2.line(kanvas, (pusat_x - int(sumbu_x * 0.30), y_mulut), (pusat_x + int(sumbu_x * 0.30), y_mulut), warna_garis, thickness=max(2, lebar // 90))

    # Alpha Blur untuk Efek Halus
    alpha = kanvas[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    kanvas[:, :, 3] = (alpha * 255).astype(np.uint8)

    return kanvas


def tempel_overlay(frame_bgr, overlay_bgra, x, y):
    """Menempelkan gambar ber-alpha transparansi ke atas foto utama."""
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


def pasang_helm_ke_foto(frame_bgr, face_cascade_obj):
    """Mendeteksi posisi kepala di foto dan menempelkan Helm secara presisi."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hasil = frame_bgr.copy()
    
    wajah_terdeteksi = []
    if face_cascade_obj is not None:
        wajah_terdeteksi = face_cascade_obj.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (fx, fy, fw, fh) in wajah_terdeteksi:
        # Menyesuaikan ukuran & posisi helm agar pas menutup kepala
        lebar_helm = int(fw * 1.65)
        tinggi_helm = int(fh * 1.95)
        helm = buat_helm_ironman(lebar_helm, tinggi_helm)

        pos_x = fx - int((lebar_helm - fw) / 2)
        pos_y = fy - int(tinggi_helm * 0.28)
        
        hasil = tempel_overlay(hasil, helm, pos_x, pos_y)

    return hasil, len(wajah_terdeteksi)


# --- LAYOUT DUA KOLOM ---
col_kamera, col_chat = st.columns([1.1, 0.9])

with col_kamera:
    st.subheader("📷 Tampilan Foto Berhelm")
    
    frame_bgr = None
    
    if mode_input == "Ambil Foto (Kamera)":
        camera_input = st.camera_input("Ambil snapshot wajah")
        if camera_input:
            bytes_data = camera_input.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        uploaded_file = st.file_uploader("Pilih file foto (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            bytes_data = uploaded_file.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
    if frame_bgr is not None:
        st.session_state.last_frame = frame_bgr
        
        if pasang_helm_ar:
            frame_hasil, jumlah_wajah = pasang_helm_ke_foto(frame_bgr, face_cascade)
            if jumlah_wajah > 0:
                st.success(f"✅ Helm Virtual berhasil dipasang pada {jumlah_wajah} wajah!")
            else:
                st.warning("⚠️ Wajah tidak terdeteksi. Pastikan foto wajah terlihat jelas dan menghadap ke depan.")
        else:
            frame_hasil = frame_bgr

        # Tampilkan Foto Hasil
        st.image(cv2.cvtColor(frame_hasil, cv2.COLOR_BGR2RGB), caption="Hasil Penempelan Helm Virtual", use_container_width=True)
    else:
        st.info("Silakan ambil foto lewat kamera atau upload foto untuk melihat tampilan berhelm.")

with col_chat:
    st.subheader("💬 Tanya JARVIS AI")
    
    chat_box = st.container(height=380)
    with chat_box:
        for chat in st.session_state.chat_history:
            role = chat["role"]
            avatar = "🪖" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.write(chat["content"])

    user_query = st.chat_input("Tanya JARVIS tentang foto di sebelah...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("JARVIS menganalisis foto..."):
            if st.session_state.last_frame is not None and api_key_input:
                try:
                    client = genai.Client(api_key=api_key_input)
                    _, buffer = cv2.imencode('.jpg', st.session_state.last_frame)
                    image_part = types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
                    
                    prompt = f"Kamu adalah JARVIS asisten virtual. Jawab pertanyaan singkat dalam Bahasa Indonesia: {user_query}"
                    res = client.models.generate_content(model=model_name, contents=[prompt, image_part])
                    jawaban_ai = res.text
                except Exception as e:
                    jawaban_ai = f"Error: {e}"
            else:
                jawaban_ai = "⚠️ Masukkan API Key Gemini di sidebar dan sediakan foto terlebih dahulu."
        
        st.session_state.chat_history.append({"role": "assistant", "content": jawaban_ai})
        st.rerun()
