"""
Helm Virtual JARVIS
====================
Aplikasi Streamlit: asisten virtual berbasis kamera + AI (Google Gemini)
untuk membantu pengendara — mendeteksi kondisi sekitar lewat kamera helm
dan menjawab pertanyaan pengguna secara real-time.

Dependencies (requirements.txt):
    streamlit
    opencv-python-headless
    numpy
    requests
    google-genai

Cara menjalankan lokal:
    streamlit run streamlit_app.py

Environment variable yang dibutuhkan:
    GEMINI_API_KEY  -> API key dari Google AI Studio (https://aistudio.google.com/apikey)
    (Di Streamlit Cloud, set ini lewat menu "Secrets": GEMINI_API_KEY = "xxxx")
"""

import os
import io
import time
import base64

import numpy as np
import cv2
import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Helm Virtual JARVIS",
    page_icon="🪖",
    layout="wide",
)

st.title("🪖 Helm Virtual JARVIS")
st.caption("Asisten virtual pintar untuk pengendara — analisis kamera & tanya-jawab AI")


# ============================================================
# SETUP GEMINI CLIENT
# ============================================================
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error(
            "GEMINI_API_KEY belum diatur. Tambahkan di Streamlit Secrets "
            "atau environment variable sebelum menjalankan aplikasi."
        )
        st.stop()
    return genai.Client(api_key=api_key)


# ============================================================
# STATE AWAL
# ============================================================
if "riwayat_chat" not in st.session_state:
    st.session_state.riwayat_chat = []  # list of {"role": ..., "text": ...}

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None


# ============================================================
# SIDEBAR - PENGATURAN
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio(
        "Mode Input Kamera",
        ["Upload Gambar", "Ambil dari Kamera (snapshot)"],
        index=0,
    )
    model_name = st.selectbox(
        "Model Gemini",
        ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview"],
        index=0,
    )
    st.markdown("---")
    st.write("**Tentang**")
    st.write(
        "Helm Virtual JARVIS menganalisis gambar dari kamera helm "
        "(jalan, kondisi lalu lintas, potensi bahaya) dan menjawab "
        "pertanyaan pengendara secara langsung."
    )


# ============================================================
# FUNGSI BANTUAN: OLAH GAMBAR DENGAN OPENCV
# ============================================================
def proses_gambar_opencv(image_bytes: bytes) -> np.ndarray:
    """Decode bytes gambar menjadi array OpenCV (BGR)."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def deteksi_tepi(frame: np.ndarray) -> np.ndarray:
    """Contoh pemrosesan sederhana: deteksi tepi (edge detection) untuk
    menyorot marka jalan / objek di sekitar."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def frame_ke_bytes(frame: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode array OpenCV (BGR) kembali menjadi bytes gambar."""
    ok, buf = cv2.imencode(ext, frame)
    if not ok:
        raise ValueError("Gagal meng-encode gambar")
    return buf.tobytes()


# ============================================================
# FUNGSI BANTUAN: PANGGIL GEMINI UNTUK ANALISIS GAMBAR
# ============================================================
def analisis_gambar_dengan_ai(client, image_bytes: bytes, pertanyaan: str, model: str) -> str:
    """Kirim gambar + pertanyaan ke Gemini, kembalikan jawaban teks."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    prompt = (
        "Kamu adalah JARVIS, asisten virtual di helm pengendara motor. "
        "Analisis gambar dari kamera helm berikut ini, lalu jawab pertanyaan "
        "pengendara dengan singkat, jelas, dan dalam Bahasa Indonesia. "
        "Jika ada potensi bahaya di jalan (kendaraan mendekat, jalan berlubang, "
        "penyeberang jalan, dll), sebutkan sebagai peringatan di awal jawaban.\n\n"
        f"Pertanyaan pengendara: {pertanyaan if pertanyaan else 'Apa yang terlihat di depan? Ada bahaya tidak?'}"
    )

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
    )
    return response.text


def tanya_jawab_teks(client, pertanyaan: str, model: str) -> str:
    """Mode tanya-jawab teks biasa tanpa gambar (misal: 'jam berapa sebaiknya istirahat')."""
    prompt = (
        "Kamu adalah JARVIS, asisten virtual pribadi untuk pengendara motor. "
        "Jawab pertanyaan berikut secara singkat dan jelas dalam Bahasa Indonesia.\n\n"
        f"Pertanyaan: {pertanyaan}"
    )
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text


# ============================================================
# LAYOUT UTAMA: DUA KOLOM
# ============================================================
kolom_kamera, kolom_chat = st.columns([1, 1])

image_bytes = None

with kolom_kamera:
    st.subheader("📷 Kamera Helm")

    if mode == "Upload Gambar":
        file_upload = st.file_uploader(
            "Upload snapshot dari kamera helm", type=["jpg", "jpeg", "png"]
        )
        if file_upload is not None:
            image_bytes = file_upload.read()

    else:  # Ambil dari Kamera (snapshot)
        camera_input = st.camera_input("Ambil gambar dari kamera")
        if camera_input is not None:
            image_bytes = camera_input.getvalue()

    if image_bytes:
        frame = proses_gambar_opencv(image_bytes)
        st.session_state.last_frame = frame

        tampilkan_edge = st.checkbox("Tampilkan mode deteksi tepi (edge detection)")
        if tampilkan_edge:
            edges = deteksi_tepi(frame)
            st.image(edges, caption="Hasil Deteksi Tepi", use_container_width=True)
        else:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Gambar dari kamera helm",
                use_container_width=True,
            )
    else:
        st.info("Belum ada gambar. Upload atau ambil snapshot dari kamera dulu.")

    st.markdown("---")
    if st.button("🔍 Analisis Gambar Sekarang", use_container_width=True, disabled=image_bytes is None):
        client = get_client()
        with st.spinner("JARVIS sedang menganalisis gambar..."):
            jpeg_bytes = frame_ke_bytes(st.session_state.last_frame)
            hasil = analisis_gambar_dengan_ai(client, jpeg_bytes, "", model_name)
        st.session_state.riwayat_chat.append({"role": "user", "text": "[Analisis gambar kamera]"})
        st.session_state.riwayat_chat.append({"role": "jarvis", "text": hasil})
        st.rerun()


with kolom_chat:
    st.subheader("💬 Tanya JARVIS")

    # Tampilkan riwayat chat
    chat_container = st.container(height=400)
    with chat_container:
        for pesan in st.session_state.riwayat_chat:
            if pesan["role"] == "user":
                st.chat_message("user").write(pesan["text"])
            else:
                st.chat_message("assistant", avatar="🪖").write(pesan["text"])

    pertanyaan = st.chat_input("Tanya sesuatu ke JARVIS, atau tanya soal gambar di kiri...")

    if pertanyaan:
        client = get_client()
        st.session_state.riwayat_chat.append({"role": "user", "text": pertanyaan})

        with st.spinner("JARVIS sedang berpikir..."):
            if st.session_state.last_frame is not None:
                jpeg_bytes = frame_ke_bytes(st.session_state.last_frame)
                jawaban = analisis_gambar_dengan_ai(client, jpeg_bytes, pertanyaan, model_name)
            else:
                jawaban = tanya_jawab_teks(client, pertanyaan, model_name)

        st.session_state.riwayat_chat.append({"role": "jarvis", "text": jawaban})
        st.rerun()

    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.riwayat_chat = []
        st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Helm Virtual JARVIS © 2026 — Dibuat dengan Streamlit, OpenCV, dan Google Gemini API."
)
