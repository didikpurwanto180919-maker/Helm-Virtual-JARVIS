"""
Helm Virtual JARVIS
====================
Aplikasi Streamlit: asisten virtual berbasis kamera + AI (Google Gemini)
untuk membantu pengendara — mendeteksi kondisi sekitar lewat kamera helm
dan menjawab pertanyaan pengguna secara real-time.

Fitur:
- Upload gambar / snapshot kamera / mode LIVE (real-time streaming) via WebRTC
- Auto-analisis: JARVIS otomatis menganalisis begitu ada gambar/frame baru
- Riwayat analisis tersimpan ke file (bertahan selama container berjalan)
  + bisa diunduh sebagai file .json

Dependencies (requirements.txt):
    streamlit
    opencv-python-headless
    numpy
    requests
    google-genai
    streamlit-webrtc
    av
"""

import os
import io
import json
import time
import threading

import numpy as np
import cv2
import requests
import streamlit as st
from google import genai
from google.genai import types

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    WEBRTC_TERSEDIA = True
except ImportError:
    WEBRTC_TERSEDIA = False


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
# PENYIMPANAN RIWAYAT (persisten selama container berjalan)
# ============================================================
FILE_RIWAYAT = os.path.join(os.path.dirname(__file__), "riwayat_analisis.json")


def muat_riwayat_dari_disk():
    if os.path.exists(FILE_RIWAYAT):
        try:
            with open(FILE_RIWAYAT, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def simpan_riwayat_ke_disk(riwayat):
    try:
        with open(FILE_RIWAYAT, "w", encoding="utf-8") as f:
            json.dump(riwayat, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan riwayat ke disk: {e}")


def tambah_ke_riwayat(role, text):
    entri = {
        "role": role,
        "text": text,
        "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.riwayat_chat.append(entri)
    simpan_riwayat_ke_disk(st.session_state.riwayat_chat)


# ============================================================
# SETUP GEMINI CLIENT
# ============================================================
def get_client(manual_api_key=None):
    api_key = manual_api_key or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error(
            "⚠️ GEMINI_API_KEY belum diatur. Masukkan API Key di sidebar, "
            "atau atur melalui Streamlit Secrets / Environment Variable."
        )
        st.stop()
    return genai.Client(api_key=api_key)


# ============================================================
# STATE AWAL
# ============================================================
if "riwayat_chat" not in st.session_state:
    st.session_state.riwayat_chat = muat_riwayat_dari_disk()

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "hash_gambar_terakhir" not in st.session_state:
    st.session_state.hash_gambar_terakhir = None

if "waktu_auto_terakhir" not in st.session_state:
    st.session_state.waktu_auto_terakhir = 0.0


# ============================================================
# SIDEBAR - PENGATURAN
# ============================================================
opsi_mode = ["Upload Gambar", "Ambil dari Kamera (snapshot)"]
if WEBRTC_TERSEDIA:
    opsi_mode.append("LIVE (Real-time Streaming)")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Input opsional jika API Key belum dipasang di Secrets
    api_key_input = st.text_input(
        "Gemini API Key (Opsional)", 
        type="password", 
        help="Kosongkan jika sudah diatur di Streamlit Secrets"
    )
    
    mode = st.radio("Mode Input Kamera", opsi_mode, index=0)

    # Menggunakan nama model Gemini resmi yang stabil
    model_name = st.selectbox(
        "Model Gemini",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    st.markdown("---")
    pasang_helm = st.checkbox(
        "🪖 Pasang Model Helm Virtual (AR)",
        value=False,
        help="Mendeteksi wajah dan menampilkan model helm virtual futuristik di atasnya (desain orisinal, real-time).",
    )

    st.markdown("---")
    auto_analisis = st.checkbox(
        "🤖 Auto-analisis (tanpa klik tombol)",
        value=False,
        help="Jika aktif, JARVIS otomatis menganalisis begitu ada gambar/frame baru.",
    )
    interval_auto = 5
    if mode == "LIVE (Real-time Streaming)" and auto_analisis:
        interval_auto = st.slider(
            "Interval auto-analisis (detik)", min_value=3, max_value=30, value=5
        )

    st.markdown("---")
    st.write(f"**Riwayat tersimpan:** {len(st.session_state.riwayat_chat)} entri")
    if st.session_state.riwayat_chat:
        st.download_button(
            "⬇️ Unduh Riwayat (.json)",
            data=json.dumps(st.session_state.riwayat_chat, ensure_ascii=False, indent=2),
            file_name="riwayat_analisis_jarvis.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.write("**Tentang**")
    st.write(
        "Helm Virtual JARVIS menganalisis gambar dari kamera helm "
        "(jalan, kondisi lalu lintas, potensi bahaya) dan menjawab "
        "pertanyaan pengendara secara langsung."
    )
    if not WEBRTC_TERSEDIA:
        st.caption(
            "ℹ️ Mode LIVE tidak tersedia karena library `streamlit-webrtc` "
            "belum terpasang. Tambahkan ke requirements.txt untuk mengaktifkannya."
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
    """Deteksi tepi (edge detection) untuk menyorot marka jalan / objek di sekitar."""
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


def hash_gambar(image_bytes: bytes) -> str:
    """Hash sederhana untuk mendeteksi apakah gambar berubah dari sebelumnya."""
    import hashlib
    return hashlib.md5(image_bytes).hexdigest()


# ============================================================
# MODEL HELM VIRTUAL (AR) — desain orisinal, digambar lewat kode
# ============================================================
@st.cache_resource
def muat_deteksi_wajah():
    """Muat model deteksi wajah bawaan OpenCV (Haar Cascade)."""
    path_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path_cascade)


def buat_model_helm_virtual(lebar: int, tinggi: int) -> np.ndarray:
    """Menggambar model helm virtual futuristik (BGRA, dengan transparansi) secara prosedural."""
    kanvas = np.zeros((tinggi, lebar, 4), dtype=np.uint8)

    pusat_x, pusat_y = lebar // 2, int(tinggi * 0.48)
    sumbu_x, sumbu_y = int(lebar * 0.46), int(tinggi * 0.46)

    warna_utama = (30, 30, 200, 255)      # merah metalik (BGR + alpha)
    warna_aksen = (40, 170, 235, 255)     # emas/oranye
    warna_visor = (235, 200, 40, 255)     # cyan-biru menyala
    warna_garis = (60, 60, 60, 255)       # abu gelap untuk garis panel

    # Bentuk dasar helm
    cv2.ellipse(kanvas, (pusat_x, pusat_y), (sumbu_x, sumbu_y), 0, 0, 360, warna_utama, -1)

    # Panel aksen dahi
    titik_dahi = np.array([
        [pusat_x - int(sumbu_x * 0.55), pusat_y - int(sumbu_y * 0.55)],
        [pusat_x + int(sumbu_x * 0.55), pusat_y - int(sumbu_y * 0.55)],
        [pusat_x + int(sumbu_x * 0.30), pusat_y - int(sumbu_y * 0.85)],
        [pusat_x - int(sumbu_x * 0.30), pusat_y - int(sumbu_y * 0.85)],
    ], dtype=np.int32)
    cv2.fillPoly(kanvas, [titik_dahi], warna_aksen)

    # Visor
    cv2.ellipse(
        kanvas, (pusat_x, int(pusat_y - sumbu_y * 0.05)),
        (int(sumbu_x * 0.62), int(sumbu_y * 0.22)),
        0, 200, 340, warna_visor, thickness=max(3, lebar // 40),
    )

    # Panel samping
    for dx in (-1, 1):
        titik_awal = (pusat_x + dx * int(sumbu_x * 0.75), pusat_y)
        titik_akhir = (pusat_x + dx * int(sumbu_x * 0.95), pusat_y + int(sumbu_y * 0.3))
        cv2.line(kanvas, titik_awal, titik_akhir, warna_garis, thickness=max(2, lebar // 80))

    # Pelindung dagu
    titik_dagu = np.array([
        [pusat_x - int(sumbu_x * 0.45), pusat_y + int(sumbu_y * 0.55)],
        [pusat_x + int(sumbu_x * 0.45), pusat_y + int(sumbu_y * 0.55)],
        [pusat_x, pusat_y + int(sumbu_y * 0.98)],
    ], dtype=np.int32)
    cv2.fillPoly(kanvas, [titik_dagu], warna_aksen)

    # Alpha anti-aliasing
    alpha = kanvas[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    kanvas[:, :, 3] = (alpha * 255).astype(np.uint8)

    return kanvas


def tempelkan_overlay_bgra(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-blend gambar BGRA ke atas frame_bgr."""
    h_ov, w_ov = overlay_bgra.shape[:2]
    h_frame, w_frame = frame_bgr.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_ov, w_frame), min(y + h_ov, h_frame)
    if x1 >= x2 or y1 >= y2:
        return frame_bgr

    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    bagian_overlay = overlay_bgra[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha = bagian_overlay[:, :, 3:4].astype(np.float32) / 255.0
    warna_overlay = bagian_overlay[:, :, :3].astype(np.float32)

    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    hasil = warna_overlay * alpha + roi * (1 - alpha)
    frame_bgr[y1:y2, x1:x2] = hasil.astype(np.uint8)
    return frame_bgr


def pasang_helm_virtual_ke_wajah(frame_bgr: np.ndarray) -> np.ndarray:
    """Deteksi wajah pada frame, lalu tempelkan model helm virtual di atasnya."""
    cascade = muat_deteksi_wajah()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    wajah_terdeteksi = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    hasil = frame_bgr.copy()
    for (fx, fy, fw, fh) in wajah_terdeteksi:
        lebar_helm = int(fw * 1.5)
        tinggi_helm = int(fh * 1.9)
        helm = buat_model_helm_virtual(lebar_helm, tinggi_helm)

        pos_x = fx - int((lebar_helm - fw) / 2)
        pos_y = fy - int(tinggi_helm * 0.28)

        hasil = tempelkan_overlay_bgra(hasil, helm, pos_x, pos_y)

    return hasil


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
    """Mode tanya-jawab teks biasa tanpa gambar."""
    prompt = (
        "Kamu adalah JARVIS, asisten virtual pribadi untuk pengendara motor. "
        "Jawab pertanyaan berikut secara singkat dan jelas dalam Bahasa Indonesia.\n\n"
        f"Pertanyaan: {pertanyaan}"
    )
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text


def jalankan_analisis(image_bytes: bytes, label_user: str = "[Analisis gambar kamera]"):
    """Helper terpusat: proses gambar, panggil AI, simpan ke riwayat."""
    client = get_client(api_key_input)
    frame = proses_gambar_opencv(image_bytes)
    st.session_state.last_frame = frame
    jpeg_bytes = frame_ke_bytes(frame)
    hasil = analisis_gambar_dengan_ai(client, jpeg_bytes, "", model_name)
    tambah_ke_riwayat("user", label_user)
    tambah_ke_riwayat("jarvis", hasil)


# ============================================================
# WEBRTC: PENYIMPANAN FRAME TERBARU
# ============================================================
class PenampungFrame:
    def __init__(self):
        self.frame = None
        self.pasang_helm = False
        self.lock = threading.Lock()

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def set_pasang_helm(self, aktif: bool):
        with self.lock:
            self.pasang_helm = aktif

    def get_pasang_helm(self) -> bool:
        with self.lock:
            return self.pasang_helm


if "penampung_frame" not in st.session_state:
    st.session_state.penampung_frame = PenampungFrame()

penampung = st.session_state.penampung_frame
penampung.set_pasang_helm(pasang_helm)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    penampung.set_frame(img)

    if penampung.get_pasang_helm():
        img_tampil = pasang_helm_virtual_ke_wajah(img)
    else:
        img_tampil = img

    return av.VideoFrame.from_ndarray(img_tampil, format="bgr24")


# ============================================================
# LAYOUT UTAMA: DUA KOLOM
# ============================================================
kolom_kamera, kolom_chat = st.columns([1, 1])

image_bytes = None
gambar_baru_terdeteksi = False

with kolom_kamera:
    st.subheader("📷 Kamera Helm")

    if mode == "Upload Gambar":
        file_upload = st.file_uploader(
            "Upload snapshot dari kamera helm", type=["jpg", "jpeg", "png"]
        )
        if file_upload is not None:
            image_bytes = file_upload.read()

    elif mode == "Ambil dari Kamera (snapshot)":
        camera_input = st.camera_input("Ambil gambar dari kamera")
        if camera_input is not None:
            image_bytes = camera_input.getvalue()

    else:  # LIVE (Real-time Streaming)
        st.caption(
            "🔴 Mode LIVE — video langsung dari kamera. "
            "Jika video tidak muncul, coba refresh atau gunakan mode Upload/Snapshot."
        )
        webrtc_streamer(
            key="jarvis-live",
            mode=WebRtcMode.SENDONLY,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )
        frame_live = penampung.get_frame()
        if frame_live is not None:
            st.session_state.last_frame = frame_live
            image_bytes = frame_ke_bytes(frame_live)
            st.caption(
                "Preview di atas sudah menampilkan model helm virtual secara langsung "
                "jika opsi diaktifkan. Analisis AI tetap memakai gambar asli (tanpa overlay)."
                if pasang_helm else
                "Frame terakhir dari LIVE stream."
            )

    # Tampilkan gambar (untuk mode Upload / Snapshot)
    if mode != "LIVE (Real-time Streaming)":
        if image_bytes:
            frame = proses_gambar_opencv(image_bytes)
            st.session_state.last_frame = frame

            tampilkan_edge = st.checkbox("Tampilkan mode deteksi tepi (edge detection)")
            if tampilkan_edge:
                edges = deteksi_tepi(frame)
                st.image(edges, caption="Hasil Deteksi Tepi", use_container_width=True)
            else:
                frame_tampil = frame
                caption = "Gambar dari kamera helm"
                if pasang_helm:
                    frame_tampil = pasang_helm_virtual_ke_wajah(frame)
                    caption = "Model Helm Virtual (AR) terpasang"
                st.image(
                    cv2.cvtColor(frame_tampil, cv2.COLOR_BGR2RGB),
                    caption=caption,
                    use_container_width=True,
                )
        else:
            st.info("Belum ada gambar. Upload atau ambil snapshot dari kamera dulu.")

    # Deteksi gambar BARU
    if image_bytes:
        h = hash_gambar(image_bytes)
        if h != st.session_state.hash_gambar_terakhir:
            st.session_state.hash_gambar_terakhir = h
            gambar_baru_terdeteksi = True

    st.markdown("---")

    tombol_manual = st.button(
        "🔍 Analisis Gambar Sekarang", use_container_width=True, disabled=image_bytes is None
    )

    perlu_analisis = False
    if tombol_manual and image_bytes:
        perlu_analisis = True
    elif auto_analisis and image_bytes:
        if mode == "LIVE (Real-time Streaming)":
            sekarang = time.time()
            if sekarang - st.session_state.waktu_auto_terakhir >= interval_auto:
                perlu_analisis = True
                st.session_state.waktu_auto_terakhir = sekarang
        elif gambar_baru_terdeteksi:
            perlu_analisis = True

    if perlu_analisis:
        with st.spinner("JARVIS sedang menganalisis gambar..."):
            try:
                jalankan_analisis(image_bytes)
            except Exception as e:
                st.error(f"Gagal menganalisis gambar: {e}")
        st.rerun()

    if mode == "LIVE (Real-time Streaming)" and auto_analisis:
        time.sleep(1)
        st.rerun()


with kolom_chat:
    st.subheader("💬 Tanya JARVIS")

    chat_container = st.container(height=400)
    with chat_container:
        for pesan in st.session_state.riwayat_chat:
            waktu = pesan.get("waktu", "")
            if pesan["role"] == "user":
                st.chat_message("user").write(f"{pesan['text']}  \n:gray[{waktu}]")
            else:
                st.chat_message("assistant", avatar="🪖").write(f"{pesan['text']}  \n:gray[{waktu}]")

    pertanyaan = st.chat_input("Tanya sesuatu ke JARVIS, atau tanya soal gambar di kiri...")

    if pertanyaan:
        client = get_client(api_key_input)
        tambah_ke_riwayat("user", pertanyaan)

        with st.spinner("JARVIS sedang berpikir..."):
            if st.session_state.last_frame is not None:
                jpeg_bytes = frame_ke_bytes(st.session_state.last_frame)
                jawaban = analisis_gambar_dengan_ai(client, jpeg_bytes, pertanyaan, model_name)
            else:
                jawaban = tanya_jawab_teks(client, pertanyaan, model_name)

        tambah_ke_riwayat("jarvis", jawaban)
        st.rerun()

    kolom_hapus, kolom_unduh = st.columns(2)
    with kolom_hapus:
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
            st.session_state.riwayat_chat = []
            simpan_riwayat_ke_disk([])
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Helm Virtual JARVIS © 2026 — Dibuat dengan Streamlit, OpenCV, dan Google Gemini API."
)
