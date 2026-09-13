import os
import io
import base64
import urllib.request
import cv2
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

st.set_page_config(
    page_title="H.E.L.M. Visor System",
    page_icon="🪖",
    layout="wide"
)

# --- FUNGSI HELPER TTS (JARVIS BERBICARA) ---
def putar_audio_gtts(teks, lang='id'):
    try:
        tts = gTTS(text=teks, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
            <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal memproses suara: {e}")

# --- SIDEBAR KONTROL MANUAL & AI ---
with st.sidebar:
    st.header("⚙️ Pengaturan System")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", "")
    )
    model_name = st.selectbox(
        "Model Gemini AI",
        ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    
    st.markdown("---")
    st.header("🔊 Pengaturan Suara JARVIS")
    suara_aktif = st.checkbox("Aktifkan Respon Suara", value=True)
    bahasa_suara = st.selectbox("Bahasa Suara", ["Bahasa Indonesia (id)", "English (en)"], index=0)
    lang_code = "id" if "Indonesia" in bahasa_suara else "en"

    st.markdown("---")
    st.header("🎯 Penyesuaian Manual Helm")
    aktifkan_helm = st.checkbox("🪖 Tampilkan Helm Virtual", value=True)
    offset_x = st.slider("Geser Kiri / Kanan (X)", -300, 300, 0, step=5)
    offset_y = st.slider("Geser Atas / Bawah (Y)", -300, 300, -20, step=5)
    skala_helm = st.slider("Skala Ukuran Helm", 0.5, 3.0, 1.2, step=0.1)

# --- LOAD HAAR CASCADE ---
@st.cache_resource
def load_cascade():
    xml_filename = "haarcascade_frontalface_alt.xml"
    if not os.path.exists(xml_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"
        try:
            urllib.request.urlretrieve(url, xml_filename)
        except Exception:
            pass
    if os.path.exists(xml_filename):
        return cv2.CascadeClassifier(xml_filename)
    return None

face_cascade = load_cascade()

# --- FUNGSI HELM IRON MAN ---
def buat_helm_ironman(lebar, tinggi):
    kanvas = np.zeros((tinggi, lebar, 4), dtype=np.uint8)
    pusat_x, pusat_y = lebar // 2, int(tinggi * 0.48)
    sumbu_x, sumbu_y = int(lebar * 0.45), int(tinggi * 0.45)

    warna_merah = (20, 15, 180, 255)
    warna_emas = (40, 200, 240, 255)
    warna_mata = (255, 240, 200, 255)
    warna_garis = (10, 10, 100, 255)

    cv2.ellipse(kanvas, (pusat_x, pusat_y), (sumbu_x, sumbu_y), 0, 0, 360, warna_merah, -1)

    titik_emas = np.array([
        [pusat_x - int(sumbu_x * 0.65), pusat_y - int(sumbu_y * 0.60)],
        [pusat_x + int(sumbu_x * 0.65), pusat_y - int(sumbu_y * 0.60)],
        [pusat_x + int(sumbu_x * 0.75), pusat_y - int(sumbu_y * 0.10)],
        [pusat_x + int(sumbu_x * 0.55), pusat_y + int(sumbu_y * 0.45)],
        [pusat_x + int(sumbu_x * 0.35), pusat_y + int(sumbu_y * 0.85)],
        [pusat_x - int(sumbu_x * 0.35), pusat_y + int(sumbu_y * 0.85)],
        [pusat_x - int(sumbu_x * 0.55), pusat_y + int(sumbu_y * 0.45)],
        [pusat_x - int(sumbu_x * 0.75), pusat_y - int(sumbu_y * 0.10)],
    ], dtype=np.int32)
    cv2.fillPoly(kanvas, [titik_emas], warna_emas)

    tinggi_mata = max(4, int(sumbu_y * 0.08))
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
    cv2.polylines(kanvas, [titik_emas], True, warna_garis, max(2, lebar // 90))

    return kanvas

def tempel_overlay(frame, overlay, x, y):
    h_ov, w_ov = overlay.shape[:2]
    h_f, w_f = frame.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_ov, w_f), min(y + h_ov, h_f)
    if x1 >= x2 or y1 >= y2:
        return frame

    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    alpha = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3:4] / 255.0
    warna = overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3]
    roi = frame[y1:y2, x1:x2]

    frame[y1:y2, x1:x2] = (warna * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return frame

st.title("🪖 H.E.L.M. Visor System - Virtual Overlay")

col_kamera, col_chat = st.columns([1.1, 0.9])

with col_kamera:
    camera_input = st.camera_input("Ambil Snapshot")

    if camera_input:
        frame = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        h_img, w_img, _ = frame.shape

        if aktifkan_helm:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            wajah = []
            if face_cascade is not None:
                wajah = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

            if len(wajah) > 0:
                fx, fy, fw, fh = wajah[0]
                lebar_h = int(fw * 1.8 * skala_helm)
                tinggi_h = int(fh * 2.2 * skala_helm)
                helm = buat_helm_ironman(lebar_h, tinggi_h)

                px = fx - int((lebar_h - fw) / 2) + offset_x
                py = fy - int(tinggi_h * 0.35) + offset_y
                frame = tempel_overlay(frame, helm, px, py)
            else:
                lebar_h = int(w_img * 0.45 * skala_helm)
                tinggi_h = int(h_img * 0.55 * skala_helm)
                helm = buat_helm_ironman(lebar_h, tinggi_h)

                px = ((w_img - lebar_h) // 2) + offset_x
                py = ((h_img - tinggi_h) // 2) + offset_y
                frame = tempel_overlay(frame, helm, px, py)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Visual Output", use_container_width=True)

with col_chat:
    st.subheader("💬 JARVIS Voice Assistant")
    
    st.markdown("##### 🎙️ Rekam Perintah Suara:")
    audio_record = mic_recorder(
        start_prompt="🔴 Mulai Bicara",
        stop_prompt="⏹️ Selesai / Kirim",
        key="jarvis_mic"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = None
    input_text = st.chat_input("Atau ketik pesan untuk JARVIS...")

    if input_text:
        user_query = input_text
    elif audio_record and "bytes" in audio_record:
        user_query = "PERINTAH_AUDIO"

    if user_query:
        if api_key_input:
            try:
                client = genai.Client(api_key=api_key_input)
                system_instruction = "Kamu adalah JARVIS, asisten AI Iron Man yang sangat sopan, cerdas, efisien, dan selalu menjawab secara singkat dan langsung dalam bahasa yang sama dengan input pengguna."

                with st.chat_message("user"):
                    if user_query == "PERINTAH_AUDIO":
                        st.audio(audio_record["bytes"], format="audio/wav")
                        st.caption("🎙️ Perintah suara terkirim")
                    else:
                        st.markdown(user_query)

                st.session_state.messages.append({
                    "role": "user", 
                    "content": "🎙️ [Perintah Suara]" if user_query == "PERINTAH_AUDIO" else user_query
                })

                # Konfigurasi Panggilan SDK Gemini
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )

                if user_query == "PERINTAH_AUDIO":
                    # Format audio byte menggunakan types.Part.from_bytes
                    audio_part = types.Part.from_bytes(
                        data=audio_record["bytes"],
                        mime_type="audio/wav"
                    )
                    res = client.models.generate_content(
                        model=model_name,
                        contents=[audio_part],
                        config=config
                    )
                else:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=user_query,
                        config=config
                    )

                jawaban = res.text

                with st.chat_message("assistant"):
                    st.markdown(jawaban)
                    if suara_aktif:
                        putar_audio_gtts(jawaban, lang=lang_code)

                st.session_state.messages.append({"role": "assistant", "content": jawaban})

            except Exception as e:
                st.error(f"Error AI: {e}")
        else:
            st.warning("Silakan masukkan API Key Gemini di sidebar terlebih dahulu.")
