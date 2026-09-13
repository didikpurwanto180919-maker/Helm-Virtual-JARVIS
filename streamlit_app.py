import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

st.set_page_config(page_title="H.E.L.M. Visor System", layout="wide")

# --- SIDEBAR KONTROL MANUAL (FALLBACK) ---
with st.sidebar:
    st.header("⚙️ Penyesuaian Manual Helm")
    offset_x = st.slider("Geser Kiri/Kanan (X)", -200, 200, 0)
    offset_y = st.slider("Geser Atas/Bawah (Y)", -200, 200, 0)
    skala_helm = st.slider("Ukuran Helm", 0.5, 2.5, 1.0, 0.1)

def buat_helm_ironman(lebar, tinggi):
    """Membuat grafik helm Iron Man BGRA."""
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

st.title("🪖 H.E.L.M. Visor System")
camera_input = st.camera_input("Ambil Foto")

if camera_input:
    bytes_data = camera_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    h_img, w_img, _ = frame.shape

    # Gunakan MediaPipe untuk deteksi posisi wajah
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                fx = int(bboxC.xmin * w_img)
                fy = int(bboxC.ymin * h_img)
                fw = int(bboxC.width * w_img)
                fh = int(bboxC.height * h_img)

                lebar_h = int(fw * 1.8 * skala_helm)
                tinggi_h = int(fh * 2.2 * skala_helm)
                helm = buat_helm_ironman(lebar_h, tinggi_h)

                px = fx - int((lebar_h - fw) / 2) + offset_x
                py = fy - int(tinggi_h * 0.35) + offset_y
                frame = tempel_overlay(frame, helm, px, py)
            st.success("✅ Wajah terdeteksi dan helm berhasil dipasang!")
        else:
            # Mode Cadangan: Jika AI gagal mendeteksi, pasang helm di tengah foto
            lebar_h = int(w_img * 0.45 * skala_helm)
            tinggi_h = int(h_img * 0.55 * skala_helm)
            helm = buat_helm_ironman(lebar_h, tinggi_h)

            px = ((w_img - lebar_h) // 2) + offset_x
            py = ((h_img - tinggi_h) // 2) + offset_y
            frame = tempel_overlay(frame, helm, px, py)
            st.warning("⚠️ Wajah tidak terdeteksi otomatis. Gunakan slider di sidebar untuk menyesuaikan posisi helm.")

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Hasil Penempelan Helm Virtual", use_container_width=True)
