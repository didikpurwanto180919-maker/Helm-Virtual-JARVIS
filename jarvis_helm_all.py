"""
JARVIS - Asisten Virtual Sederhana (Python)
=============================================

Fitur:
- Mendengarkan perintah suara (Speech-to-Text via microphone)
- Menjawab dengan suara (Text-to-Speech)
- Bisa juga dipakai lewat mode teks (ketik perintah manual)
- Perintah dasar: buka website, cek waktu, cari di Google, cerita lucu, dll.

Instalasi library yang dibutuhkan (jalankan di terminal):
    pip install pyttsx3 SpeechRecognition pyaudio wikipedia

Catatan:
- Di Windows, pyaudio biasanya langsung berhasil di-install.
- Di Linux, mungkin perlu: sudo apt-get install portaudio19-dev python3-pyaudio
- Di Mac, mungkin perlu: brew install portaudio
- Jika microphone/pyaudio bermasalah, JARVIS otomatis akan pindah ke mode teks.
"""

import datetime
import webbrowser
import sys

# ---------- Coba import library suara, fallback ke mode teks jika gagal ----------
VOICE_MODE = True
try:
    import pyttsx3
    import speech_recognition as sr
except ImportError:
    VOICE_MODE = False
    print("[INFO] Library suara tidak ditemukan. Berjalan dalam mode teks saja.")
    print("       Install dengan: pip install pyttsx3 SpeechRecognition pyaudio\n")


class Jarvis:
    def __init__(self, nama_user="Tuan"):
        self.nama_user = nama_user
        self.voice_mode = VOICE_MODE

        if self.voice_mode:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 170)  # kecepatan bicara
                voices = self.engine.getProperty("voices")
                # Coba pilih suara wanita/pria jika tersedia (opsional)
                if voices:
                    self.engine.setProperty("voice", voices[0].id)
            except Exception as e:
                print(f"[WARNING] Gagal inisialisasi text-to-speech: {e}")
                self.voice_mode = False

    # ---------- OUTPUT ----------
    def bicara(self, teks):
        """Menampilkan teks + mengucapkannya jika mode suara aktif."""
        print(f"JARVIS: {teks}")
        if self.voice_mode:
            try:
                self.engine.say(teks)
                self.engine.runAndWait()
            except Exception:
                pass  # kalau gagal bicara, tetap lanjut pakai teks

    # ---------- INPUT ----------
    def dengarkan(self):
        """Mendengarkan suara dari mikrofon dan mengembalikan teks (lowercase)."""
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("\n[Mendengarkan...]")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            perintah = recognizer.recognize_google(audio, language="id-ID")
            print(f"Anda: {perintah}")
            return perintah.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            self.bicara("Maaf, saya tidak menangkap ucapan Anda.")
            return ""
        except sr.RequestError:
            self.bicara("Koneksi ke layanan pengenalan suara bermasalah.")
            return ""
        except Exception as e:
            print(f"[ERROR mikrofon] {e}")
            return ""

    def ambil_input(self):
        """Mengambil perintah dari suara (jika aktif) atau dari keyboard."""
        if self.voice_mode:
            teks = self.dengarkan()
            if teks:
                return teks
            # fallback ke teks kalau suara gagal menangkap
            return ""
        else:
            return input("Anda (ketik): ").lower()

    # ---------- LOGIKA PERINTAH ----------
    def proses_perintah(self, perintah):
        if not perintah:
            return True  # tidak ada perintah, lanjut loop

        if any(k in perintah for k in ["jam berapa", "waktu sekarang"]):
            sekarang = datetime.datetime.now().strftime("%H:%M")
            self.bicara(f"Sekarang jam {sekarang}")

        elif any(k in perintah for k in ["tanggal berapa", "hari ini tanggal"]):
            hari_ini = datetime.datetime.now().strftime("%d %B %Y")
            self.bicara(f"Hari ini tanggal {hari_ini}")

        elif "buka youtube" in perintah:
            self.bicara("Membuka YouTube")
            webbrowser.open("https://youtube.com")

        elif "buka google" in perintah:
            self.bicara("Membuka Google")
            webbrowser.open("https://google.com")

        elif perintah.startswith("cari ") or "cari di google" in perintah:
            query = perintah.replace("cari di google", "").replace("cari", "").strip()
            if query:
                self.bicara(f"Mencari {query} di Google")
                webbrowser.open(f"https://www.google.com/search?q={query}")
            else:
                self.bicara("Mau mencari apa?")

        elif any(k in perintah for k in ["siapa kamu", "siapa nama kamu"]):
            self.bicara("Saya JARVIS, asisten virtual pribadi Anda.")

        elif any(k in perintah for k in ["terima kasih", "makasih"]):
            self.bicara("Sama-sama, senang bisa membantu!")

        elif any(k in perintah for k in ["berhenti", "keluar", "matikan", "sampai jumpa"]):
            self.bicara("Baik, sampai jumpa!")
            return False  # keluar dari loop utama

        else:
            self.bicara("Maaf, saya belum mengerti perintah itu.")

        return True

    # ---------- LOOP UTAMA ----------
    def jalankan(self):
        mode = "SUARA" if self.voice_mode else "TEKS"
        self.bicara(f"Halo {self.nama_user}, saya JARVIS. Mode aktif: {mode}. Ada yang bisa saya bantu?")

        aktif = True
        while aktif:
            try:
                perintah = self.ambil_input()
                aktif = self.proses_perintah(perintah)
            except KeyboardInterrupt:
                self.bicara("Program dihentikan paksa. Sampai jumpa!")
                break


if __name__ == "__main__":
    nama = "Tuan"
    if len(sys.argv) > 1:
        nama = sys.argv[1]

    jarvis = Jarvis(nama_user=nama)
    jarvis.jalankan()
