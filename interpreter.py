"""Two-way voice interpreter (Tk GUI).

Person A speaks in language A -> app speaks the translation in language B.
Person B speaks in language B -> app speaks the translation in language A.

Free services, no API keys:
  - Google Web Speech via SpeechRecognition (STT)
  - Google Translate via deep-translator
  - Google Text-to-Speech via gTTS, played with pygame.

Mic capture uses sounddevice (bundled portaudio) so no system portaudio
headers are required.
"""

import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import pygame
import sounddevice as sd
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS


# name -> (STT BCP-47 code, translate code, gTTS code)
LANGUAGES = {
    "English":    ("en-US", "en",    "en"),
    "Korean":     ("ko-KR", "ko",    "ko"),
    "Japanese":   ("ja-JP", "ja",    "ja"),
    "Chinese":    ("zh-CN", "zh-CN", "zh-CN"),
    "Spanish":    ("es-ES", "es",    "es"),
    "French":     ("fr-FR", "fr",    "fr"),
    "German":     ("de-DE", "de",    "de"),
    "Italian":    ("it-IT", "it",    "it"),
    "Portuguese": ("pt-BR", "pt",    "pt"),
    "Russian":    ("ru-RU", "ru",    "ru"),
    "Vietnamese": ("vi-VN", "vi",    "vi"),
    "Thai":       ("th-TH", "th",    "th"),
    "Hindi":      ("hi-IN", "hi",    "hi"),
    "Arabic":     ("ar-SA", "ar",    "ar"),
}


SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # int16
CHUNK_SECONDS = 0.05
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)
START_TIMEOUT_SECONDS = 6
MAX_UTTERANCE_SECONDS = 15
TRAILING_SILENCE_SECONDS = 1.2


def _rms(int16_chunk):
    arr = int16_chunk.astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


class InterpreterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interpreter")
        self.root.geometry("860x580")

        self.recognizer = sr.Recognizer()
        pygame.mixer.init()

        self.busy_lock = threading.Lock()
        self.ui_queue = queue.Queue()

        self._build_ui()
        self.root.after(60, self._drain_ui_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Interpreter",
                  font=("Helvetica", 20, "bold")).pack(side="left")
        ttk.Button(header, text="Swap ⇅",
                   command=self._swap_languages).pack(side="right")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.panel_a = self._build_panel(body, "Person A", "English", column=0)
        self.panel_b = self._build_panel(body, "Person B", "Korean",  column=1)

        footer = ttk.Frame(self.root, padding=(16, 8))
        footer.pack(fill="x")
        self.status_var = tk.StringVar(value="Tap a mic and start speaking.")
        ttk.Label(footer, textvariable=self.status_var,
                  foreground="#555").pack(side="left")

    def _build_panel(self, parent, title, default_lang, column):
        frame = ttk.LabelFrame(parent, text=title, padding=12)
        frame.grid(row=0, column=column, sticky="nsew", padx=6, pady=6)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        frame.rowconfigure(5, weight=1)

        lang_row = ttk.Frame(frame)
        lang_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(lang_row, text="Language:").pack(side="left")
        lang_var = tk.StringVar(value=default_lang)
        combo = ttk.Combobox(lang_row, textvariable=lang_var,
                             values=list(LANGUAGES.keys()),
                             state="readonly", width=16)
        combo.pack(side="left", padx=(8, 0))

        mic_btn = ttk.Button(frame, text="🎤  Speak",
                             command=lambda: self._on_mic_click(title))
        mic_btn.grid(row=1, column=0, sticky="ew", pady=(0, 12), ipady=10)

        ttk.Label(frame, text="Heard",
                  foreground="#888").grid(row=2, column=0, sticky="w")
        heard = tk.Text(frame, height=4, wrap="word",
                        font=("Helvetica", 12), relief="solid", borderwidth=1)
        heard.grid(row=3, column=0, sticky="nsew", pady=(2, 10))
        heard.configure(state="disabled")

        ttk.Label(frame, text="Translated",
                  foreground="#888").grid(row=4, column=0, sticky="w")
        translated = tk.Text(frame, height=4, wrap="word",
                             font=("Helvetica", 13, "bold"),
                             relief="solid", borderwidth=1)
        translated.grid(row=5, column=0, sticky="nsew", pady=(2, 0))
        translated.configure(state="disabled")

        return {
            "title": title,
            "lang_var": lang_var,
            "mic_btn": mic_btn,
            "heard": heard,
            "translated": translated,
        }

    # --------------------------------------------------------------- helpers
    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _post(self, fn, *args):
        self.ui_queue.put((fn, args))

    def _drain_ui_queue(self):
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.root.after(60, self._drain_ui_queue)

    def _set_status(self, text):
        self.status_var.set(text)

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.panel_a["mic_btn"].configure(state=state)
        self.panel_b["mic_btn"].configure(state=state)

    def _swap_languages(self):
        a = self.panel_a["lang_var"].get()
        b = self.panel_b["lang_var"].get()
        self.panel_a["lang_var"].set(b)
        self.panel_b["lang_var"].set(a)

    # ---------------------------------------------------------------- flow
    def _on_mic_click(self, speaker_title):
        if not self.busy_lock.acquire(blocking=False):
            return
        if speaker_title == "Person A":
            src, dst = self.panel_a, self.panel_b
        else:
            src, dst = self.panel_b, self.panel_a

        self._post(self._set_buttons_enabled, False)
        threading.Thread(target=self._run_pipeline,
                         args=(src, dst), daemon=True).start()

    def _run_pipeline(self, src, dst):
        try:
            src_lang = src["lang_var"].get()
            dst_lang = dst["lang_var"].get()
            src_stt, src_tx, _ = LANGUAGES[src_lang]
            _, dst_tx, dst_tts = LANGUAGES[dst_lang]

            self._post(self._set_status, f"Listening ({src_lang})…")
            self._post(self._set_text, src["heard"], "")
            self._post(self._set_text, dst["translated"], "")

            audio = self._listen()
            if audio is None:
                self._post(self._set_status,
                           "Didn't catch anything. Try again.")
                return

            self._post(self._set_status, "Recognizing…")
            try:
                heard = self.recognizer.recognize_google(
                    audio, language=src_stt)
            except sr.UnknownValueError:
                self._post(self._set_status,
                           "Couldn't understand the audio.")
                return
            except sr.RequestError as exc:
                self._post(self._set_status, f"Speech API error: {exc}")
                return

            self._post(self._set_text, src["heard"], heard)

            self._post(self._set_status, "Translating…")
            translated = GoogleTranslator(
                source=src_tx, target=dst_tx).translate(heard)
            if not translated:
                self._post(self._set_status, "Translation returned empty.")
                return
            self._post(self._set_text, dst["translated"], translated)

            self._post(self._set_status, f"Speaking ({dst_lang})…")
            self._speak(translated, dst_tts)

            self._post(self._set_status, "Done. Tap a mic for the next turn.")
        except Exception as exc:
            self._post(self._set_status, f"Error: {exc}")
        finally:
            self._post(self._set_buttons_enabled, True)
            self.busy_lock.release()

    # ---------------------------------------------------------------- audio
    def _listen(self):
        """Capture one utterance with light VAD. Returns sr.AudioData or None."""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", blocksize=CHUNK_FRAMES) as stream:
            # Calibrate on ~0.4s of ambient noise.
            calib_chunks = int(0.4 / CHUNK_SECONDS)
            rms_samples = []
            for _ in range(calib_chunks):
                data, _ = stream.read(CHUNK_FRAMES)
                rms_samples.append(_rms(data))
            ambient = float(np.mean(rms_samples)) if rms_samples else 0.0
            threshold = max(ambient * 2.5, 350.0)

            # Wait for speech to start.
            start_chunks_max = int(START_TIMEOUT_SECONDS / CHUNK_SECONDS)
            buffer = []
            started = False
            for _ in range(start_chunks_max):
                data, _ = stream.read(CHUNK_FRAMES)
                if _rms(data) > threshold:
                    buffer.append(data.copy())
                    started = True
                    break
            if not started:
                return None

            # Record until trailing silence or max duration.
            max_chunks = int(MAX_UTTERANCE_SECONDS / CHUNK_SECONDS)
            silence_limit = int(TRAILING_SILENCE_SECONDS / CHUNK_SECONDS)
            silence_run = 0
            for _ in range(max_chunks):
                data, _ = stream.read(CHUNK_FRAMES)
                buffer.append(data.copy())
                if _rms(data) < threshold:
                    silence_run += 1
                    if silence_run >= silence_limit:
                        break
                else:
                    silence_run = 0

        audio = np.concatenate(buffer).reshape(-1).astype(np.int16)
        return sr.AudioData(audio.tobytes(), SAMPLE_RATE, SAMPLE_WIDTH)

    def _speak(self, text, tts_lang):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            gTTS(text=text, lang=tts_lang).save(path)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(30)
            pygame.mixer.music.unload()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def main():
    root = tk.Tk()
    InterpreterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
