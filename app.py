"""Two-way voice interpreter (Gradio, for Hugging Face Spaces).

Person A speaks in language A -> app speaks translation in language B.
Person B speaks in language B -> app speaks translation in language A.

Free services, no API keys:
  - Google Web Speech via SpeechRecognition (STT)
  - Google Translate via deep-translator
  - Google Text-to-Speech via gTTS
"""

import tempfile

import gradio as gr
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS


# name -> (STT BCP-47 code, translate code, gTTS code)
LANGUAGES = {
    "English":    {"stt": "en-US", "tx": "en",    "tts": "en"},
    "Korean":     {"stt": "ko-KR", "tx": "ko",    "tts": "ko"},
    "Japanese":   {"stt": "ja-JP", "tx": "ja",    "tts": "ja"},
    "Chinese":    {"stt": "zh-CN", "tx": "zh-CN", "tts": "zh-CN"},
    "Spanish":    {"stt": "es-ES", "tx": "es",    "tts": "es"},
    "French":     {"stt": "fr-FR", "tx": "fr",    "tts": "fr"},
    "German":     {"stt": "de-DE", "tx": "de",    "tts": "de"},
    "Italian":    {"stt": "it-IT", "tx": "it",    "tts": "it"},
    "Portuguese": {"stt": "pt-BR", "tx": "pt",    "tts": "pt"},
    "Russian":    {"stt": "ru-RU", "tx": "ru",    "tts": "ru"},
    "Vietnamese": {"stt": "vi-VN", "tx": "vi",    "tts": "vi"},
    "Thai":       {"stt": "th-TH", "tx": "th",    "tts": "th"},
    "Hindi":      {"stt": "hi-IN", "tx": "hi",    "tts": "hi"},
    "Arabic":     {"stt": "ar-SA", "tx": "ar",    "tts": "ar"},
}

LANG_NAMES = list(LANGUAGES.keys())
recognizer = sr.Recognizer()


def interpret(audio_path, source_name, target_name):
    if audio_path is None:
        return "", "", None
    if source_name not in LANGUAGES or target_name not in LANGUAGES:
        return "(invalid language)", "", None

    src = LANGUAGES[source_name]
    dst = LANGUAGES[target_name]

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        heard = recognizer.recognize_google(audio, language=src["stt"])
    except sr.UnknownValueError:
        return "(couldn't understand the audio)", "", None
    except sr.RequestError as exc:
        return f"(speech API error: {exc})", "", None

    heard = (heard or "").strip()
    if not heard:
        return "(no speech detected)", "", None

    try:
        translated = GoogleTranslator(
            source=src["tx"], target=dst["tx"]
        ).translate(heard)
    except Exception as exc:
        return heard, f"(translation failed: {exc})", None

    translated = (translated or "").strip()
    if not translated:
        return heard, "(translation empty)", None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    gTTS(text=translated, lang=dst["tts"]).save(tmp.name)

    return heard, translated, tmp.name


with gr.Blocks(title="Interpreter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Interpreter\nTwo-way voice translation.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Person A")
            lang_a = gr.Dropdown(LANG_NAMES, value="English", label="Language")
            mic_a = gr.Audio(sources=["microphone"], type="filepath",
                             label="Tap record, speak, then stop")
            heard_a = gr.Textbox(label="Heard", interactive=False)
            translated_for_b = gr.Textbox(
                label="→ Translation for Person B", interactive=False)
            tts_for_b = gr.Audio(label="Spoken to Person B",
                                 autoplay=True, interactive=False)

        with gr.Column():
            gr.Markdown("### Person B")
            lang_b = gr.Dropdown(LANG_NAMES, value="Korean", label="Language")
            mic_b = gr.Audio(sources=["microphone"], type="filepath",
                             label="Tap record, speak, then stop")
            heard_b = gr.Textbox(label="Heard", interactive=False)
            translated_for_a = gr.Textbox(
                label="→ Translation for Person A", interactive=False)
            tts_for_a = gr.Audio(label="Spoken to Person A",
                                 autoplay=True, interactive=False)

    mic_a.stop_recording(
        fn=interpret,
        inputs=[mic_a, lang_a, lang_b],
        outputs=[heard_a, translated_for_b, tts_for_b],
    )
    mic_b.stop_recording(
        fn=interpret,
        inputs=[mic_b, lang_b, lang_a],
        outputs=[heard_b, translated_for_a, tts_for_a],
    )


if __name__ == "__main__":
    demo.launch()
