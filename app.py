"""Two-way voice interpreter (Gradio, Hugging Face Spaces).

All services free, no API keys:
  - STT:       OpenAI Whisper (runs locally via `transformers`)
  - Translate: deep-translator (Google public endpoint)
  - TTS:       gTTS (Google public endpoint)

Whisper runs inside the Space, so there are no IP-based blocks or rate
limits on speech recognition.
"""

import tempfile

import gradio as gr
from deep_translator import GoogleTranslator
from gtts import gTTS
from transformers import pipeline


# name -> (Whisper code, deep-translator code, gTTS code)
LANGUAGES = {
    "English":    {"whisper": "en", "tx": "en",    "tts": "en"},
    "Korean":     {"whisper": "ko", "tx": "ko",    "tts": "ko"},
    "Japanese":   {"whisper": "ja", "tx": "ja",    "tts": "ja"},
    "Chinese":    {"whisper": "zh", "tx": "zh-CN", "tts": "zh-CN"},
    "Spanish":    {"whisper": "es", "tx": "es",    "tts": "es"},
    "French":     {"whisper": "fr", "tx": "fr",    "tts": "fr"},
    "German":     {"whisper": "de", "tx": "de",    "tts": "de"},
    "Italian":    {"whisper": "it", "tx": "it",    "tts": "it"},
    "Portuguese": {"whisper": "pt", "tx": "pt",    "tts": "pt"},
    "Russian":    {"whisper": "ru", "tx": "ru",    "tts": "ru"},
    "Vietnamese": {"whisper": "vi", "tx": "vi",    "tts": "vi"},
    "Thai":       {"whisper": "th", "tx": "th",    "tts": "th"},
    "Hindi":      {"whisper": "hi", "tx": "hi",    "tts": "hi"},
    "Arabic":     {"whisper": "ar", "tx": "ar",    "tts": "ar"},
}

LANG_NAMES = list(LANGUAGES.keys())

# Load the ASR pipeline once at startup. whisper-small is a good quality
# / speed trade-off on CPU (~240 MB, ~5-10s for a short utterance).
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
)


def interpret(audio_path, source_name, target_name):
    if audio_path is None:
        return "", "", None
    if source_name not in LANGUAGES or target_name not in LANGUAGES:
        return "(invalid language selection)", "", None

    src = LANGUAGES[source_name]
    dst = LANGUAGES[target_name]

    try:
        result = asr(
            audio_path,
            generate_kwargs={"language": src["whisper"], "task": "transcribe"},
        )
        heard = (result.get("text") or "").strip()
    except Exception as exc:
        return f"(speech recognition failed: {exc})", "", None

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
    try:
        gTTS(text=translated, lang=dst["tts"]).save(tmp.name)
    except Exception as exc:
        return heard, translated, None

    return heard, translated, tmp.name


with gr.Blocks(title="Interpreter", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎤 Interpreter\n"
        "Two-way voice translation. Pick a language on each side, record, "
        "and the translation is spoken to the other person."
    )

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
