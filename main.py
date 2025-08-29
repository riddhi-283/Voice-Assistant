"""

"""
import os
import io
import time
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq

# For mic recording widget (simple, free)
from audio_recorder_streamlit import audio_recorder

# Optional local Whisper fallback
# pip install -U openai-whisper && ensure ffmpeg is installed
try:
    import whisper as local_whisper
except Exception:
    local_whisper = None

# Coqui TTS (free, natural voices)
# pip install TTS==0.22.0
from TTS.api import TTS


# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Voice ↔ Voice Chatbot (Groq + Coqui)", page_icon="🗣️", layout="wide")
st.title("🗣️ Voice ↔ Voice Chatbot (Groq + Coqui TTS)")
st.caption("Free stack: Groq Whisper + Groq LLMs + Coqui TTS — runs in Streamlit.")


# ---------------------------
# Sidebar: Settings
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # API key intake (env or UI)
    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.subheader("LLM Model (Groq)")
    # Curated, production models from Groq docs
    llm_model = st.selectbox(
        "Choose a chat model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "deepseek-r1-distill-llama-70b",  # preview
        ],
        index=0,
        help="These are model IDs from Groq's /models list."
    )

    st.subheader("Speech-to-Text (STT)")
    stt_provider = st.radio(
        "Transcription engine",
        ["Groq Whisper (recommended)", "Local Whisper (offline fallback)"],
        index=0
    )
    groq_whisper_model = st.selectbox(
        "Groq Whisper model",
        ["whisper-large-v3-turbo", "whisper-large-v3"],
        index=0,
        help="Turbo = faster & cheaper; v3 = max accuracy."
    )

    st.subheader("Text-to-Speech (TTS)")
    # Some lightweight, good-sounding free models
    tts_model_name = st.selectbox(
        "Coqui TTS model",
        [
            "tts_models/en/ljspeech/glow-tts",     # light, natural
            "tts_models/en/ljspeech/tacotron2-DDC",# classic
            "tts_models/en/vctk/vits",             # multi-speaker English
        ],
        index=0
    )
    tts_speaker = st.text_input(
        "Speaker (for multi-speaker models; leave blank otherwise)",
        value="p225" if "vctk" in tts_model_name else ""
    )

    st.subheader("Response Controls")
    temperature = st.slider("LLM Temperature", 0.0, 1.5, 0.4, 0.1)
    max_tokens = st.slider("Max completion tokens", 64, 4096, 1024, 64)

    st.markdown("---")
    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        st.session_state.history_text = ""
        st.session_state.last_audio_path = None
        st.rerun()


# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    # OpenAI/Groq-compatible message format
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, concise, friendly voice assistant."}
    ]
if "history_text" not in st.session_state:
    st.session_state.history_text = ""
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None


# ---------------------------
# Lazy singletons
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource(show_spinner=True)
def load_tts(model_name: str):
    # This loads once and is reused across requests
    return TTS(model_name=model_name)


@st.cache_resource(show_spinner=True)
def load_local_whisper():
    if local_whisper is None:
        raise RuntimeError("Local Whisper not installed. Run: pip install -U openai-whisper && install ffmpeg")
    return local_whisper.load_model("base")  # small & reasonably accurate; change to 'small' / 'medium' if you can


# ---------------------------
# Utilities
# ---------------------------
def write_wav(b: bytes, suffix=".wav") -> str:
    """Write raw audio bytes to a temporary WAV file and return the path."""
    tmpdir = tempfile.mkdtemp(prefix="st_v2v_")
    fpath = str(Path(tmpdir) / f"input{suffix}")
    with open(fpath, "wb") as f:
        f.write(b)
    return fpath


def transcribe_with_groq(audio_path: str, model_id: str, language: str = None) -> str:
    client = get_groq_client()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=model_id,
            language=language,
            response_format="json",
        )
    # Both Whisper v3 & v3-turbo return .text
    return resp.text.strip()


def transcribe_with_local_whisper(audio_path: str, language: str = None) -> str:
    model = load_local_whisper()
    result = model.transcribe(audio_path, language=language)
    return result.get("text", "").strip()


def llm_reply_groq(messages, model_id: str, temperature: float, max_tokens: int) -> str:
    client = get_groq_client()
    chat = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return chat.choices[0].message.content


def synthesize_tts(tts_model, text: str, speaker: str = "") -> str:
    out_dir = Path(tempfile.mkdtemp(prefix="st_v2v_out_"))
    out_path = out_dir / f"reply_{int(time.time())}.wav"

    if speaker:
        # For multi-speaker models (e.g., vctk/vits) you can pass speaker_name
        tts_model.tts_to_file(text=text, speaker=speaker, file_path=str(out_path))
    else:
        tts_model.tts_to_file(text=text, file_path=str(out_path))

    return str(out_path)


# ---------------------------
# Main UI
# ---------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎙️ Speak your question")
    st.caption("Click to start/stop recording. Or upload a file below if the mic widget is blocked by your browser.")
    audio_bytes = audio_recorder(pause_threshold=1.0, sample_rate=16000, text="Record / Stop")

    uploaded = st.file_uploader("…or drop a WAV/MP3/M4A", type=["wav", "mp3", "m4a", "ogg", "webm"], accept_multiple_files=False)

    transcript_text = ""

    if audio_bytes or uploaded:
        if uploaded:
            audio_path = write_wav(uploaded.read(), suffix=f".{uploaded.name.split('.')[-1]}")
        else:
            audio_path = write_wav(audio_bytes, suffix=".wav")

        st.audio(audio_path, format="audio/wav")
        st.session_state.last_audio_path = audio_path

        with st.spinner("Transcribing…"):
            if stt_provider.startswith("Groq"):
                if not groq_key:
                    st.error("Please provide your Groq API key in the sidebar to use Groq Whisper.")
                    st.stop()
                transcript_text = transcribe_with_groq(audio_path, groq_whisper_model)
            else:
                try:
                    transcript_text = transcribe_with_local_whisper(audio_path)
                except Exception as e:
                    st.error(f"Local Whisper error: {e}")
                    st.stop()

        if not transcript_text:
            st.warning("No speech detected. Try again.")
        else:
            st.success("Transcription complete.")
            st.text_area("Transcription", transcript_text, height=90)

            # Append user message into conversation + keep a plaintext log
            st.session_state.messages.append({"role": "user", "content": transcript_text})
            st.session_state.history_text += f"\nUser: {transcript_text}"

            # Call LLM
            with st.spinner(f"Calling LLM ({llm_model})…"):
                try:
                    assistant_text = llm_reply_groq(
                        st.session_state.messages, llm_model, temperature, max_tokens
                    )
                except Exception as e:
                    st.error(f"Groq chat error: {e}")
                    st.stop()

            # Save to history + display
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.session_state.history_text += f"\nAssistant: {assistant_text}"

            st.markdown("### 🤖 Assistant reply (text)")
            st.write(assistant_text)

            # TTS
            with st.spinner("Synthesizing speech…"):
                try:
                    tts_engine = load_tts(tts_model_name)
                    audio_out_path = synthesize_tts(tts_engine, assistant_text, tts_speaker.strip())
                    st.audio(audio_out_path)
                except Exception as e:
                    st.error(f"TTS error: {e}")
                else:
                    st.success("Playback ready ✅")


with col_right:
    st.subheader("🧠 Conversation")
    if len(st.session_state.messages) <= 1:
        st.info("No messages yet — speak or upload audio on the left.")
    else:
        for m in st.session_state.messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**Assistant:** {m['content']}")

    st.markdown("---")
    st.subheader("💬 Send a text message")
    user_text = st.text_input("Type instead of speaking", "")
    if st.button("Send"):
        if user_text.strip():
            st.session_state.messages.append({"role": "user", "content": user_text.strip()})
            with st.spinner(f"Calling LLM ({llm_model})…"):
                reply = llm_reply_groq(
                    st.session_state.messages, llm_model, temperature, max_tokens
                )
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.history_text += f"\nUser: {user_text}\nAssistant: {reply}"
            st.rerun()
