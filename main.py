import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from TTS.api import TTS

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Voice ↔ Voice Chatbot", page_icon="🗣️", layout="centered")
st.title("🗣️ Voice ↔ Voice Chatbot")
st.caption("Ask by speaking — responses are spoken back.")

# ---------------------------
# Sidebar (Minimal)
# ---------------------------
with st.sidebar:
    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    llm_model = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a friendly helpful assistant."}
        ]
        st.session_state.last_audio = None
        st.session_state.pending_text = ""
        st.rerun()

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a friendly helpful assistant."}]
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "pending_text" not in st.session_state:
    st.session_state.pending_text = ""

# ---------------------------
# Helper Functions
# ---------------------------
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_reply(messages, model_id, temperature):
    client = get_client()
    chat = client.chat.completions.create(
        model=model_id, messages=messages, temperature=temperature, stream=False
    )
    return chat.choices[0].message.content


def synthesize_tts(text: str) -> str:
    tts = TTS("tts_models/en/ljspeech/glow-tts")
    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / f"reply_{int(time.time())}.wav"
    tts.tts_to_file(text=text, file_path=str(out_path))
    return str(out_path)


def write_wav(b: bytes) -> str:
    tmpdir = tempfile.mkdtemp()
    fpath = str(Path(tmpdir) / "input.wav")
    with open(fpath, "wb") as f:
        f.write(b)
    return fpath

# ---------------------------
# UI Input (VOICE ONLY)
# ---------------------------
st.subheader("Your Question — voice input only")
st.markdown("Click the mic, speak your question. The transcription will appear below and is editable before you press **Send**.")

# audio recorder (only input UI element)
audio_bytes = audio_recorder(text="🎤", icon_size="2x", sample_rate=16000)

# If we have a new recording, transcribe and populate the editable field
if audio_bytes:
    audio_path = write_wav(audio_bytes)
    st.audio(audio_path, format="audio/wav")

    with st.spinner("Transcribing..."):
        client = get_client()
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=f, model="whisper-large-v3-turbo", response_format="json"
            )
        # always overwrite pending_text for new recordings
        st.session_state.pending_text = resp.text.strip()

# Display the editable transcription if present
if st.session_state.pending_text:
    st.subheader("Transcribed (editable)")
    edited_text = st.text_area("Edit the transcribed text before sending:", value=st.session_state.pending_text, height=120)
    # Update session state with edited text
    st.session_state.pending_text = edited_text

# ---------------------------
# Send Button
# ---------------------------
if st.button("Send"):
    question = st.session_state.pending_text.strip()
    if not question:
        st.warning("Please record a question using the mic first (then edit the transcription if needed).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            reply = llm_reply(st.session_state.messages, llm_model, temperature)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.spinner("Generating voice..."):
            audio_out = synthesize_tts(reply)
            st.session_state.last_audio = audio_out

        # reset pending text after send
        st.session_state.pending_text = ""
        st.rerun()

# ---------------------------
# Chat Log Display
# ---------------------------
st.markdown("## 💬 Conversation")
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**🤖 Assistant:** {msg['content']}")
        if "last_audio" in st.session_state and st.session_state.last_audio:
            st.audio(st.session_state.last_audio)
