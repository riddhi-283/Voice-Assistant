# 🗣️ Voice ↔ Voice Chatbot (Streamlit + Whisper + Coqui TTS)

A lightweight, browser‑based **voice‑to‑voice** assistant without using any PAID APIs. Speak your question in the UI, the app transcribes it using **Groq Whisper**, gets an answer from a **Groq LLM (Llama 3.x)**, and speaks the reply back using **Coqui TTS** — all inside a simple Streamlit app.

---

## ✨ Features

* 🎙️ **Mic-to-text** in the browser via `audio-recorder-streamlit`
* 🧠 **LLM chat** with Groq (e.g., `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
* 🗣️ **Text-to-speech** via Coqui **Glow-TTS** (`tts_models/en/ljspeech/glow-tts`)
* 📝 **Editable transcription** before sending
* 🧹 **Clear conversation** with one click
* 🔐 Works with **GROQ\_API\_KEY** from sidebar or env var

---

## 🧩 How it works (high-level flow)

1. **Record audio** in the browser → saved as wav
2. **Transcribe** using Groq Whisper (`whisper-large-v3-turbo`)
3. **Chat completion** via Groq LLM (model & temperature configurable)
4. **Synthesize voice** with Coqui TTS (Glow‑TTS) → WAV file
5. **Playback** assistant audio in Streamlit

---

## 📁 Project structure

```
.
├─ main.py                 # The Streamlit app 
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

> **Note:** If your main script has a different name, replace `main.py` below accordingly.

---

## ✅ Prerequisites

* **Python** 3.9–3.11 (recommended: **3.10**)
* **FFmpeg** installed and on PATH
* Internet access to download model weights & call Groq API
* (Optional) **NVIDIA GPU** with a CUDA-compatible PyTorch build

### Install FFmpeg

* **macOS**: `brew install ffmpeg`
* **Ubuntu/Debian**: `sudo apt update && sudo apt install -y ffmpeg libsndfile1`
* **Windows**: Download from ffmpeg.org → extract → add `bin` folder to **PATH**

> Coqui TTS also uses `soundfile` (libsndfile). On Linux, we install `libsndfile1` above. If you see phonemizer/espeak warnings, install `espeak-ng` (`sudo apt install espeak-ng`).

---

## 🚀 Quickstart

### 1) Clone and enter the repo

```bash
git clone https://github.com/riddhi-283/Voice-Assistant.git
cd Voice-Assistant
```

### 2) Create & activate a virtual environment

**venv (cross‑platform):**

```bash
py -3.10 -m venv .venv 
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` (for reference):

```txt
streamlit==1.37.0
groq==0.11.0
audio-recorder-streamlit==0.0.8
TTS==0.22.0
torch==2.1.0
soundfile==0.12.1
numpy==1.22.0
openai-whisper==20240930
ffmpeg-python==0.2.0
```

### 4) Set your Groq API key

Get your free API key from **groq.com** and set it either via the sidebar or as an env var.

* **macOS/Linux (bash/zsh):**

  ```bash
  export GROQ_API_KEY="YOUR_KEY"
  ```
* **Windows (PowerShell):**

  ```powershell
  $env:GROQ_API_KEY = "YOUR_KEY"
  ```

> You can also paste the key into the Streamlit **sidebar** field.

### 5) Run the app

```bash
streamlit run main.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 🖱️ Using the app

1. Enter/paste your **Groq API Key** in the sidebar (or rely on env var)
2. Pick **LLM Model** and **Temperature**
3. Click the **mic button** → allow browser mic access → speak
4. Review the **editable transcription**, make changes if needed
5. Click **Send**
6. Read the text reply and listen to the generated **voice response**
7. Use **🗑️ Clear Conversation** to reset the session

---

## 🔧 Code tour (key functions)

* `get_client()` – creates a Groq client using `GROQ_API_KEY`
* `llm_reply(messages, model_id, temperature)` – calls Groq Chat Completions
* `synthesize_tts(text)` – runs Coqui `glow-tts` → writes a WAV to a temp folder
* `write_wav(bytes)` – persists incoming audio bytes as a wav file
* UI wiring:

  * `audio_recorder(..., sample_rate=16000)` gathers mic input
  * On new audio → Groq **Whisper** `audio.transcriptions.create(..., model="whisper-large-v3-turbo")`
  * Transcription saved to `st.session_state.pending_text` (editable `st.text_area`)
  * On **Send** → add user msg → call LLM → TTS → play audio & append to chat

