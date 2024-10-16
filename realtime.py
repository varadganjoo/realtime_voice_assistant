import asyncio
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import sys
import wave
import io
from groq import Groq
import collections
import os
import logging
import edge_tts
from pydub import AudioSegment
from io import BytesIO
import json
import time

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Groq client securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=groq_api_key)

# Chat history file
chat_history_file = "chat_history_realtime.json"

# Streamlit Session State Initialization (within Streamlit UI thread)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Initialize chat history

# Parameters
sample_rate = 16000  # 16 kHz
frame_duration_ms = 30  # Frame size in milliseconds
frame_size = int(sample_rate * frame_duration_ms / 1000)  # Number of samples per frame
channels = 1  # Mono audio

# Initialize VAD
vad = webrtcvad.Vad(3)  # Aggressiveness mode (0-3)

# Queue to hold audio frames
audio_queue = queue.Queue()

# Pre-buffering before and after speech detection
pre_speech_padding = 300  # Buffer duration before speech in milliseconds
post_speech_padding = 1000  # Allow for pauses in speech
pre_speech_frames = int(pre_speech_padding / frame_duration_ms)
post_speech_frames = int(post_speech_padding / frame_duration_ms)

# Ring buffer to hold pre-speech audio frames
ring_buffer = collections.deque(maxlen=pre_speech_frames)

# Define a threading event to control TTS playback
tts_stop_event = threading.Event()

# Chat History Management
def read_chat_history():
    """Read chat history from JSON file."""
    if not os.path.exists(chat_history_file):
        return []
    try:
        with open(chat_history_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f"Chat history file is corrupted: {e}.")
        return []

def write_to_chat_history(role, content):
    """Write message to chat history JSON file."""
    history = read_chat_history()
    message = {"role": role, "content": content}
    try:
        history.append(message)
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=4)
        logging.info(f"Message written to chat history: {message}")
    except Exception as e:
        logging.error(f"Error writing to chat history: {e}")

def audio_callback(indata, frames, time_info, status):
    """Callback function to receive audio data."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

def record_and_transcribe():
    post_speech_counter = 0
    triggered = False
    voiced_frames = []
    last_transcription = None  # To track the last transcribed text

    while True:
        frame = audio_queue.get()
        if len(frame) == 0:
            continue

        frame_bytes = frame.tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if is_speech:
            tts_stop_event.set()

            if not triggered:
                triggered = True
                logging.info("Speech detected. Recording started.")
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()

            voiced_frames.append(frame_bytes)
            post_speech_counter = 0
        else:
            if triggered:
                ring_buffer.append(frame_bytes)
                post_speech_counter += 1
                if post_speech_counter > post_speech_frames:
                    logging.info("Speech ended. Stopping recording.")
                    triggered = False
                    audio_data = b''.join(voiced_frames)
                    transcribed_text = process_audio(audio_data)

                    # Only process if the transcription differs from the last one
                    if transcribed_text and transcribed_text != last_transcription:
                        last_transcription = transcribed_text  # Update last_transcription
                        write_to_chat_history("user", transcribed_text)
                        threading.Thread(target=process_query, args=(transcribed_text,), daemon=True).start()
                    voiced_frames = []
                    ring_buffer.clear()
            else:
                ring_buffer.append(frame_bytes)

def process_audio(audio_data):
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_io.seek(0)
        file = ("audio.wav", wav_io.read())

        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3",
            response_format="text"
        )
        logging.info(f"Transcription: {transcription.strip()}")
        return transcription.strip()
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

def process_query(query):
    response = run_general(query)
    write_to_chat_history("assistant", response)
    asyncio.run(text_to_speech_streamed(response))
    return response

def run_general(query):
    try:
        chat_history = read_chat_history()
        messages = []
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system_prompt = f"Your name is Jenny. You are a smart assistant. Today is {current_datetime}."
        messages.append({"role": "system", "content": system_prompt})
        for msg in chat_history:
            if msg["role"] == "user" or msg["role"] == "assistant":
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            max_tokens=1024,
            stream=False
        )
        final_response = response.choices[0].message.content.strip()
        logging.info(f"LLM Response: {final_response}")
        return final_response
    except Exception as e:
        logging.error(f"Error during general model interaction: {e}")
        return "An error occurred while processing your request."

async def text_to_speech_streamed(text):
    try:
        tts_stop_event.clear()
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        audio_stream = BytesIO()
        response_buffer = ""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
            if chunk["type"] == "text":
                response_buffer += chunk["data"]
                write_to_chat_history("assistant", response_buffer)  # Update letter by letter
                time.sleep(0.05)  # Sleep to simulate letter-by-letter typing effect
        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play_audio(audio)
    except Exception as e:
        logging.error(f"Error during TTS: {e}")

def play_audio(audio_segment):
    sample_rate = audio_segment.frame_rate
    channels = audio_segment.channels
    samples = np.array(audio_segment.get_array_of_samples())
    if channels > 1:
        samples = samples.reshape((-1, channels))
    chunk_size = int(sample_rate * 0.1)
    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=samples.dtype) as stream:
        index = 0
        while index < len(samples):
            if tts_stop_event.is_set():
                logging.info("TTS stopped due to user speech.")
                break
            chunk = samples[index:index+chunk_size]
            stream.write(chunk)
            index += chunk_size

def start_listening():
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16',
                        blocksize=frame_size, callback=audio_callback):
        logging.info("Listening... Press Ctrl+C to stop.")
        threading.Thread(target=record_and_transcribe, daemon=True).start()
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            logging.info("Stopping...")
            sys.exit()

# Streamlit UI
def streamlit_ui():
    st.title("AI Voice Assistant")

    # Add a container to display chat history
    chat_container = st.empty()

    # Continuously poll for new chat history updates
    while True:
        chat_history = read_chat_history()

        # Render the chat history
        with chat_container.container():
            st.write("### Chat History")
            for chat in chat_history:
                if chat["role"] == "user":
                    st.markdown(f"**You:** {chat['content']}")
                elif chat["role"] == "assistant":
                    st.markdown(f"**Jenny:** {chat['content']}")

        # Sleep for a short interval before polling again (simulate real-time updates)
        time.sleep(1)

# Run everything
if __name__ == "__main__":
    threading.Thread(target=start_listening, daemon=True).start()  # Start the listening thread
    streamlit_ui()  # Run the Streamlit UI
