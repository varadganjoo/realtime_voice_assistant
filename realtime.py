import asyncio
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
import json  # For chat history handling

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Groq client securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=groq_api_key)

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

# Chat history file
chat_history_file = "chat_history_realtime.json"

# Audio Callback Function
def audio_callback(indata, frames, time_info, status):
    """Callback function to receive audio data."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    # Convert the data to bytes and put it into the queue
    audio_queue.put(indata.copy())

# Background Noise Reduction (optional)
def reduce_background_noise(audio_data):
    """Reduce background noise in the audio data."""
    # Implement noise reduction if desired
    return audio_data

# Chat History Management
def read_chat_history(limit=None):
    """
    Read chat history from JSON file.
    If limit is specified, return only the last 'limit' messages.
    """
    if not os.path.exists(chat_history_file):
        return []
    try:
        with open(chat_history_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            if limit is not None:
                return data[-limit:]
            return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f"Chat history file is corrupted or has encoding issues: {e}. Starting fresh.")
        # Reset the chat history file with an empty list
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=4)
        return []

def write_to_chat_history(role, content):
    """
    Write a message to chat history in JSON format.
    """
    history = read_chat_history()
    message = {"role": role, "content": content}
    try:
        # Append the message to the chat history
        history.append(message)
        # Write the updated chat history to the JSON file
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=4)
        logging.info(f"Message written to chat history: {message}")
    except Exception as e:
        logging.error(f"Error writing to chat history: {e}")

# Recording and Transcription
def record_and_transcribe():
    """Function to record audio when speech is detected and transcribe it."""
    post_speech_counter = 0
    triggered = False
    voiced_frames = []

    while True:
        frame = audio_queue.get()
        if len(frame) == 0:
            continue

        # Convert frame to bytes for VAD analysis
        frame_bytes = frame.tobytes()

        # Voice Activity Detection (VAD)
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if is_speech:
            # Set the TTS stop event if speech is detected
            tts_stop_event.set()

            if not triggered:
                triggered = True
                logging.info("Speech detected. Recording started.")
                # Extend the buffer with the pre-speech audio
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()

            voiced_frames.append(frame_bytes)
            post_speech_counter = 0
        else:
            if triggered:
                ring_buffer.append(frame_bytes)
                post_speech_counter += 1

                # If there is no speech for enough frames, consider stopping the recording
                if post_speech_counter > post_speech_frames:
                    logging.info("Speech ended. Stopping recording.")
                    triggered = False
                    audio_data = b''.join(voiced_frames)

                    # Reduce background noise
                    audio_data = reduce_background_noise(audio_data)

                    # Transcribe the audio
                    transcribed_text = process_audio(audio_data)

                    if transcribed_text:
                        # Write the user's message to chat history
                        write_to_chat_history("user", transcribed_text)
                        # Process the query
                        threading.Thread(target=process_query, args=(transcribed_text,), daemon=True).start()

                    # Clear the voiced frames for the next session
                    voiced_frames = []
                    ring_buffer.clear()
            else:
                ring_buffer.append(frame_bytes)

# Audio Processing and Transcription
def process_audio(audio_data):
    """Function to send the recorded audio to the Groq API for transcription."""
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_io.seek(0)
        file = ("audio.wav", wav_io.read())

        # Transcribe the audio using the Groq API
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

# Query Processing
def process_query(query):
    """Process the query and get a response from the LLM."""
    response = run_general(query)
    # Write the assistant's response to chat history
    write_to_chat_history("assistant", response)
    asyncio.run(text_to_speech(response))
    return response

# General-Purpose Model Interaction
def run_general(query):
    """Use the general model to answer the query."""
    try:
        # Read the last 10 messages from the chat history
        chat_history = read_chat_history(limit=10)
        # Prepare the messages
        messages = []
        # Include the system prompt
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system_prompt = f"Your name is Jenny. You are a smart assistant. Today is {current_datetime}."
        messages.append({"role": "system", "content": system_prompt})
        # Add the chat history to the messages
        for msg in chat_history:
            if msg["role"] == "user" or msg["role"] == "assistant":
                messages.append({"role": msg["role"], "content": msg["content"]})
        # Add the current user query
        messages.append({"role": "user", "content": query})
        # Get response from LLM
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # General-purpose model
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

# Text-to-Speech
async def text_to_speech(text):
    """
    Convert the input text to speech using edge-tts and play it directly in the program.
    """
    try:
        # Before starting, clear the stop event
        tts_stop_event.clear()

        # Initialize the TTS object
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")

        # Stream the audio directly into memory
        audio_stream = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])

        # Convert the streamed bytes to an AudioSegment
        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream, format="mp3")

        # Play the audio using sounddevice
        play_audio(audio)

    except Exception as e:
        logging.error(f"An error occurred during TTS: {e}")

def play_audio(audio_segment):
    """
    Play an AudioSegment using sounddevice, allowing interruption.
    """
    # Convert AudioSegment to raw data for playback
    sample_rate = audio_segment.frame_rate
    channels = audio_segment.channels
    samples = np.array(audio_segment.get_array_of_samples())

    # Reshape samples for multi-channel audio
    if channels > 1:
        samples = samples.reshape((-1, channels))

    # Define the chunk size (number of frames)
    chunk_size = int(sample_rate * 0.1)  # 100 ms chunks

    # Open an output stream
    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=samples.dtype) as stream:
        index = 0
        while index < len(samples):
            if tts_stop_event.is_set():
                logging.info("TTS stopped due to user speech.")
                break
            # Get the next chunk
            chunk = samples[index:index+chunk_size]
            stream.write(chunk)
            index += chunk_size

# Start Listening
def start_listening():
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16',
                        blocksize=frame_size, callback=audio_callback):
        logging.info("Listening... Press Ctrl+C to stop.")
        threading.Thread(target=record_and_transcribe, daemon=True).start()
        try:
            while True:
                sd.sleep(1000)  # Keep the main thread alive
        except KeyboardInterrupt:
            logging.info("Stopping...")
            sys.exit()

def main():
    start_listening()

if __name__ == '__main__':
    main()
