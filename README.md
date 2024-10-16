# Realtime Voice Conversation Assistant

Welcome to the **Realtime Voice Conversation Assistant**, a voice-controlled assistant built using Groq's language model APIs. It enables real-time transcription and intelligent response generation, much like OpenAI's voice assistant, but is powered by Groq's LLM (Large Language Model) for conversational AI. The system features voice input, natural language understanding, and audio output capabilities for a smooth, interactive conversation experience.

## How It Works

This assistant listens to your voice in real time, detects when you're speaking, transcribes the audio to text, and sends it to the Groq LLM for understanding and response generation. It also converts text-based responses into speech using `edge-tts`, allowing a full real-time conversational experience.

### Key Features

- **Voice Activity Detection**: The assistant uses WebRTC's Voice Activity Detection (VAD) to capture spoken input, which triggers when speech is detected.
- **Realtime Transcription**: Uses Groq's Whisper model to transcribe the recorded voice data into text.
- **Conversational AI**: Queries are sent to Groq's LLM for intelligent, context-aware responses.
- **Text-to-Speech**: Uses `edge-tts` to convert responses from the assistant into natural-sounding speech.
- **Chat History**: Stores chat history in JSON format, providing context for better conversational experiences.

## Getting Started

### Prerequisites

- Python 3.8+
- Access to Groq APIs (Sign up and get your API key from the [Groq website](https://groq.com/))

### Installation

1. Clone the repository:

   bash
   git clone https://github.com/your-repository-url.git

2. Navigate to the project directory:

   bash
   cd realtime-voice-assistant

3. Install the required dependencies:

   bash
   pip install -r requirements.txt

4. Set up your environment variables by creating a `.env` file with your Groq API key:

```python
GROQ_API_KEY=your_groq_api_key
```


### Running the Assistant

To start the real-time assistant, simply run:

bash
python app.py

Once the program is running, it will begin listening to your voice. When speech is detected, it transcribes the audio, sends it to Groq's LLM, and reads back the response via TTS.

### How the Assistant Works

- **Audio Input**: Listens for user speech using `sounddevice` and processes it with WebRTC VAD for detecting when you're talking.
- **Transcription**: Uses Groq's Whisper model to transcribe the audio input.
- **Conversational AI**: Queries Groq's Llama model for intelligent responses.
- **Text-to-Speech (TTS)**: Converts the assistant's responses into speech using Microsoft's `edge-tts`.
- **Chat History**: Keeps a record of interactions in a JSON file (`chat_history_realtime.json`) to maintain context for better conversation flow.

## File Structure

- `app.py`: Main application file.
- `chat_history_realtime.json`: Stores the chat history.
- `requirements.txt`: List of required dependencies.
- `.env`: Stores the Groq API key.

## Customization

You can modify the following parameters:

- **VAD Aggressiveness**: The `webrtcvad.Vad(3)` function defines how aggressive the speech detection is. Higher values (0-3) make it more sensitive to non-speech noise.
- **Sample Rate**: The assistant operates at a sample rate of 16kHz (`16000 Hz`). You can adjust this depending on your audio input quality requirements.
- **Models**: You can choose different models from Groq for transcription and conversation, such as `whisper-large-v3` for transcription and `llama-3.1-70b-versatile` for chat completions.

## Limitations

- **Background Noise**: While VAD helps in detecting speech, excessive background noise may still interfere with detection and transcription. Implementing advanced noise reduction techniques would improve performance.
- **Latency**: Response times depend on the network connection, API latency, and processing time for transcription and TTS.

## Future Improvements

- Implementing more advanced noise reduction to improve transcription accuracy in noisy environments.
- Adding support for multi-lingual transcription and response generation.
- Expanding the assistant’s capabilities to include more specialized tasks.

## Contributing

Feel free to fork this repository, submit pull requests, or suggest improvements. Contributions are welcome!

---

### Acknowledgements

Special thanks to Groq for providing the APIs used for both transcription and conversational responses, and to Microsoft for `edge-tts`, which powers the text-to-speech functionality.
