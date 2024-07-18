# AI4ALPHA: Multilingual Speech Bot

## Overview
AI4ALPHA is a multilingual speech bot application built using Streamlit. The app enables users to record audio, transcribe speech, translate it into multiple languages, and generate responses using a language model. It supports language identification, speech-to-text transcription, and text translation.

## Features
- Record and save audio
- Language identification
- Speech-to-text transcription
- Speech-to-text translation
- Text-to-text translation
- Text-to-speech translation for non-Indian languages
- Integration with a language model for generating responses

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/ai4alpha.git
    cd ai4alpha
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Dependencies
- `streamlit`
- `numpy`
- `torchaudio`
- `soundfile`
- `concurrent.futures`
- `langchain_core`
- `langchain_community`
- `transformers`
- `audio_recorder_streamlit`
- `speechbrain`
- `logging`

## Running the Application

1. **Ensure the virtual environment is activated.**

2. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

3. **Open the app in your web browser:**
    The app will usually be accessible at `http://localhost:8501`

## Project Structure


## Usage

1. **Record Audio:**
    - Click on the record button to capture audio.
    - The recorded audio will be saved and processed.

2. **Transcription and Translation:**
    - The app will identify the language, transcribe the speech, and translate it as needed.

3. **Language Model Interaction:**
    - The transcribed text will be used to generate a response using the integrated language model.
    - The response will be translated back to the detected language and provided as text and audio.

## Functions Overview

### `load_models()`
Loads the required models for language identification, speech-to-text, and language model.

### `lid(file_path)`
Identifies the language of the audio file.

### `stt(waveform, sample_rate, lang_name)`
Transcribes speech to text.

### `sttt(waveform, sample_rate)`
Translates speech to text in English.

### `tttt(text, lang_name)`
Translates text to the target language text.

### `ttts(text, lang_name)`
Translates text to the target language speech.

### `llm(translated_text_from_audio)`
Generates a response using the language model.

### `handle_audio(output_file)`
Handles the audio recording, processing, and displaying the results.

## Logging
The application uses Python's `logging` module to log information and errors. Logs are helpful for debugging and understanding the application's flow.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
