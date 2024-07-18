import streamlit as st
import numpy as np
import torchaudio
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import Ollama
from transformers import SeamlessM4TModel, AutoProcessor
from transformers import AutoProcessor, SeamlessM4Tv2Model
import time
import threading
import os
import subprocess
from audio_recorder_streamlit import audio_recorder
from speechbrain.inference.classifiers import EncoderClassifier

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Set page configuration
st.set_page_config(page_title="AI4ALPHA", page_icon="icons/ai_icon.png")
st.title("Multilingual Speech Bot")
st.text("")
st.text("")

# Custom CSS to adjust recorder height
st.markdown(
    '''
    <style>
        iframe[title="audio_recorder_streamlit.audio_recorder"] {
            height: auto;
        }
    </style>
    '''
, unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "file_path" not in st.session_state:
    st.session_state.file_path = ""
if "translated_text_from_audio" not in st.session_state:
    st.session_state.translated_text_from_audio = ""
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = ""
if "indic_lang" not in st.session_state:
    st.session_state.indic_lang = ""



# Load models
@st.cache_resource
def load_models():
    sb_lid_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
    seamless_model_idm = "facebook/hf-seamless-m4t-medium"
    seamless_model_id = "facebook/seamless-m4t-v2-large"
    seamless_processorm = AutoProcessor.from_pretrained(seamless_model_idm)
    seamless_processor = AutoProcessor.from_pretrained(seamless_model_id)
    seamless_modelm = SeamlessM4TModel.from_pretrained(seamless_model_idm)
    seamless_model = SeamlessM4Tv2Model.from_pretrained(seamless_model_id)
    device = "cpu"
    seamless_model = seamless_model.to(device)
    llm_model = Ollama(model="gemma:2b")

    return sb_lid_model, seamless_processor, seamless_processorm, seamless_model, seamless_modelm, llm_model, device

sb_lid_model, seamless_processor, seamless_processorm, seamless_model, seamless_modelm, llm_model, device = load_models()


# Language identification
def lid(file_path):
    try:
        signal = sb_lid_model.load_audio(file_path)
        prediction = sb_lid_model.classify_batch(signal)
        detected_lang = prediction[3][0]
        language_map = {
            "hi: Hindi": "hin",
            "ml: Malayalam": "mal",
            "ur: Urdu": "hin",
            "ta: Tamil": "tam",
            "bn: Bengali": "ben",
            "te: Telugu": "tel",
            "eng: English": "eng"
        }
        detected_lang = language_map.get(detected_lang, detected_lang)
        logging.info(f"Detected language: {detected_lang}")
        return detected_lang
    except Exception as e:
        logging.error(f"Error in language identification: {e}")
        raise


def indic_id(detected_lang):
    try:
        indic_map = {
            "mal": "ml",  
            "tam": "ta",
            "tel": "te",
        }
        indic_lang = indic_map.get(detected_lang)
        logging.info(f"Detected language mapped to Indic code: {indic_lang}")
        return indic_lang
    
    except Exception as e:
        logging.error(f"Error in Indic language identification: {e}")
        raise

# Speech transcription
def stt(waveform, sample_rate, lang_name):
    try:
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
        audio_inputs = seamless_processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt").to(device)
        output_tokens = seamless_model.generate(**audio_inputs, tgt_lang=lang_name, generate_speech=False)
        transcription = seamless_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        logging.info(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        logging.error(f"Error in speech transcription: {e}")
        raise


# Speech to translated text
def sttt(waveform, sample_rate):
    try:
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
        audio_inputs = seamless_processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt").to(device)
        output_tokens = seamless_model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
        translated_text_from_audio = seamless_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        logging.info(f"Translated text from audio: {translated_text_from_audio}")
        return translated_text_from_audio
    except Exception as e:
        logging.error(f"Error in speech-to-text translation: {e}")
        raise


# Text to translated text
def tttt(text, lang_name):
    try:
        text_inputs = seamless_processor(text=text, src_lang="eng", return_tensors="pt").to(device)
        output_tokens = seamless_model.generate(**text_inputs, tgt_lang=lang_name, generate_speech=False)
        translated_text_from_text = seamless_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        logging.info(f"Translated text from text: {translated_text_from_text}")
        return translated_text_from_text
    except Exception as e:
        logging.error(f"Error in text-to-text translation: {e}")
        raise

# Text to translated text
def ttttm(text, lang_name):
    try:
        text_inputs = seamless_processorm(text=text, src_lang="eng", return_tensors="pt").to(device)
        output_tokens = seamless_modelm.generate(**text_inputs, tgt_lang=lang_name, generate_speech=False)
        #output_tokens = seamless_model.generate(**text_inputs, tgt_lang=lang_name, generate_speech=False)
        translated_text_from_text = seamless_processorm.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        logging.info(f"Translated text from text: {translated_text_from_text}")
        return translated_text_from_text
    except Exception as e:
        logging.error(f"Error in text-to-text translation: {e}")
        raise

# Text to translated speech (for non-Indian languages)
def ttts(text, lang_name):
        text_inputs = seamless_processor(text=text, src_lang="eng", return_tensors="pt").to(device)
        audio_array_from_text = seamless_model.generate(**text_inputs, tgt_lang=lang_name)[0].cpu().numpy().squeeze()
        sample_rate = seamless_model.config.sampling_rate
        output_path = "audio_files/translated_audio.wav"
        sf.write(output_path, audio_array_from_text, sample_rate)
        print(f"Audio path: {output_path}")
        return output_path


def indic_tts(text, lang_name):
    try:
        # Define paths based on the language
        model_path = f"models/v1/{lang_name}/fastpitch/best_model.pth"
        config_path = f"models/v1/{lang_name}/fastpitch/config.json"
        vocoder_path = f"models/v1/{lang_name}/hifigan/best_model.pth"
        vocoder_config_path = f"models/v1/{lang_name}/hifigan/config.json"
        
        # Output directory and file name
        out_dir = "output"
        out_filename = f"output.wav"
        
        # Ensure the output directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Full path to the output file
        output_path = os.path.join(out_dir, out_filename)
    
        # Construct the command
        command = [
            "python3", "-m", "TTS.bin.synthesize",
            "--text", text,
            "--model_path", model_path,
            "--config_path", config_path,
            "--vocoder_path", vocoder_path,
            "--vocoder_config_path", vocoder_config_path,
            "--out_path", output_path,
            "--speaker_id", "male",
        ]
        
        try:
            # Run the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Synthesis completed successfully. Output saved to {output_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred during synthesis: {e.stderr}")
    
        return output_path
    except Exception as e:
        logging.error(f"Error in TTS synthesis: {e}")
        raise


# Initialize the LLM model
def llm(translated_text_from_audio):
    try:
        llm_prompt = translated_text_from_audio + " You should always answer the questions under 20 words in a single paragraph."
        llm_response = llm_model.invoke(llm_prompt)
        logging.info(f"LLM Response: {llm_response}")
        return llm_response
    except Exception as e:
        logging.error(f"Error in LLM model invocation: {e}")
        raise


# Function to stream and display text with typewriter effect
def stream_and_write(text, placeholder):
    current_text = ""
    for char in text:
        current_text += char
        placeholder.markdown(f"<p style='display:inline'>{current_text}</p>", unsafe_allow_html=True)
        time.sleep(0.01)


def handle_audio(output_file):
    try:
        audio_bytes = audio_recorder()

        if audio_bytes:
            ## Save the Recorded File
            output_file = "audio_file.wav"
            logging.info(f"Audio successfully saved to {output_file}.")
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

        logging.info(f"Output WAV: {output_file}")
        st.session_state.file_path = output_file
        waveform, sample_rate = torchaudio.load(output_file)

        # Perform language identification and speech-to-text translation concurrently
        with ThreadPoolExecutor() as executor:
            lid_future = executor.submit(lid, st.session_state.file_path)
            sttt_future = executor.submit(sttt, waveform, sample_rate)

            detected_lang = lid_future.result()
            translated_text_from_audio = sttt_future.result()

        # Update session state variables
        st.session_state.detected_lang = detected_lang
        st.session_state.translated_text_from_audio = translated_text_from_audio

        if detected_lang in ['tam', 'mal','tel']:

            # Perform speech-to-text and language model processing concurrently
            with ThreadPoolExecutor() as executor:
                stt_future = executor.submit(stt, waveform, sample_rate, detected_lang)
                llm_future = executor.submit(llm, translated_text_from_audio)

                for future in as_completed([stt_future, llm_future]):
                    if future == stt_future:
                        transcribed_text = stt_future.result()
                        st.session_state.transcribed_text = transcribed_text
                        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
                        os.remove('audio_file.wav')
                        # logging.DEBUG("Removed Audio File")
                        # Display user message
                        with st.container():
                            col1, col2 = st.columns([1, 9])
                            with col1:
                                st.image("icons/user_icon.png", width=40)
                            with col2:
                                placeholder = st.empty()
                                stream_and_write(transcribed_text, placeholder)

                    elif future == llm_future:
                        llm_response = llm_future.result()
                        st.session_state.chat_history.append(AIMessage(content=llm_response))

            # Perform text-to-text translation and display translated response
            translated_response = tttt(llm_response, st.session_state.detected_lang)
            st.session_state.chat_history.append(AIMessage(content=translated_response))
            with st.container():
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.image("icons/ai_icon.png", width=40)
                with col2:
                    placeholder = st.empty()
                    stream_and_write(translated_response, placeholder)

            # Identify the Indic language and synthesize the translated response
            # logging.debug("++++++DETETCTED LANGUAGE",detected_lang)
            indic_lang = indic_id(detected_lang)
            st.session_state.detected_lang = indic_lang
            audio_path = indic_tts(translated_response, indic_lang)
            if audio_path:
                audio_file = open(audio_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")

        else:
            
            # Perform speech-to-text and language model processing concurrently
            with ThreadPoolExecutor() as executor:
                stt_future = executor.submit(stt, waveform, sample_rate, detected_lang)
                llm_future = executor.submit(llm, translated_text_from_audio)

                for future in as_completed([stt_future, llm_future]):
                    if future == stt_future:
                        transcribed_text = stt_future.result()
                        st.session_state.transcribed_text = transcribed_text
                        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
                        os.remove('audio_file.wav')
                        # logging.DEBUG("Removed Audio File")
                        # Display user message
                        with st.container():
                            col1, col2 = st.columns([1, 9])
                            with col1:
                                st.image("icons/user_icon.png", width=40)
                            with col2:
                                placeholder = st.empty()
                                stream_and_write(transcribed_text, placeholder)

                    elif future == llm_future:
                        llm_response = llm_future.result()
                        st.session_state.chat_history.append(AIMessage(content=llm_response))

                        # Combined concurrent execution for TTTT and TTTS
                        with ThreadPoolExecutor() as executor:
                            ttttm_future = executor.submit(ttttm, llm_response, detected_lang)
                            ttts_future = executor.submit(ttts, llm_response, detected_lang)

                            for future in as_completed([ttttm_future, ttts_future]):
                                if future == ttttm_future:
                                    translated_text_from_text = ttttm_future.result()
                                    st.session_state.translated_text_from_text = translated_text_from_text

                                    # Stream the translated text response
                                    with st.container():
                                        col1, col2 = st.columns([1, 9])
                                        with col1:
                                            st.image("icons/ai_icon.png", width=20)
                                        with col2:
                                            placeholder = st.empty()
                                            stream_and_write(translated_text_from_text, placeholder)

                                elif future == ttts_future:
                                    translated_audio_path = ttts_future.result()
                                    st.session_state.translated_audio_path = translated_audio_path

                                    # Display translated audio
                                    if st.session_state.translated_audio_path:
                                        audio_file = open(st.session_state.translated_audio_path, "rb")
                                        audio_bytes = audio_file.read()
                                        st.audio(audio_bytes, format="audio/wav")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")


# Main app flow
if __name__ == "__main__":
    st.session_state.chat_history.append(AIMessage(content="Feel free to ask questions in your language"))
    output_file = "audio_file.wav"  # Define output_file globally or in the appropriate scope
    handle_audio(output_file)
