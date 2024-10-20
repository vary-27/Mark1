import streamlit as st
import warnings
import yt_dlp
import whisper
import spacy
import os
from string import punctuation as punct
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
import numpy as np
import pandas as pd
import spacy.cli
from spacy.cli import download
#spacy.cli.download("en_core_web_sm")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
nlp = spacy.load('en_core_web_sm')


# Initialize spaCy model and Whisper model
#nlp = spacy.load('en_core_web_sm')
model = whisper.load_model("base")

# Stopwords for summarization
stopwords = list(STOP_WORDS)

# Function to download audio from YouTube
def process_video(link):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(link, download=True)
            file_path = ydl.prepare_filename(info)
            file_path = file_path.rsplit('.', 1)[0] + '.mp3'
            return file_path
        except Exception as e:
            st.error(f"Error downloading video: {e}")
            return None

# Function to transcribe audio
def transcribe_audio(file_path):
    if os.path.exists(file_path):
        try:
            result = model.transcribe(file_path)
            return result['text']
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None
    else:
        st.error("Audio file not found.")
        return None

# Function to summarize the transcript
def summarize(text, max_word_count=100):
    doc = nlp(text)
    punctuation = punct + '\n'
    word_freq = {}

    # Calculate word frequency
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    
    max_freq = max(word_freq.values(), default=0)

    # Normalize frequencies
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq if max_freq > 0 else 0

    # Scoring sentences based on word frequencies
    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text.lower()]

    # Get the top sentences
    sorted_sents = nlargest(len(sent_scores), sent_scores, key=sent_scores.get)

    summary = []
    word_count = 0

    # Generate summary with word limit
    for sent in sorted_sents:
        summary.append(sent.text)
        word_count += len(sent.text.split())
        if word_count >= max_word_count:
            break

    return ' '.join(summary)

# Streamlit app layout
st.set_page_config(page_title="YouTube Video Summary Generator", layout="centered")
st.title("🎥 YouTube Video Summary Generator ✂️")
st.markdown("""
    **Welcome!** This app will help you summarize any YouTube video.
    Just enter the link below, and I'll take care of the rest! 🚀
""")

# User input for YouTube video link
video_url = st.text_input("YouTube Video Link:", placeholder="Paste your YouTube link here...")

# Button to generate summary
if st.button("Generate Summary"):
    if video_url:
        with st.spinner("Downloading and processing video..."):
            file_path = process_video(video_url)
            if file_path:
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(file_path)
                    if transcript:
                        summary = summarize(transcript)
                        st.subheader("📋 Summary:")
                        st.write(summary)
    else:
        st.warning("Please enter a valid YouTube video link!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Vardaan Mathur")
st.markdown("### How it works:")
st.markdown("""
1. **Enter a YouTube video link** in the input box above.
2. Click on **Generate Summary**.
3. Enjoy the summarized content! 🎉

Feel free to test it out with different video links!
""")
