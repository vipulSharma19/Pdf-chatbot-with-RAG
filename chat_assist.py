# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import requests
import sounddevice as sd
import numpy as np
import io
import time
from dotenv import load_dotenv
import os

load_dotenv()

# FastAPI endpoint URL
API_URL = os.getenv("API_URL")

st.set_page_config(page_title="AI Chat Assistant")

st.title("AI Chat Assistant")

# Text input
text_query = st.text_input("Enter your query:")

# Audio input
st.write("Or speak your query:")
if st.button("Record Audio"):
    duration = 5  # Recording duration in seconds
    fs = 44100  # Sample rate
    
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.write("Recording finished!")
    
    # Convert the NumPy array to bytes
    audio_bytes = recording.tobytes()
    
    st.write(f"Audio data size: {len(audio_bytes)} bytes")

    # Send the audio bytes to the API
    files = {"audio": ("query.wav", audio_bytes, "audio/wav")}
    try:
        response = requests.post(API_URL, files=files)
        st.write(f"Response status code: {response.status_code}")
        st.write(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Query: {result.get('query', 'No query returned')}")
                st.write(f"Response: {result.get('response', 'No response returned')}")
        else:
            st.error(f"Error processing audio query: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")

# Process text query
if st.button("Submit Text Query"):
    if text_query:
        try:
            response = requests.post(API_URL, data={"text": text_query})
            st.write(f"Response status code: {response.status_code}")
            st.write(f"Response content: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                st.write(f"Query: {result.get('query', 'No query returned')}")
                st.write(f"Response: {result.get('response', 'No response returned')}")
            else:
                st.error(f"Error processing text query: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Exception occurred: {str(e)}")
    else:
        st.warning("Please enter a text query")

# Add this at the end of the file
if __name__ == "__main__":
    st.write("AI Chat Assistant is running!")
