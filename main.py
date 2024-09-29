from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from test import record_audio
from app import ChatBot, PostgreSQLDatabase, PDFLoader, DocumentSplitter, speech_to_text
import io
import numpy as np
import logging
import tempfile
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize ChatBot (you might want to do this in a startup event)
PDF_PATHS = ["docs\iesc111.pdf"]
postgres_db = PostgreSQLDatabase()
chat_bot = ChatBot(PDF_PATHS, postgres_db)

class QueryRequest(BaseModel):
    text: Optional[str] = None

@app.post("/query")
async def process_query(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    logger.info(f"Received request. Text: {text is not None}, Audio: {audio is not None}")
    
    if text:
        query = text
        logger.info(f"Processing text query: {query}")
    elif audio:
        audio_bytes = record_audio()
        logger.info(f"Received audio data of size: {len(audio_bytes)} bytes")
        
        try:
            # Process the audio bytes
            query = speech_to_text(audio_bytes)
            print("spoked qurey",query)
            if query is None or query == '':
                return {"error": "Speech recognition could not understand the audio. Please try again."}
            logger.info(f"Speech-to-text result: {query}")
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {"error": f"Error processing audio: {str(e)}"}
    else:
        logger.warning("No text or audio provided")
        return {"error": "Please provide either text or audio input"}

    # Process the query using the ChatBot
    logger.info(f"Processing query with ChatBot: {query}")
    response = chat_bot.get_response(query)
    logger.info(f"ChatBot response: {response}")
    
    return {"query": query, "response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)