# AI Chat Assistant with Speech Recognition and Multi-Source Query Processing

This project implements an advanced AI Chat Assistant capable of processing both text and speech queries. It can automatically decide between PDF documents and a PostgreSQL database as data sources based on the nature of the query.

## Features

- Text and speech input support
- Automatic data source selection (PDF documents or PostgreSQL database)
- PDF document processing and summarization
- PostgreSQL database querying
- Speech-to-text conversion using Sarvam AI API
- Integration with Google's Generative AI (Gemini) for natural language processing
- Dynamic PDF summary generation
- Streamlit-based user interface
- FastAPI backend for query processing

## Components

1. **Streamlit Frontend** (`chat_assist.py`):
   - Provides a user interface for text input and audio recording
   - Sends requests to the FastAPI backend

2. **FastAPI Backend** (`main.py`):
   - Handles incoming requests for text and audio queries
   - Integrates with the ChatBot for query processing

3. **Core Logic** (`app.py`):
   - Implements the ChatBot class with the following features:
     - PDF document loading and processing
     - Vector store creation for efficient document querying
     - PostgreSQL database integration
     - Automatic source selection based on query content
     - Speech-to-text conversion
     - Response generation using Google's Generative AI

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   GOOGLE_API_KEY=your_google_api_key
   SARVAM_API_KEY=your_sarvam_api_key
   SARVAM_API_URL=https://api.sarvam.ai/speech-to-text-translate
   API_URL=http://localhost:8000/query
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   EMBEDDING_MODEL=models/embedding-001
   GENERATIVE_MODEL=gemini-pro
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_HOST=your_db_host
   DB_PORT=your_db_port
   ```

4. Update PDF paths in `app.py`:
   ```python
   PDF_PATHS = ["path/to/your/pdf1.pdf", "path/to/your/pdf2.pdf"]
   ```

## Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. In a new terminal, run the Streamlit frontend:
   ```
   streamlit run chat_assist.py
   ```

3. Access the chat interface through your web browser at `http://localhost:8501`

## Usage

- Type your query in the text input field and click "Submit Text Query"
- Or click "Record Audio" to speak your query (5-second recording)
- The AI assistant will process your query, select the appropriate data source, and provide a response

## Note

Ensure that your microphone is properly configured and that you have the necessary permissions for audio recording.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.