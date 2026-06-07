📽️ Clippify — AI Shorts Generator (Microservices System)

Clippify is a full AI-powered video processing pipeline that automatically converts long videos into viral short clips with subtitles, smart segmentation, resizing, and AI-based analysis.

It is built using a FastAPI microservices architecture with FFmpeg, Whisper, and LLM-based content analysis.


Features:

🎬 Upload long videos and auto-generate short viral clips

✂️ Smart clip detection using AI (viral segment extraction)

🎙️ Audio extraction using FFmpeg

🧠 Transcription using OpenAI Whisper

🔥 AI-based clip scoring (viral potential detection)

🎞️ Auto clip extraction with FFmpeg

📝 Karaoke-style animated subtitles

📱 TikTok / Instagram / Square format support

🔄 Fully automated pipeline orchestration

⚡ Microservices-based scalable architecture


Microservices:

| Service         | Port | Description                         |
| --------------- | ---- | ----------------------------------- |
| Orchestrator    | 8000 | Main pipeline controller            |
| Audio Extractor | 8001 | Extracts audio from video           |
| Transcriber     | 8002 | Whisper-based transcription         |
| Clip Analyzer   | 8003 | AI-based viral segment detection    |
| Clip Extractor  | 8004 | Extracts video clips using FFmpeg   |
| Subtitle Burner | 8005 | Burns animated subtitles            |
| Resizer Service | 8006 | Converts videos to portrait formats |


Tech Stack:

Python 3.10+

FastAPI

FFmpeg

OpenAI Whisper 

Groq (LLM analysis)

OpenCV

Streamlit (testing dashboard)

httpx (service communication)


⚙️ Setup Instructions:

1. Clone Repository:

git clone https://github.com/NayabAshraf/Clippify---AI-Shorts-Generator-Video-Tools.git
   
cd Clippify---AI-Shorts-Generator-Video-Tools

2. Create Virtual Environment:

python -m venv venv

venv\Scripts\activate   # Windows

3. Install Dependencies:

pip install -r requirements.txt

4. Install FFmpeg:

https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip?utm_source=chatgpt.com

5. Add Groq API Key:

In services/clip_analyzer_service.py, add your groq api key.

After download:

Extract ZIP, Rename folder to: ffmpeg, Move it to: C:\ffmpeg

Final path should be: C:\ffmpeg\bin\ffmpeg.exe

6. Run Services (in separate terminals):

uvicorn orchestrator.api:app --port 8000

uvicorn services.audio_extractor_service:app --port 8001

uvicorn services.transcriber_service:app --port 8002

uvicorn services.clip_analyzer_service:app --port 8003

uvicorn services.clip_extractor_service:app --port 8004

uvicorn services.subtitle_burner_service:app --port 8005

uvicorn resizer_service:app --port 8006

7. Run Test Dashboard:

streamlit run streamlit_test.py


API Endpoint:
POST /api/v1/generate


Author:
Nayab Ashraf
