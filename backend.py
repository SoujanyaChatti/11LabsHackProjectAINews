from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import os
import base64
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import httpx

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
FAL_TTS_DIALOG_ENDPOINT = "https://fal.run/fal-ai/playai/tts/dialog"
FAL_TTS_V3_ENDPOINT = "https://fal.run/fal-ai/playai/tts/v3"

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_root():
    return FileResponse("frontend.html")



# ... (rest of your existing backend.py code remains unchanged)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

def clean_query(text):
    words = text.lower().split()
    stopwords = {"give", "me", "news", "about", "the", "case", "latest", "report", "vs", "is", "explain"}
    keywords = [word for word in words if word not in stopwords]
    return " ".join(keywords)

def limit_words(text, max_words=450):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " [Truncated to 3 minutes]"
    return text

async def fetch_audio_bytes(url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
            print(f"Fetched audio bytes from {url}, size: {len(response.content)} bytes")
            return base64.b64encode(response.content).decode('utf-8')
        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching audio: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching audio: {str(e)}")
            return None

async def generate_dialog_audio(debate_text):
    print(f"Generating dialog audio (length: {len(debate_text.split())} words):\n{debate_text}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                FAL_TTS_DIALOG_ENDPOINT,
                headers={
                    "Authorization": f"Key {FAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": debate_text,
                    "voices": [
                        {"voice": "Jennifer (English (US)/American)", "turn_prefix": "Alice: "},
                        {"voice": "Furio (English (IT)/Italian)", "turn_prefix": "Bob: "}
                    ],
                    "response_format": "url"
                },
                timeout=120  # Increased from 60 to 120 seconds
            )
            response.raise_for_status()
            data = response.json()
            print(f"Dialog API response: {data}")
            audio_url = data.get("audio", {}).get("url")
            if not audio_url:
                print(f"Warning: No audio URL in dialog response: {data}")
                return None
            return audio_url
        except httpx.HTTPStatusError as e:
            print(f"HTTP error (dialog): {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"Request error (dialog) details: {str(e)} - Request: {e.request.url}")
            return None
        except Exception as e:
            print(f"Unexpected error (dialog): {str(e)}")
            return None

async def generate_v3_audio(report_text):
    print(f"Generating v3 audio (length: {len(report_text.split())} words):\n{report_text}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                FAL_TTS_V3_ENDPOINT,
                headers={
                    "Authorization": f"Key {FAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": report_text,
                    "voice": "Jennifer (English (US)/American)",
                    "response_format": "url"
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            print(f"V3 API response: {data}")
            audio_url = data.get("audio", {}).get("url")
            if not audio_url:
                print(f"Warning: No audio URL in v3 response: {data}")
                return None
            return audio_url  # Return URL directly
        except httpx.HTTPStatusError as e:
            print(f"HTTP error (v3): {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"Request error (v3): {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error (v3) details: {str(e)}")
            return None

@app.get("/news_report")
async def generate_news_report(topic: str, area: str = None, debate: bool = False):
    processed_topic = clean_query(topic)
    url = f"https://newsapi.org/v2/everything?q={processed_topic}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to fetch news"}

    articles = response.json().get("articles", [])[:5]
    if not articles:
        return {"error": "No relevant articles found"}

    if area:
        articles = [a for a in articles if area.lower() in a["title"].lower() or area.lower() in a["description"].lower()]
        if not articles:
            return {"error": f"No news found for {processed_topic} in {area}"}

    news_content = "\n\n".join([f"Title: {a['title']}\nSummary: {a['description']}" for a in articles])

    prompt = f"""
    You are an AI news reporter. Create a concise news report (max 450 words, ~3 minutes at 150 wpm) based on:
    
    {news_content}
    
    Keep it engaging and structured.
    """
    if debate:
        prompt += """
        **Format:**
        1. A debate between Alice and Bob (max 300 words total):
           - **Alice** argues FOR the topic (1-2 sentences per turn).
           - **Bob** argues AGAINST the topic (1-2 sentences per turn).
           - At least 3 rounds of exchange.
           - Use "Alice: " and "Bob: " prefixes.
        2. A neutral summary (max 150 words).
        """

    try:
        response = mistral_client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=prompt)]
        )
        news_report = limit_words(response.choices[0].message.content)
        print(f"Generated report ({len(news_report.split())} words):\n{news_report}")
    except Exception as e:
        return {"error": str(e)}

    audio = await (generate_dialog_audio(news_report) if debate else generate_v3_audio(news_report))
    return {"report": news_report, "audio_url": audio}