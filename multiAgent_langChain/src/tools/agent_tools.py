from google import genai
import certifi
import requests
from google.genai import types
import streamlit as st
import os
from dotenv import load_dotenv
from rich import print as p
from bs4 import BeautifulSoup
import trafilatura
from readability import Document
import re
import asyncio
import time
import aiohttp
from functools import partial
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from langsmith import Client
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import tool



############################# Load ENV Variables
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

model = os.getenv("MODEL_NAME")
temperature = os.getenv("MODEL_TEMPERATURE")
top_k = os.getenv("MODEL_TOP_K")
top_p = os.getenv("MODEL_TOP_P")
output_tokens = os.getenv("MODEL_MAX_OUTPUT_TOKENS")
tavily_key = os.getenv("TAVILY_API_KEY")
#############################

tavily_client = TavilyClient(api_key=tavily_key)



llm_obj = ChatGoogleGenerativeAI(
    model=model,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=output_tokens
)

@tool
def scrape_url(url: str) -> str:
    '''
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    '''
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US, en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        response = requests.get(
            url=url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ──────────────────────────────────────────────────
        # Strategy 1 → trafilatura (BEST for articles/blogs)
        # ──────────────────────────────────────────────────
        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=False
        )

        if extracted and len(extracted.strip()) >= 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 2 → readability
        # ──────────────────────────────────────────────────
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 3 → fallback full page extraction
        # ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:5000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Search the query on the internet
    """
    answers = []
    results = tavily_client.search(query=query, max_results=2)

    for result in results['results']:
        answers.append(
            f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content'][:300]}"
        )
    return "\n----\n".join(answers)