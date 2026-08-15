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
# import Document
import re
import asyncio
import time
import aiohttp
from functools import partial
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.agent_tools import web_search, scrape_url

load_dotenv()

model = os.getenv("MODEL_NAME")
temperature = os.getenv("MODEL_TEMPERATURE")
top_k = os.getenv("MODEL_TOP_K")
top_p = os.getenv("MODEL_TOP_P")
output_tokens = os.getenv("MODEL_MAX_OUTPUT_TOKENS")
tavily_key = os.getenv("TAVILY_API_KEY")

llm_obj = ChatGoogleGenerativeAI(
    model=model,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=output_tokens
)

# First Agent - Search Agent
def build_search_agent():
    return create_agent(
        model=llm_obj,
        tools=[web_search],
    )

# Second Agent - Scrape Agent
def reader_agent():
    return create_agent(
        model=llm_obj,
        tools=[scrape_url]
    )

# Writer Chain (LangChain LCL)
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic.
    
    Topic: {topic}

    Research Gathered:
    {research}

    Structure the report as:
    - Introduction
    - Key Findings
    - Conclusion
    - Sources

    Be detailed, factual and professional.""")
])

writer_chain = writer_prompt | llm_obj | StrOutputParser()

# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ..."""),
])

critic_chain = critic_prompt | llm_obj | StrOutputParser()