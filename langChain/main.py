
from google import genai
import certifi
import requests
from google.genai import types
import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
import time
import aiohttp
from functools import partial
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langsmith import Client
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import tool

from tools.weather import get_weather_data


 
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


##################
# Streamlit page config

st.set_page_config(
    page_title="Agentic AI Interface",
    layout="centered"
)

st.title("Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

##################

llm_obj = ChatGoogleGenerativeAI(
    model=model,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=output_tokens
)

search_tool = TavilySearch(max_results=2)
prompt = Client().pull_prompt('hwchase17/react', dangerously_pull_public_prompt=True)

tools = [get_weather_data, search_tool]

# final agent object
agent = create_react_agent(
    llm=llm_obj,
    tools=tools,
    prompt=prompt
)

agent_executer = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

user_query = st.text_input(
    "Enter your query: ",
    placeholder="Example: Find the capital of Washington, DC and get its weather, humidity and chance of rain in %"
)

## Run Agent
if st.button("Run Agent"):
    if user_query:
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executer.invoke({
                    "input": user_query
                })

                st.success("Response Generated")
                st.markdown("### Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query!")