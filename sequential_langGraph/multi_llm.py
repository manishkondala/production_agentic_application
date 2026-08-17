import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict
import pydantic

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

class Blog(TypedDict):
    topic: str
    title: str
    brief_summary: str

graph = StateGraph(Blog)

# Nodes
def llm_title(state: Blog) -> Blog:
    topic = state['topic']

    prompt = f"On the following {topic},  provide a catchy title based on the topic"

    answer = llm_obj.invoke(prompt).content

    state['title'] = answer

    return state

def llm_brief_summary(state: Blog) -> Blog:
    topic = state['topic']
    title = state['title']

    prompt = f"using the {topic} and {title}, generate a 2 line summary that will hook anyone who reads it. Be rude as possible"

    brief_summary = llm_obj.invoke(prompt).content

    state['brief_summary'] = brief_summary

    return state

graph.add_node('llm_title', llm_title)
graph.add_node('llm_brief_summary', llm_brief_summary)

# Edges
graph.add_edge(START, 'llm_title')
graph.add_edge('llm_title', 'llm_brief_summary')
graph.add_edge('llm_brief_summary', END)

workflow = graph.compile()

input_state = {'topic': 'AI in Healthcare'}
final_state = workflow.invoke(input_state)
print(final_state['topic'])
print(final_state['title'])
print(final_state['brief_summary'])