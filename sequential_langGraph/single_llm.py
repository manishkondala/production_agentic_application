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

# Define State
class TemperatureState(TypedDict):
    temperature_celcius: float
    temperature_farenhite: float
    label_weather: str

# Create Graph object with StateGraph
graph = StateGraph(TemperatureState)

# Add nodes 
def convert_temperature(state: TemperatureState) -> TemperatureState:
    celcius = state['temperature_celcius']
    farenhite = (celcius * 9/5) + 32
    state['temperature_farenhite'] = round(farenhite, 2)

    return state

def label_weather(state: TemperatureState) -> TemperatureState:
    celcius = state['temperature_celcius']
    farenhite = state['temperature_farenhite']

    if 20 <= celcius <= 50:
        state['label_weather'] = 'Snow'
    elif 50 < celcius <= 70:
        state['label_weather'] = 'Normal'
    elif 70 < celcius <= 100:
        state['label_weather'] = 'Hot Hot Hot!!!'
    else:
        state['label_weather'] = 'Cannot Compute Weather - Unrealistic Weather Conditions'

    return state

graph.add_node('convert_temperature', convert_temperature)
graph.add_node('label_weather', label_weather)

# Add Edge to Graph
graph.add_edge(START, 'convert_temperature')
graph.add_edge('convert_temperature', 'label_weather')
graph.add_edge('label_weather', END)

# Graph COmpilation
workflow = graph.compile()

# Graph Execution
input_state = {'temperature_celcius': 1}
final_state = workflow.invoke(input_state)
print(final_state)

# Visualize Graph
from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())