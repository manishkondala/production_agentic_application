### Weather using weatherstack.com
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def get_weather_data(city: str) -> str:
    """
    Get current weather information for a city\n
    Get the humidity for the city\n
    Get the % of rain for the city\n
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={os.getenv('WEATHERSTACK_API_KEY')}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather for data for {city}"

    return (
        f"{city} weather: {data}, humiditiy: {data}, chance of rain %: {data}\n"
    )