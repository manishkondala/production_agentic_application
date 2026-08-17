import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, List, Annotated, Literal
from pydantic import BaseModel, Field
import operator
import textwrap

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

class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    issue_type: Literal['UX', 'Performance', 'Bug', 'Support', 'Other'] = Field(description='The category of issue mentioned in the review')
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(description='The emotional tone expressed by the user')
    urgency: Literal['low', 'medium', 'high'] = Field(description='How urgent or critical the issue appears to be')

sentiment_llm = llm_obj.with_structured_output(SentimentSchema)
diagnosis_llm = llm_obj.with_structured_output(DiagnosisSchema)

def findSentiment(state: ReviewState):
    prompt = f"Take the customer review comment and find out the sentiment {state['review']}"
    review_sentiment = sentiment_llm.invoke(prompt).sentiment

    return {
        'sentiment': review_sentiment
    }

def runDiagnosis(state: ReviewState):
    prompt = f"Take the customer review comment and find out the diagnosis {state['review']}"
    diagnosis = diagnosis_llm.invoke(prompt)

    return {
        'diagnosis': diagnosis.model_dump()
    }

def negativeResponse(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"You are a support assistant. \n\n The user had a {diagnosis['issue_type']} issue, souded {diagnosis['tone']} and marked urgency as {diagnosis['urgency']} \n\n Write an empathatic, helpful resolution message."

    result = llm_obj.invoke(prompt).content
    return {
        'response': result
    }

def positiveRespone(state: ReviewState):
    prompt = f"Write a warm thank-you message in response to this review: \n\n{state['review']}\n\n Also, kindly ask the user to leave feedback on our website!"
    response = llm_obj.invoke(prompt).content
    return {
        'response': response
    }

def check_condition(state: ReviewState) -> Literal["positiveRespone", "runDiagnosis"]:
    if state['sentiment'] == "positive":
        return "positiveRespone"
    else:
        return "runDiagnosis"

graph = StateGraph(ReviewState)

graph.add_node('findSentiment', findSentiment)
graph.add_node('runDiagnosis', runDiagnosis)
graph.add_node('negativeResponse', negativeResponse)
graph.add_node('positiveRespone', positiveRespone)

graph.add_edge(START, 'findSentiment')

graph.add_conditional_edges('findSentiment', check_condition)

graph.add_edge('negativeResponse', END)

graph.add_edge('runDiagnosis', 'negativeResponse')
graph.add_edge('positiveRespone', END)

workflow = graph.compile()

intial_state={
    'review': "I dont like my HughesNet service, my video always buffers and I always stream at SD. I have wifi issues where my wifi drops and shows packet loss. I am a HughesNet employee"
}
result = workflow.invoke(intial_state)


WIDTH = 84


def rule(char: str = "─") -> None:
    print(char * WIDTH)


def section(title: str, body: str) -> None:
    print()
    rule()
    print(f"  {title.upper()}")
    rule()
    for para in str(body).split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=WIDTH,
                                initial_indent="  ", subsequent_indent="  "))
        else:
            print()


print()
rule("=")
print("REVIEW TRIAGE".center(WIDTH))
rule("=")

section("Customer Review", result["review"])

# Verdict
print()
rule()
print("  VERDICT")
rule()
print(f"  Sentiment    {result['sentiment']}")

diagnosis = result.get("diagnosis")
if diagnosis:
    print(f"  Issue type   {diagnosis['issue_type']}")
    print(f"  Tone         {diagnosis['tone']}")
    print(f"  Urgency      {diagnosis['urgency']}")
    print(f"  Branch       negative → runDiagnosis → negativeResponse")
else:
    print(f"  Branch       positive → positiveRespone")

section("Generated Reply", result["response"])

rule("=")
print()