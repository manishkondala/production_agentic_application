import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
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

class PostState(TypedDict):

    topic: str
    post: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    post_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

class PostEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the Facebook post.")

evaluator_llm = llm_obj.with_structured_output(PostEvaluation)

def generate_post(state: PostState):

    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Facebook influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious Facebook post on the topic: "{state['topic']}".

        Rules:
        - Do NOT use question-answer format.
        - Max 500 characters.
        - Use observational humor, irony, sarcasm, or cultural references.
        - Think in meme logic, punchlines, or relatable takes.
        - Use simple, day to day english
        """)
    ]

    # send generator_llm
    response = llm_obj.invoke(messages).content

    # return response
    return {'post': response, 'post_history': [response]}

def evaluate_post(state: PostState):

    # prompt
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Facebook critic. You evaluate posts based on humor, originality, virality, and post format."),
        HumanMessage(content=f"""
        Evaluate the following Facebook post:

        Post: "{state['post']}"

        Use the criteria below to evaluate the post:

        1. Originality – Is this fresh, or have you seen it a hundred times before?  
        2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
        3. Punchiness – Is it short, sharp, and scroll-stopping?  
        4. Virality Potential – Would people share, react, or comment on it?  
        5. Format – Is it a well-formed Facebook post (not a setup-punchline joke, not a Q&A joke, and under 500 characters)?

        Auto-reject if:
        - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
        - It exceeds 500 characters
        - It reads like a traditional setup-punchline joke
        - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

        ### Respond ONLY in structured format:
        - evaluation: "approved" or "needs_improvement"  
        - feedback: One paragraph explaining the strengths and weaknesses 
        """)
    ]

    response = evaluator_llm.invoke(messages)

    return {'evaluation':response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}

def optimize_post(state: PostState):

    messages = [
        SystemMessage(content="You punch up Facebook posts for virality and humor based on given feedback."),
        HumanMessage(content=f"""
        Improve the Facebook post based on this feedback:
        "{state['feedback']}"

        Topic: "{state['topic']}"
        Original Post:
        {state['post']}

        Re-write it as a short, viral-worthy Facebook post. Avoid Q&A style and stay under 500 characters.
        """)
    ]

    response = llm_obj.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'post': response, 'iteration': iteration, 'post_history': [response]}

def route_evaluation(state: PostState):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

graph = StateGraph(PostState)

# add nodes
graph.add_node('generate', generate_post)
graph.add_node('evaluate', evaluate_post)
graph.add_node('optimize', optimize_post)


# add edges
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')
graph.add_conditional_edges('evaluate',route_evaluation,{'approved': END, 'needs_improvement': 'optimize'})
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

initial_state = {
    "topic": "agentic AI",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)


WIDTH = 84


def rule(char: str = "─") -> None:
    print(char * WIDTH)


def block(body: str, indent: str = "  ") -> None:
    for para in str(body).split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=WIDTH,
                                initial_indent=indent, subsequent_indent=indent))
        else:
            print()


print()
rule("=")
print(f"POST GENERATOR  ·  topic: {result['topic']}".center(WIDTH))
rule("=")

# Draft-by-draft history. post_history[i] was critiqued by feedback_history[i].
posts = result.get("post_history", [])
feedbacks = result.get("feedback_history", [])

for i, post in enumerate(posts):
    print()
    rule()
    print(f"  DRAFT {i + 1}" + ("  (initial)" if i == 0 else "  (revised)"))
    rule()
    block(post)

    if i < len(feedbacks):
        print()
        print("  ── critic ──")
        block(feedbacks[i], indent="    ")

print()
rule("=")
print("  FINAL")
rule("=")
print(f"  Verdict      {result['evaluation']}")
print(f"  Iterations   {result['iteration']} of {result['max_iteration']}")
print(f"  Drafts       {len(posts)}")
print()
block(result["post"])
rule("=")
print()