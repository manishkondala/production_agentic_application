import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, List, Annotated
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

class EssayEval(TypedDict):
    essay: str
    language: str
    analysis: str
    clarity: str
    overall: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)

struct_llm = llm_obj.with_structured_output(EvaluationSchema)

graph = StateGraph(EssayEval)


def eval_analysis(state: EssayEval) -> EssayEval:
    prompt = f'Evaluate the depth of the analysis of the following essay and provide a short feedback and assign a score out of 10 related to the topic analysis in the essay \n {state['essay']}'
    output = struct_llm.invoke(prompt)

    return {'analysis': output.feedback, 'individual_score': [output.score]}

def eval_language(state: EssayEval) -> EssayEval:
    prompt = f'Evaluate the language quality of the following essay and provide a short feedback and assign a score out of 10 \n {state['essay']}'
    output = struct_llm.invoke(prompt)

    return {'language': output.feedback, 'individual_score': [output.score]}

def eval_thought(state: EssayEval) -> EssayEval:
    prompt = f'Evaluate the clarity of thought of the following essay and provide a short feedback and assign a score out of 10 related to the thought in the essay \n {state['essay']}'
    output = struct_llm.invoke(prompt)

    return {'clarity': output.feedback, 'individual_score': [output.score]}
    
def final_eval(state: EssayEval) -> EssayEval:
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language"]} \n depth of analysis feedback - {state["analysis"]} \n thought analysis feedback - {state["clarity"]}'
    overall_feedback = llm_obj.invoke(prompt).content

    avg_score = sum(state['individual_score']) / len(state['individual_score'])

    return {'overall': overall_feedback, 'avg_score': avg_score}


graph.add_node('eval_analysis', eval_analysis)
graph.add_node('eval_language', eval_language)
graph.add_node('eval_thought', eval_thought)
graph.add_node('final_eval', final_eval)

graph.add_edge(START, 'eval_analysis')
graph.add_edge(START, 'eval_language')
graph.add_edge(START, 'eval_thought')

graph.add_edge(['eval_analysis', 'eval_language', 'eval_thought'], 'final_eval')

graph.add_edge('final_eval', END)

workflow = graph.compile()

essay = """The Storyteller in the Machine: On Narrative-Optimized AI

For most of the last decade, progress in artificial intelligence has been measured in leaderboards. Models are ranked by how many competition mathematics problems they solve, how much code compiles on the first attempt, how reliably they recall obscure facts under adversarial questioning. These are legible metrics, and legibility is seductive. But a quieter development has begun to complicate this picture: the emergence of models tuned less for correctness than for craft — systems built to write, to sustain a voice, to hold a story together across tens of thousands of words. Whether this represents a genuine branch in the evolutionary tree or merely a marketing distinction is one of the more interesting open questions in the field.

The argument for narrative specialization rests on a simple observation: the qualities that make prose good are largely orthogonal to the qualities that make an answer correct. A model can be flawless on a reasoning benchmark and still produce fiction that is tonally flat, structurally inert, and populated by characters who all speak in the same faintly corporate register. Long-form writing demands consistency of voice, deliberate pacing, restraint about when to withhold information, and an ear for rhythm at the sentence level. None of these are captured by the evaluations that dominate the field, which means none of them are directly optimized for. What is not measured is not improved.

There is also a structural case. Sustained narrative is one of the few tasks that genuinely stresses long-context reasoning in a way that is easy to verify by reading. A model writing chapter twenty must remember what it established in chapter three — not merely retrieve it when asked, but honor it unprompted, in the shape of a scene. This is a meaningfully different capability from retrieval accuracy on a synthetic needle-in-a-haystack test, and it may be that optimizing for it produces gains that generalize well beyond fiction.

The objections are substantial. The most serious is that "creative writing ability" resists definition, and what resists definition resists honest measurement. Aesthetic judgment varies across readers, cultures, and genres; a model tuned to a particular notion of good prose may simply be tuned to the taste of whoever assembled its preference data. This raises the uncomfortable possibility that narrative specialization amounts to homogenization — a great many outputs converging on a single house style, confidently mistaken for quality. A field that cannot agree on what it is optimizing for cannot know whether it is improving.

A second objection is economic rather than technical. Maintaining a distinct model family is expensive, and the history of machine learning is largely a history of specialized systems being absorbed by general ones. Separate models for translation, summarization, and classification were all eventually subsumed. If capability at writing turns out to be a byproduct of scale and general post-training rather than a distinct axis, then narrative-tuned variants are a transitional artifact, useful now and obsolete within a few generations.

The strongest version of the case for specialization does not actually depend on winning that argument. It depends instead on the claim that the *interface* of creative work is different — that a writer collaborating with a model wants something other than a correct answer. They want a collaborator that offers three bad options and one surprising one, that pushes back on a premise, that does not resolve ambiguity prematurely because ambiguity is the point. This is a difference in disposition rather than raw capability, and disposition is exactly what post-training shapes. On this reading, a narrative model is not a smarter model but a differently mannered one, and manner may be the thing that matters.

What follows is a genuine test of the field's maturity. If the industry can develop evaluations for craft that are as rigorous as its evaluations for correctness — rejecting both the pretense that quality is unmeasurable and the pretense that it reduces to a single score — then specialization will have earned its place. If it cannot, narrative models will remain a matter of vibes and vendor claims, and readers will be left to adjudicate on their own. That may be the honest outcome. Literature has never had a benchmark, and it has managed."""

initial_state = {
    'essay': essay
}
output_state = workflow.invoke(initial_state)


WIDTH = 84

def section(title: str, body: str) -> None:
    print()
    print("─" * WIDTH)
    print(f"  {title.upper()}")
    print("─" * WIDTH)
    for para in str(body).split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=WIDTH, initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()


print()
print("=" * WIDTH)
print("ESSAY EVALUATION".center(WIDTH))
print("=" * WIDTH)

section("Depth of Analysis", output_state["analysis"])
section("Language Quality",  output_state["language"])
section("Clarity of Thought", output_state["clarity"])
section("Overall Feedback",  output_state["overall"])

print()
print("─" * WIDTH)
print("  SCORES")
print("─" * WIDTH)
print(f"  Individual   {output_state['individual_score']}")
print(f"  Average      {output_state['avg_score']:.2f} / 10")
print("=" * WIDTH)
print()