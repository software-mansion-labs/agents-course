from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


ITERATION_LIMIT = 5
load_dotenv()
model = init_chat_model("openai:gpt-4o")


def ask_llm(state: State) -> State:
    user_query = input("query: ")
    user_message = HumanMessage(content=user_query)
    answer_message: AIMessage = model.invoke(state["messages"] + [user_message])
    print("answer: ", answer_message.content)

    return {
        "messages": [user_message, answer_message],
        "iteration": state["iteration"] + 1,
    }


graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)

graph.add_edge(START, "ask_llm")
graph.add_conditional_edges(
    "ask_llm",
    lambda state: state["iteration"] < ITERATION_LIMIT,
    {
        True: "ask_llm",
        False: END,
    },
)

workflow = graph.compile()

with open("graph3.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

workflow.invoke({"iteration": 0}, {"recursion_limit": 100})
