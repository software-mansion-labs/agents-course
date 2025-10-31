from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Returns the weather conditions in a specified city.

    Args:
        city (str): city to check.

    Returns:
        str: description of the weather.
    """
    return f"It's rainy in {city}."


checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model="openai:gpt-4o", tools=[get_weather], checkpointer=checkpointer
)

with open("graph4.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())

while True:
    query = input("query: ")

    new_state = agent.invoke(
        {"messages": [HumanMessage(query)]},
        config,
    )
    answer = new_state["messages"][-1].content
    print("answer:", answer)
