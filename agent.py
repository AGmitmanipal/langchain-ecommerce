from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langsmith import traceable
from dotenv import load_dotenv
import os


load_dotenv()

os.environ.get("GOOGLE_API_KEY")

MAX_ITERATIONS = 10

@tool
def get_product_price(product: str)->float:
    """This tool is to get the price of the product"""
    prices = {"Pen": 100.23, "Paper": 90.09, "Page": 20.78}
    return prices.get(product, 0)

@tool
def get_discount(price: float, discount_tier: str)->float:
    """This tool is to get the discount for the product"""
    discounts = {"silver": 0.05, "gold": 0.1, "platinum": 0.3}
    discount = discounts.get(discount_tier, 0)
    return round(price*(1-discount), 2)

@traceable(name="Langchain Agent")
def run_agent(query: str):
    tools = [get_product_price, get_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google-genai", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    messages = [
            SystemMessage(
    content="""
You are a helpful AI agent that can use tools to answer user queries.

You have access to the following tools:
1. get_product_price(product: str) -> float
2. get_discount(price: float, discount_tier: str) -> float

Rules:
- Always use tools when required instead of guessing.
- If the user asks for a product price, call get_product_price first.
- If a discount is requested, use the result from get_product_price and then call get_discount.
- Perform operations step-by-step using tools.
- Do NOT assume values — always rely on tool outputs.
- Return the final answer clearly.
            """),
            HumanMessage(content=query)
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration: {iteration}")

        response = llm_with_tools.invoke(messages)

        tool_calls = response.tool_calls

        if not tool_calls:
            print(f"AI message: {response}")
            return response.content
        
        tool_call = tool_calls[0]

        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args")


        tool = tools_dict.get(tool_name)
        if tool is None:
            raise ValueError("Tool {tool_name} not found")
        
        observation = tool.invoke(tool_args)

        print(f"[Tool Result] {observation}")
        
        messages.append(response)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))



if __name__ == "__main__":
    print("Langchain Agent Loop")
    run_agent("Get me the price of the Pen after applying a silver discount")
