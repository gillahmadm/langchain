from dotenv import load_dotenv;
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")


response = model.invoke("Is pizza Healthy?, Should be one line response")

print("OpenAI response :")
print(response.content)


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


response = model.invoke("Is pizza Healthy?, Should be one line response")

print("Google response :")
print(response.content)
# ========================
# print("Chat response :")
# print(response.content)


# System Message
# Human Message
# AI message
# messages = [
#     SystemMessage(content="Give to the point answer, you answer should be consist of only 1 line"),
#     HumanMessage(content="How long is flight from London to New york?"),
#     AIMessage(content="7 hours"),
#     HumanMessage(content="How long is flight from London to Dubai?")
# ]

# response = model.invoke(messages)
# print(f"response from AI : {response.content}")