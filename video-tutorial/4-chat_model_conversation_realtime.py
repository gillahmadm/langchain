from dotenv import load_dotenv;
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

messages=[
    SystemMessage(content="You are friendly AI assitant.")
]
while True:
    query = input("User : ")
    if(query.lower()=='stop'):
        break;

    messages.append(HumanMessage(content=query));


    response = model.invoke(messages)
    ai_message = response.content;
    messages.append(AIMessage(content=ai_message));

    print(f"AI Message : {ai_message}")

print("Chat history :")
print(messages)

