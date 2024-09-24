from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv;
load_dotenv()
# template = "Describe a {animal}, it lives in {place} and eat {food}"

# prompt_template = ChatPromptTemplate.from_template(template)
# prompt = prompt_template.invoke({
#     "animal" : "lion",
#     "place" : "bengal",
#     "food" : "meat"
# })

# print(prompt)

model = ChatOpenAI(model="gpt-4o")

messages= [
    ("system", "You are expert of {subject}, you will give right answers."),
    ("human", "Describe a {animal}, it lives in {place} and eat {food}")
]


prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    "animal" : "lion",
    "place" : "bengal",
    "food" : "meat",
    "subject" : "animals"
})

print(prompt)

response = model.invoke(prompt);
print(response.content)