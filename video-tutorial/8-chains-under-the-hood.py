from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv;
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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
    ("system", "You are expert of {subject}, you will give right answers in one line"),
    ("human", "Describe a {animal}, it lives in {place} and eat {food}")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(x))
model = RunnableLambda(lambda x : model.invoke(x))
final_response = RunnableLambda(lambda x : x.content)

chain = RunnableSequence(format_prompt, model, final_response)
# prompt_template = ChatPromptTemplate.from_messages(messages)
# # prompt = prompt_template.invoke({
# #     "animal" : "lion",
# #     "place" : "bengal",
# #     "food" : "meat",
# #     "subject" : "animals"
# # })

# chain = prompt_template | model | StrOutputParser()

response = chain.invoke({
    "animal" : "lion",
    "place" : "bengal",
    "food" : "meat",
    "subject" : "animals"
})
print("------------ Chain basic Example -------------")
print(response)