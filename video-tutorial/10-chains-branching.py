from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv;
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch
load_dotenv()



model = ChatOpenAI(model="gpt-4o")

messages= [
    ("system", "You are AI classification assitant, you will respond back with only one word from these, positive, negative and neutral"),
    ("human", "Classify the sentiment of feedback : {feedback}")
]
prompt_template = ChatPromptTemplate.from_messages(messages)



positive_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are AI assitant"),
    ("human", "Generate a response for this positive feedback : {feedback}")
])

negative_prompt_template = ChatPromptTemplate.from_messages([
   ("system", "You are AI assitant"),
    ("human", "Generate a response for this negative feedback : {feedback}")
])

neutral_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are AI assitant"),
    ("human", "Generate a response for this neutral feedback : {feedback}")
])

branches = RunnableBranch(
    (
        lambda x : "Positive" in x,
        positive_prompt_template | model | StrOutputParser()
    ),
    (
        lambda x : "Negative" in x,
        negative_prompt_template | model | StrOutputParser()
    ),
    neutral_prompt_template | model | StrOutputParser()
)
classification_chain = prompt_template | model | StrOutputParser()
chain = classification_chain | branches


# chain = (
#     prompt_template
#     |model
#     |StrOutputParser()
#     | RunnableParallel(branches ={"pros" : pros_parallel_chain, "cons" : cons_parallel_chain})
#     | RunnableLambda(lambda x : combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"] ))
# )

result = chain.invoke({"feedback" : "This indian curry was terrible."})
print("------------ Chain basic Example -------------")
print(result)