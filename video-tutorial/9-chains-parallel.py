from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv;
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
load_dotenv()



model = ChatOpenAI(model="gpt-4o")

messages= [
    ("system", "You are expert about products, and you will help customer with product features."),
    ("human", "List 3 feature of product {product}")
]
prompt_template = ChatPromptTemplate.from_messages(messages)


def create_pros_list(product_details):
    messages= [
        ("system", "You are expert about products, and you will help customer with product pros."),
        ("human", "List 3 pros of product {product_details}")
    ]
    return ChatPromptTemplate.from_messages(messages)

def create_cons_list(product_details):
    messages= [
        ("system", "You are expert about products, and you will help customer with product cons."),
        ("human", "List 3 cons of product {product_details}")
    ]
    return ChatPromptTemplate.from_messages(messages)

def combine_pros_cons(pros, cons):
    return f"Pros : \n {pros} \n\n\n Cons : \n{cons}"

pros_parallel_chain = (
    RunnableSequence(lambda x : create_pros_list(x), model, StrOutputParser())
)

cons_parallel_chain = (
    RunnableSequence(lambda x : create_cons_list(x), model, StrOutputParser())
)

chain = (
    prompt_template
    |model
    |StrOutputParser()
    | RunnableParallel(branches ={"pros" : pros_parallel_chain, "cons" : cons_parallel_chain})
    | RunnableLambda(lambda x : combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"] ))
)

result = chain.invoke({"product" : " Latest Iphone"})
print("------------ Chain basic Example -------------")
print(result)