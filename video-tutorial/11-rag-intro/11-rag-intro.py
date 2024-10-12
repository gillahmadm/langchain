from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv;
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch
load_dotenv()

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 1. Read the complete document

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,  "Dubai.txt")
persistent_directory = os.path.join(current_dir, "chroma_db")

embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
)

if not os.path.exists(persistent_directory):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} doesn't exist"
        )

    loader = TextLoader(file_path)
    document = loader.load()
    # print(document)

    # 2. Create chunks from this document
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # 7000 - 1000 (1-1000), 900-1900
    chunks = text_splitter.split_documents(document)

    # print("\n---------- Chunks Info-----------")
    # print(f" Chunk count :  {len(chunks)}")

    # 3. Create embedding and save in the vector DB


    db = Chroma.from_documents(
            chunks, embedding_model, persist_directory=persistent_directory)
    num_documents = db._collection.count()  # Or use another method depending on your Chroma version
    print(f"Number of documents in the collection: {num_documents}")
    print("--------- Step 3 done ---------")
else:
    print(" Chroma DB, aleady with requirement information")
# # 4. Get the query from the user. 
db =Chroma(persist_directory=persistent_directory, 
            embedding_function=embedding_model)
num_documents = db._collection.count()  # Or use another method depending on your Chroma version
print(f"Number of documents in the collection: {num_documents}")
query = "Account of oil in Dubai GDP?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)

# # 5. convert embedding of user query.
# # 6. Retrieve relevant chunks.

relevant_chunks = retriever.invoke(query)

# print("--------Relevant chunks ---------")
# for i, chunk in enumerate(relevant_chunks, 1):
#     print(f"Chunk : {i} : \n {chunk.page_content}\n")

# # Display the relevant results with metadata
chunks_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

model = ChatOpenAI(model="gpt-4o")
rag_system_prompt = (
    "You are expert in answering question."
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say I don't know the answer."
    "Give answer in one line"
    "\n\n"
    "{relevant_chunks}"
)

formatted_rag_prompt = rag_system_prompt.format(relevant_chunks=chunks_text)
messages= [
    ("system", formatted_rag_prompt),
    ("human", query)
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
})
response = model.invoke(prompt);
print(response.content)



