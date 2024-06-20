from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

def get_docs_from_web(url):
    loader= WebBaseLoader(url)
    docs=loader.load()
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs= splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding= OpenAIEmbeddings()
    vectorStore= FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    llm= ChatOpenAI(
    temperature=0.4,
    model= "gpt-3.5-turbo-1106"
)

    prompt= ChatPromptTemplate.from_messages(
        [
            ("system","Answer the user's questions based on the context :{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ]
    )

    # chain= prompt | llm
    chain= create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever= vectorStore.as_retriever(search_kwargs={"k":3})
    
    retrieval_chain= create_retrieval_chain(
        retriever,chain
    )
    return retrieval_chain


def process_chat(chain,question,chat_history):
    response= chain.invoke({
    "input":question,
    "chat_history":chat_history,
    })

    return response["answer"]

if __name__=='__main__':
    docs= get_docs_from_web('https://python.langchain.com/v0.1/docs/expression_language/')
    vectorStore= create_db(docs)
    chain= create_chain(vectorStore)


    chat_history = []

    while True:    
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break 
        response= process_chat(chain, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)



