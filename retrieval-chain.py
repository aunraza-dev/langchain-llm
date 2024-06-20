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

def get_docs_from_web(url):
    loader= WebBaseLoader(url)
    docs=loader.load()
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs= splitter.split_documents(docs)
    print(len(splitDocs))
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

    prompt= ChatPromptTemplate.from_template("""
    Answer the User's Question:
    Context: {context}
    Question: {input}
    """)

    # chain= prompt | llm
    chain= create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )


    retriever= vectorStore.as_retriever(search_kwargs={"k":2})
    
    retrieval_chain= create_retrieval_chain(
        retriever,chain
    )
    return retrieval_chain


docs= get_docs_from_web('https://python.langchain.com/v0.1/docs/expression_language/')
vectorStore= create_db(docs)
chain= create_chain(vectorStore)


response= chain.invoke({
    "input":"What is LCEL?"
    })
print(response)