from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field

llm= ChatOpenAI(
    temperature=0.7,
    model= "gpt-3.5-turbo-1106"
)

def get_docs_from_web(url):
    loader= WebBaseLoader(url)
    docs=loader.load()
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs= splitter.split_documents(docs)
    return splitDocs

def call_JSON_OP():
    url = input("Enter the URL of the product page: ")
    docs = get_docs_from_web(url)

    prompt= ChatPromptTemplate.from_messages(
        [
            ("system","Extract information from the following phrase. Formatting Instructions: {format_instructions}"),
            ("human","{phrase}")
        ]
    )

    class Product(BaseModel):
        title: str= Field(description="the title of the product")
        price: float= Field(description="the price of the product")
        color: str= Field(description="the color of the product")

    parser=JsonOutputParser(pydantic_object=Product) 
    chain= prompt | llm | parser
    return chain.invoke({
        "phrase":docs,
        "format_instructions":parser.get_format_instructions()
                         })

print(call_JSON_OP())
