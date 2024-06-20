from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm= ChatOpenAI(
    temperature=0.7,
    model= "gpt-3.5-turbo-1106"
)

prompt= ChatPromptTemplate.from_messages(
    [
        ("system","Tell me a joke about the follwoing subject."),
        ("human","{input}")
    ]
)

chain= prompt | llm 

response= chain.invoke({"input":"sad"})
print(type(response.content)) 