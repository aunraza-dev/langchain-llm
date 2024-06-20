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
        ("system","Generate a list of 10 synonyms of the following word. Return the results in a comma seperated list."),
        ("human","{input}")
    ]
)

chain= prompt | llm 

response= chain.invoke({"input":"sad"})
print(type(response.content))