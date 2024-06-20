from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field

llm= ChatOpenAI(
    temperature=0.7,
    model= "gpt-3.5-turbo-1106"
)

def call_string_OP():
    prompt= ChatPromptTemplate.from_messages(
        [
            ("system","Tell me a joke about the follwoing subject."),
            ("human","{input}")
        ]
    )

def call_list_OP():
    prompt= ChatPromptTemplate.from_messages(
        [
            ("system","Generate a list of 10 synonyms of the following word. Return the results in a comma seperated list."),
            ("human","{input}")
        ]
    )

def call_JSON_OP():
    prompt= ChatPromptTemplate.from_messages(
        [
            ("system","Extract information from the following phrase. Formatting Instructions: {format_instructions}"),
            ("human","{phrase}")
        ]
    )

    class Person(BaseModel):
        name: str= Field(description="the name of person")
        age: int= Field(description="the age of person")

    # parser= StrOutputParser()
    # parser= CommaSeparatedListOutputParser()
    parser=JsonOutputParser(pydantic_object=Person) 
    chain= prompt | llm | parser
    return chain.invoke({
        "phrase":"Aun is 22 years old",
        "format_instructions":parser.get_format_instructions()
                         })

# print(call_string_OP())
# print(call_list_OP())
print(call_JSON_OP())