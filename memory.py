from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

UPSTASH_REDIS_REST_URL="https://patient-mudfish-45434.upstash.io"
UPSTASH_REDIS_REST_TOKEN="AbF6AAIncDEzMjM2YjA0YWNmODA0MDlmYWM4ZWE5OWNhMGUzYzkzYXAxNDU0MzQ"

history=UpstashRedisChatMessageHistory(
    url= UPSTASH_REDIS_REST_URL,
    token=UPSTASH_REDIS_REST_TOKEN,
    session_id="chat1",
    ttl=0
)

model= ChatOpenAI(
    model= "gpt-3.5-turbo",
    temperature=0.7,
)

prompt= ChatPromptTemplate.from_messages([
    ("system","You are a friendly AI Assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])

memory= ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

# chain= memory | prompt | model

chain=LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# msg1= {
#     "input":"My name is Aun"
# }

# response1= chain.invoke(msg1)
# print(response1)

msg2= {
    "input":"What is my name ?"
}

response2= chain.invoke(msg2)
print(response2)