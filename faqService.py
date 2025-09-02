from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
import uuid
import asyncio
from typing import AsyncGenerator
import os
import cx_Oracle
from dotenv import load_dotenv
load_dotenv()

user = os.getenv("ORACLE_USER")
password = os.getenv("ORACLE_PASSWORD")
host = os.getenv("ORACLE_HOST")
port = os.getenv("ORACLE_PORT")
sid = os.getenv("ORACLE_SID")

def load_documents():
    dsn = cx_Oracle.makedsn(host, port, sid)
    conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
    cursor = conn.cursor()

    cursor.execute("SELECT QUESTION, ANSWER FROM CHATBOT_SCRIPT")

    rows = cursor.fetchall()
    documents = [Document(page_content=row[0]) for row in rows]

    cursor.close()
    conn.close()
    return documents

documents = load_documents()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = ConversationBufferMemory(return_messages=True)
# 영구저장은 db에 직접 해야함

prompt = ChatPromptTemplate.from_messages(
    [("system", """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Answer in Korean."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """#Question:\n{query}\n#Retrieved_docs:\n{retrieved_docs}\n#Answer:""")]
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

store = {}
# user_no별로 대화기록 가져오기
def get_history(user_no:str):
    if(user_no not in store):
        store[user_no] = ChatMessageHistory()
    return store[user_no]

query_extractor = RunnableLambda(lambda x : x["query"])

chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="query",
    history_messages_key="chat_history"
)

class ChatRequest(BaseModel):
    question: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(request:Request, response:Response, chat_request: ChatRequest):
    user_no = request.cookies.get("user_no")
    if user_no is None:
        user_no = str(uuid.uuid4()) # 로그인하지 않은 사용자에게 랜덤 session id 부여
        response.set_cookie(key="user_no", value=user_no, httponly=True)

    try:
        retrieved_docs = retriever.invoke(chat_request.question)

        async def response_generator() -> AsyncGenerator[str, None]:
            for chunk in chain_with_history.stream(
                input={
                    "query":chat_request.question,
                    "retrieved_docs":retrieved_docs
                },
                config={"configurable":{"session_id":user_no}},
            ):
                yield chunk
                await asyncio.sleep(0)

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        print("error occured", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
# uvicorn faqService:app --host 0.0.0.0 --port 8080 --reload