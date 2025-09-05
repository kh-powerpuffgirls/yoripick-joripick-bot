from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
import uuid
import asyncio
from typing import AsyncGenerator
import os
import oracledb
from dotenv import load_dotenv
load_dotenv()

user = os.getenv("ORACLE_USER")
password = os.getenv("ORACLE_PASSWORD")
host = os.getenv("ORACLE_HOST")
port = os.getenv("ORACLE_PORT")
sid = os.getenv("ORACLE_SID")

dsn = f"{host}:{port}/{sid}"
    
def load_documents():
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cursor = conn.cursor()

    cursor.execute("SELECT question, answer, link_url FROM CHATBOT_SCRIPT")
    rows = cursor.fetchall()
    documents = []
    for question, answer, link_url in rows:
        content = f"Question:{question}Answer:{answer}Link:{link_url}"
        documents.append(Document(page_content=content))
    
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

prompt = ChatPromptTemplate.from_messages(
    [("system", """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Answer in Korean."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Answer the following question.
Question: {query}
Retrieved documents: {retrieved_docs}

**Rules:**
1. Always respond in JSON format.
2. JSON structure:
{{
    "content": "The content of the answer",
    "button": {{"linkUrl": "Link if available"}}  // if no link, set button to null
}}
3. Do not include any text outside of the JSON.
"""
)]
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

store = {}
def get_history_from_db(user_no):
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM CHATBOT WHERE user_no=:1 ORDER BY created_at",
        (user_no,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if user_no not in store:
        store[user_no] = ChatMessageHistory()
    
    for role, content in rows:
        if role == "USER":
            store[user_no].add_user_message(content)
        else:
            store[user_no].add_ai_message(content)
    
    return store[user_no]

def get_history(params) -> ChatMessageHistory:
    if isinstance(params, dict):
        session_id = params["session_id"]
        user_no = params["user_no"]
        if user_no: get_history_from_db(user_no)
        
    else:
        session_id = params
        if isinstance(session_id, int): get_history_from_db(session_id)
    
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    return store[session_id]

def set_history(user_no:str, role:str, content:str):
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO CHATBOT (chatbot_no, user_no, role, content)" +
                   "VALUES (SEQ_CHATBOT_NO.NEXTVAL, :1, :2, :3)",
                  (user_no, role, content))
    conn.commit()
    cursor.close()
    conn.close()

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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(request:Request, chat_request: ChatRequest):
    user_no = request.cookies.get("user_no")
    session_id = request.cookies.get("session_id")
    # user_no = 1

    if user_no is None and session_id is None:
        session_id = str(uuid.uuid4())

    elif user_no is not None:
        session_id = user_no

    try:
        retrieved_docs = retriever.invoke(chat_request.question)
        
        if user_no:
            set_history(user_no, "USER", chat_request.question)
        params = {"session_id": session_id, "user_no": user_no}
        get_history(params).add_user_message(chat_request.question)

        async def response_generator() -> AsyncGenerator[str, None]:
            assistant_response = ""
            for chunk in chain_with_history.stream(
                input={
                    "query":chat_request.question,
                    "retrieved_docs":retrieved_docs,
                },
                config={"configurable":params},
            ):
                assistant_response += chunk
                yield chunk
                await asyncio.sleep(0)
                
            if user_no:
                set_history(user_no, "BOT", assistant_response)
            get_history(params).add_ai_message(assistant_response)

        streaming = StreamingResponse(response_generator(), media_type="application/json")
        streaming.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=False
        )
        return streaming

    except Exception as e:
        print("error occured", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{user_no}")
async def chat(user_no: int, request:Request):
    session_id = request.cookies.get("session_id")
    
    if user_no in store:
        del store[user_no]
    if session_id in store:
        del store[session_id]

    return {"message": "Chat session deleted successfully"}

# uvicorn faqService:app --host 0.0.0.0 --port 8080 --reload