from fastapi import FastAPI
from models.main import initiate_agent, generate_answer
from contextlib import asynccontextmanager
import nest_asyncio

nest_asyncio.apply()
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = await initiate_agent()  
    print("Agent initialized:", agent)
    
    yield 

    print("Shutting down agent...")  

app = FastAPI(lifespan=lifespan)

@app.post("/response")
def agent_response(prompt: str):
    result = generate_answer(prompt, agent)
    return result



@app.get("/")
async def root():
    return {"message": "Hello World", "agent": agent}

