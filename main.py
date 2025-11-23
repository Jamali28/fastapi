from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

main_agent = Agent(
    name="Python Assistant",
    instructions="You are a helpful assistant that can help with Python code.",
    model=model,
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, Jamali"}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def main(req: ChatMessage):
    result = await Runner.run(
        main_agent,
        req.message
    )
    return {"response": result.final_output}
