from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from models.prompts import context
import os


load_dotenv()

async def initiate_agent():

    llm = Ollama(model="mistral", request_timeout=30.0)

    parser = LlamaParse(result_type="markdown", api_key=os.getenv("API_KEY"))

    documents = await parser.aload_data("./data/data.pdf")

    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)

    tools = [QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name="FAQ", description="This tool answers questions solely based on the data in the 'data.pdf' document about Jakub Zimny.")
    )]


    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True,context=context)
    return agent

#    while (prompt := input("Enter a prompt (q to quit): ")) != 'q':
#        result = agent.query(prompt)
#        print(result)

def generate_answer(prompt: str, agent: ReActAgent):
    result = agent.query(prompt)
    return result
