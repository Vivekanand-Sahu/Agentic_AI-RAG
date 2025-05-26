from typing import Type
import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import argparse
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

from typing import Type, Dict, List, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
#from get_embedding_function import get_embedding_function
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


from typing import Type, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from typing import Type, Dict, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import numpy as np
from datetime import datetime

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):

    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

# ---------------------------------------------------------------------------------


# Creating tool for uploading data




class PDF_to_vector_db_converter_INPUT(BaseModel):
    """Input schema for MyCustomTool."""
    DATA_PATH: str = Field(
        ...,
        description="Path of the folder where the input files are stored"
    )

    CHROMA_PATH: str = Field(
        ...,
        description="Path of the folder where the chroma db exists"
    )

class PDF_to_vector_db_converter_TOOL(BaseTool):
    name: str = "PDF to vector db converter"
    description: str = (
        "Reads the pdf document, extracts the contents, and loads the information to vector database"
    )
    args_schema: Type[BaseModel] = PDF_to_vector_db_converter_INPUT

    def _run(self, DATA_PATH: str, CHROMA_PATH: str) -> str:

        output = 'The execution failed!'


        def main():
            # Check if the database should be cleared (using the --clear flag).
            # parser = argparse.ArgumentParser()
            # parser.add_argument("--reset", action="store_true", help="Reset the database.")
            # args = parser.parse_args()

            # if args.reset:
            #     print("✨ Clearing Database")
            #     clear_database()

            # Create (or update) the data store.
            documents = load_documents()
            chunks = split_documents(documents)
            return add_to_chroma(chunks)

        def get_embedding_function():
            # embeddings = BedrockEmbeddings(
            #     credentials_profile_name="default", region_name="us-east-1"
            # )
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            return embeddings

        def load_documents():
            document_loader = PyPDFDirectoryLoader(DATA_PATH)
            return document_loader.load()


        def split_documents(documents: list[Document]):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
                is_separator_regex=False,
            )
            return text_splitter.split_documents(documents)


        def add_to_chroma(chunks: list[Document]):
            # Load the existing database.
            db = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
            )

            # Calculate Page IDs.
            chunks_with_ids = calculate_chunk_ids(chunks)

            # Add or Update the documents.
            existing_items = db.get(include=[])  # IDs are always included by default
            existing_ids = set(existing_items["ids"])
            output = f"Number of existing documents in DB: {len(existing_ids)}"
            #print(f"Number of existing documents in DB: {len(existing_ids)}")

            # Only add documents that don't exist in the DB.
            new_chunks = []
            for chunk in chunks_with_ids:
                if chunk.metadata["id"] not in existing_ids:
                    new_chunks.append(chunk)

            if len(new_chunks):
                output = output + f"\n👉 Adding new documents: {len(new_chunks)}"
                #print(f"👉 Adding new documents: {len(new_chunks)}")
                new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                db.add_documents(new_chunks, ids=new_chunk_ids)
                db.persist()
            else:
                output = output + "\n✅ No new documents to add"
                #print(output)
            return output


        def calculate_chunk_ids(chunks):

            # This will create IDs like "data/monopoly.pdf:6:2"
            # Page Source : Page Number : Chunk Index

            last_page_id = None
            current_chunk_index = 0

            for chunk in chunks:
                source = chunk.metadata.get("source")
                page = chunk.metadata.get("page")
                current_page_id = f"{source}:{page}"

                # If the page ID is the same as the last one, increment the index.
                if current_page_id == last_page_id:
                    current_chunk_index += 1
                else:
                    current_chunk_index = 0

                # Calculate the chunk ID.
                chunk_id = f"{current_page_id}:{current_chunk_index}"
                last_page_id = current_page_id

                # Add it to the page meta-data.
                chunk.metadata["id"] = chunk_id

            return chunks


        def clear_database():
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)


        return main()

# ---------------------------------------------------------------------------------

# Creating tool for retrieving data

# context_text = ''
# sources = []



class Retrival_from_RAG_INPUT(BaseModel):
    """Input schema for MyCustomTool."""

    QUERY_TEXT: str = Field(
        ...,
        description="The input query from the user"
    )

    CHROMA_PATH: str = Field(
        ...,
        description="Path of the folder where the chroma db exists"
    )

class RetrievalOutput(BaseModel):
    context: str
    sources: List[Optional[str]]


class Retrival_from_RAG_TOOL(BaseTool):
    name: str = "Retrival from RAG"
    description: str = (
        "Retrieves the data from the rag"
    )
    args_schema: Type[BaseModel] = Retrival_from_RAG_INPUT

    def _run(self, QUERY_TEXT: str, CHROMA_PATH: str) -> dict:
        SCORE_THRESHOLD = 500  # adjust as needed for your use case

        def main():
            # Create CLI.
            # parser = argparse.ArgumentParser()
            # parser.add_argument("query_text", type=str, help="The query text.")
            # args = parser.parse_args()
            # query_text = args.query_text
            # query_rag(query_text)

            return query_rag(QUERY_TEXT)

        def get_embedding_function():
            # embeddings = BedrockEmbeddings(
            #     credentials_profile_name="default", region_name="us-east-1"
            # )
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            return embeddings

        def query_rag(query_text: str):
            # Prepare the DB.
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            results = db.similarity_search_with_score(query_text, k=5)
            # Filter results by threshold
            filtered = [(doc, score) for doc, score in results if score <= SCORE_THRESHOLD]
            if not filtered:
                return json.dumps({
                    "context": "",
                    "sources": [],
                    "scores": [],
                    "no_relevant_docs": True
                })
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered])
            sources = [doc.metadata.get("id", None) for doc, _score in filtered]
            scores = [_score for doc, _score in filtered]
            return json.dumps({
                "context": context_text,
                "sources": sources,
                "scores": scores,
                "no_relevant_docs": False
            })

        output = main()
        print('from tool', output)
        return output


# def pass_values_to_main_code():
#     print('Here', context_text, sources)
#     return context_text, sources

# ---------------------------------------------------------------------------------


# creating tool for web search

# No custom tools

# from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# search_tool = SerperDevTool(n_results=3)
# scrape_tool = ScrapeWebsiteTool()

# ---------------------------------------------------------------------------------

# creating tool for summarization



class Collecting_retrieved_info_INPUT(BaseModel):
    """Input schema for MyCustomTool."""

    QUERY_TEXT: str = Field(
        ...,
        description="The input query from the user"
    )

    CONTEXT_TEXT: str = Field(
        ...,
        description="The information retrieved from the RAG"
    )

    PROMPT_TEMPLATE: str = Field(
        ...,
        description="The template that needs to be used to feed content and source t"
    )

    SOURCES: List[Optional[str]] = Field(
        ...,
        description="The sources from which the information was retrieved"
    )

class Collecting_retrieved_info_TOOL(BaseTool):
    name: str = "Collecting retrieved info"
    description: str = (
        "Retrieves the data from the rag"
    )
    args_schema: Type[BaseModel] = Collecting_retrieved_info_INPUT

    def _run(self, QUERY_TEXT: str, CONTEXT_TEXT: str, PROMPT_TEMPLATE: str, SOURCES: List[Optional[str]]) -> str:



        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=CONTEXT_TEXT, question=QUERY_TEXT)
        # print(prompt)

        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)

        #sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {SOURCES}"
        #print(formatted_response)

        return response_text

# ---------------------------------------------------------------------------------
