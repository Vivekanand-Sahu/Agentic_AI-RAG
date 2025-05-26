import ast
import json
import os
from datetime import datetime, timedelta

from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from crewai.flow.flow import Flow, listen, start, or_, and_

from agentic_rag.crews.purpose_identifier.purpose_identifier import purpose_identifier
from agentic_rag.crews.document_uploader.document_uploader import document_uploader
from agentic_rag.crews.information_retriever.information_retriever import information_retriever
from agentic_rag.crews.rag_retriever.rag_retriever import rag_retriever
from agentic_rag.crews.web_searcher.web_searcher import web_searcher
from agentic_rag.crews.summarizer.summarizer import summarizer
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

# from .tools.custom_tool import pass_values_to_main_code
import re
import numpy as np

def clean_text(text):
    # Remove triple backticks
    text = text.replace("```", "")
    
    # Replace problematic characters
    text = text.replace("—", "-").replace("–", "-")  # Replace dashes
    text = text.replace("’", "'").replace("‘", "'")  # Replace single quotes
    text = text.replace("“", '"').replace("”", '"')  # Replace double quotes
    
    # Remove excessive whitespace
    text = text.strip()
    
    return text


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}
{sources}

---

Answer the question based on the above context: {question}
"""


class Agentic_RAG_State(BaseModel):
    QUERY_TEXT: str = ''
    PURPOSE_DECIDER_OUTPUT: str = ''
    DOCUMENT_UPLOADER_OUTPUT: str = ''
    INFORMATION_RETRIEVER_OUTPUT: str = ''
    CONTEXT_TEXT: str = ''
    SOURCES: list = []
    WEB_OUTPUT: str = ''
    WEB_SOURCES: list = []
    DATA_PATH: str = "/part-vol-3/weaver-core/particle_transformer_quant/Proj_Files/Agentic_AI-RAG/Upload_File"
    CHROMA_PATH: str = "/part-vol-3/weaver-core/particle_transformer_quant/Proj_Files/Agentic_AI-RAG/ChromaDB_File"
    SUMMARY: str = ''
    PROMPT_TEMPLATE: str = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """
    SCORES: list = []
    NO_RELEVANT_DOCS: bool = False
    SESSION_LOG_DIR: str = ''
    CACHE_EMBED_MODEL: str = 'nomic-embed-text'
    CACHE: list = []  # List of dicts (log_entry)
    CACHE_MAX_SIZE: int = 5
    CACHE_MAX_AGE_MINUTES: int = 60
    CACHED_HIT: bool = False
    QUESTION_EMBEDDING: list = []

# ------------------------------------------------------------------------------------

class Agentic_RAG_Flow(Flow[Agentic_RAG_State]):
    
    @start()
    def reset_retrieval_state(self):
        self.state.CONTEXT_TEXT = ''
        self.state.SOURCES = []
        self.state.WEB_OUTPUT = ''
        self.state.WEB_SOURCES = []
        self.state.SUMMARY = ''
        self.state.SCORES = []
        self.state.NO_RELEVANT_DOCS = False
        self.state.CACHED_HIT = False
        self.state.QUESTION_EMBEDDING = []

# -----------------------------

    @listen(reset_retrieval_state)
    def Cache_check(self):
        # Get embedding for the question
        embedding_function = OllamaEmbeddings(model=self.state.CACHE_EMBED_MODEL)
        self.state.QUESTION_EMBEDDING = embedding_function.embed_query(self.state.QUERY_TEXT)

        # Check for similar question
        for entry in self.state.CACHE:
            emb = np.array(entry['embedding'])
            q_emb = np.array(self.state.QUESTION_EMBEDDING)
            sim = np.dot(emb, q_emb) / (np.linalg.norm(emb) * np.linalg.norm(q_emb))
            if sim > 0.85: # Adjust threshold as needed
                print("Cache hit!", sim)
                self.state.SUMMARY = entry['answer']
                self.state.CACHED_HIT = True
                break
        
        print("Cache hit, Answer:", self.state.SUMMARY)

# -----------------------------
    
    @listen(Cache_check)
    def Purpose_Identification(self):
        if not self.state.CACHED_HIT:
            # self.state.QUERY_TEXT = 'Look for info in the documents and tell me how to get out of jail?'
            result = (
                purpose_identifier()
                .crew()
                .kickoff(inputs={
                    "QUERY_TEXT": self.state.QUERY_TEXT
                    })
            )

            self.state.PURPOSE_DECIDER_OUTPUT = result
            print("PURPOSE_DECIDER_OUTPUT", self.state.PURPOSE_DECIDER_OUTPUT)

# -----------------------------

    @listen(Purpose_Identification)
    def Document_Uploading(self):
        if 'Upload' in self.state.PURPOSE_DECIDER_OUTPUT and not self.state.CACHED_HIT:
            result = (
                document_uploader()
                .crew()
                .kickoff(inputs={
                    "PURPOSE_DECIDER_OUTPUT": str(self.state.PURPOSE_DECIDER_OUTPUT),
                    "DATA_PATH": self.state.DATA_PATH,
                    "CHROMA_PATH": self.state.CHROMA_PATH
                    })
            )

            self.state.DOCUMENT_UPLOADER_OUTPUT = result
            print("DOCUMENT_UPLOADER_OUTPUT:", result)
        else:
            print("No document uploaded")

# -----------------------------

    @listen(Purpose_Identification)
    def Information_Retrieving(self):
        
        if 'Upload' != self.state.PURPOSE_DECIDER_OUTPUT and not self.state.CACHED_HIT:
            result = (
                information_retriever()
                .crew()
                .kickoff(inputs={
                    "QUERY_TEXT": self.state.QUERY_TEXT,
                    })
            )

            self.state.INFORMATION_RETRIEVER_OUTPUT = result
            print("INFORMATION_RETRIEVER_OUTPUT:", result)

# -----------------------------

    @listen(Information_Retrieving)
    def RAG_Retrieving(self):

        if str(self.state.INFORMATION_RETRIEVER_OUTPUT) == 'RAG' and not self.state.CACHED_HIT:
            result = (
                rag_retriever()
                .crew()
                .kickoff(inputs={
                    "QUERY_TEXT": self.state.QUERY_TEXT,
                    "INFORMATION_RETRIEVER_OUTPUT": str(self.state.INFORMATION_RETRIEVER_OUTPUT),
                    "DATA_PATH": self.state.DATA_PATH,
                    "CHROMA_PATH": self.state.CHROMA_PATH
                    })
            )
            # Handle CrewOutput object
            raw_data = {}
            if hasattr(result, "raw"):
                try:
                    raw_data = json.loads(result.raw)
                except Exception as e:
                    print(f"Error decoding CrewOutput.raw: {e}")
            else:
                print("CrewOutput has no 'raw' attribute")
            self.state.CONTEXT_TEXT = raw_data.get("context", "")
            self.state.SOURCES = raw_data.get("sources", [])
            self.state.SCORES = raw_data.get("scores", [])
            self.state.NO_RELEVANT_DOCS = raw_data.get("no_relevant_docs", False)
            print("CONTEXT_TEXT:", self.state.CONTEXT_TEXT)
            print("SOURCES:", self.state.SOURCES)
            print("SCORES:", self.state.SCORES)
            print("NO_RELEVANT_DOCS:", self.state.NO_RELEVANT_DOCS)

            #self.state.CONTEXT_TEXT, self.state.SOURCES = pass_values_to_main_code()
            # print(result.pydantic, result.raw)

# -----------------------------

    @listen(or_(Information_Retrieving, RAG_Retrieving))
    def Web_Search(self):

        if not self.state.CACHED_HIT and str(self.state.INFORMATION_RETRIEVER_OUTPUT) == 'Web' or (self.state.NO_RELEVANT_DOCS and self.state.WEB_OUTPUT == '' and self.state.DOCUMENT_UPLOADER_OUTPUT == ''):
            result = (
                web_searcher()
                .crew()
                .kickoff(inputs={
                    "QUERY_TEXT": self.state.QUERY_TEXT,
                    "INFORMATION_RETRIEVER_OUTPUT": str(self.state.INFORMATION_RETRIEVER_OUTPUT)
                    })
            )
            result = str(result)
            result = clean_text(str(result))
            print(result)
            try:
                # Try to extract tuple from result
                if not (result and result[0] == '(' and result[-1] == ')'):
                    left_most = result.find('(')
                    right_most = result.rfind(')')
                    if left_most != -1 and right_most != -1 and right_most > left_most:
                        result_tuple = result[left_most:right_most+1]
                    else:
                        raise ValueError("No tuple found in result string.")
                else:
                    result_tuple = result
                decoded_tuple_form = ast.literal_eval(result_tuple)
                # Truncate WEB_OUTPUT to avoid exceeding model context window
                MAX_WEB_OUTPUT_CHARS = 6000
                web_output = decoded_tuple_form[1]
                if isinstance(web_output, str) and len(web_output) > MAX_WEB_OUTPUT_CHARS:
                    web_output = web_output[:MAX_WEB_OUTPUT_CHARS]
                self.state.WEB_SOURCES, self.state.WEB_OUTPUT = decoded_tuple_form[0], web_output
                print("WEB_SOURCES:", self.state.WEB_SOURCES)
                print("WEB_OUTPUT:", self.state.WEB_OUTPUT)
            except Exception as e:
                print(f"Error parsing web search result: {e}")
                self.state.WEB_SOURCES, self.state.WEB_OUTPUT = [], result
                print("WEB_SOURCES:", self.state.WEB_SOURCES)
                print("WEB_OUTPUT:", self.state.WEB_OUTPUT)

# -----------------------------

    @listen(and_(Web_Search, RAG_Retrieving))
    def Summarization(self):
        
        if self.Purpose_Identification != 'Upload' and not self.state.CACHED_HIT:
            note = ''
            if self.state.NO_RELEVANT_DOCS or len(self.state.SOURCES) + len(self.state.WEB_SOURCES) == 0:
                note = 'No relevant documents were found in the database and/or web.\n\n'    # enhance the logic for specific handeling of the consitions in if case. 
            result = (
                summarizer()
                .crew()
                .kickoff(inputs={
                    "QUERY_TEXT": self.state.QUERY_TEXT,
                    "CONTEXT_TEXT": self.state.CONTEXT_TEXT,
                    "SOURCES": self.state.SOURCES,
                    "WEB_OUTPUT": self.state.WEB_OUTPUT,
                    "WEB_SOURCES": self.state.WEB_SOURCES,
                    "PROMPT_TEMPLATE": PROMPT_TEMPLATE      # self.PROMPT_TEMPLATE not working
                    })
            )
            self.state.SUMMARY = note + str(result)
            print("SUMMARY:", self.state.SUMMARY)
    
# -----------------------------
    
    @listen(Summarization)
    def log_chat(self):
        if not self.state.SESSION_LOG_DIR:
            return
        date = datetime.now()
        log_entry = {
            'timestamp': date.isoformat(),
            'question': self.state.QUERY_TEXT,
            'answer': self.state.SUMMARY,
            'sources': self.state.SOURCES + self.state.WEB_SOURCES,
            'document_uploaded': self.state.DOCUMENT_UPLOADER_OUTPUT,
            'cache_hit': self.state.CACHED_HIT,
            #'embedding': self.state.QUESTION_EMBEDDING
        }
        log_file = os.path.join(self.state.SESSION_LOG_DIR, 'chat_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Only add to cache if not a cache hit (i.e., new answer)
        if not self.state.CACHED_HIT:
            self.state.CACHE.append({
                'question': self.state.QUERY_TEXT,
                'answer': self.state.SUMMARY,
                'embedding': self.state.QUESTION_EMBEDDING,
                'timestamp': date  # store as datetime object
            })
        # Prune cache (remove expired/oversized entries)
        else:
            now = date
            self.state.CACHE = [
                entry for entry in self.state.CACHE
                if (now - entry['timestamp']).total_seconds() < self.state.CACHE_MAX_AGE_MINUTES * 60
            ]
            if len(self.state.CACHE) > self.state.CACHE_MAX_SIZE:
                self.state.CACHE = self.state.CACHE[-self.state.CACHE_MAX_SIZE:]
        print("Chat log entry added.")
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


def create_session_dir():
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    session_dir = os.path.join(base_dir, "logs", f"session_{session_time}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session log directory: {session_dir}")
    return session_dir


def kickoff():
    flow_Agentic_RAG = Agentic_RAG_Flow()
    # Create session log directory once per session
    session_dir = create_session_dir()
    flow_Agentic_RAG.state.SESSION_LOG_DIR = session_dir

    print('Hi I am Amar, how may I help you? ')
    
    while True:
        user_question = input("Amar: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Ending session.")
            break
        flow_Agentic_RAG.reset_retrieval_state()
        flow_Agentic_RAG.state.QUERY_TEXT = user_question
        flow_Agentic_RAG.kickoff()


def plot():
    flow_Agentic_RAG = Agentic_RAG_Flow()
    flow_Agentic_RAG.plot()


# Add this function to allow single question/answer for Streamlit
def run_agentic_rag_flow(flow, user_question):
    flow.reset_retrieval_state()
    flow.state.QUERY_TEXT = user_question
    flow.kickoff()
    return flow.state.SUMMARY


if __name__ == "__main__":
    # argparse 
    kickoff()
