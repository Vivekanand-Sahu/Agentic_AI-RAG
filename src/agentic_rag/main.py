
import ast

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


# from .tools.custom_tool import pass_values_to_main_code
import re

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

# ------------------------------------------------------------------------------------

# import ast
# def string_to_dict(input_string):
#     try:
#         result_dict = ast.literal_eval(str(input_string))
#         if isinstance(result_dict, dict):
#             return(result_dict)
#         else:
#             raise ValueError("The input string does not represent a dictionary.")
#     except (SyntaxError, ValueError) as e:
#         print(f"Error: {e}")
#         return None

# ------------------------------------------------------------------------------------

class Agentic_RAG_State(BaseModel):
    QUERY_TEXT: str = ''
    PURPOSE_DECIDER_OUTPUT: str = ''
    DOCUMENT_UPLOADER_OUTPUT: str = ''
    INFORMATION_RETRIEVER_OUTPUT: str = ''
    CONTEXT_TEXT: str = ''
    SOURCES: list = []
    WEB_OUTPUT: str = ''
    WEB_SOURCES: list = []
    DATA_PATH: str = "/content/rag-tutorial-v2/data"
    CHROMA_PATH: str = "/content/rag-tutorial-v2"
    SUMMARY: str = ''
    PROMPT_TEMPLATE: str = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """

# ------------------------------------------------------------------------------------


class Agentic_RAG_Flow(Flow[Agentic_RAG_State]):

    @start()
    def Purpose_Identification(self):
        self.state.QUERY_TEXT = input('Hi I am Amar, how may I help you?')
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

# -----------------------------

    @listen(Purpose_Identification)
    def Information_Retrieving(self):
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

        if str(self.state.INFORMATION_RETRIEVER_OUTPUT) != 'Web':
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


            decoded_tuple = ast.literal_eval(str(result))
            #print(type(decoded_tuple), decoded_tuple)
            self.state.CONTEXT_TEXT, self.state.SOURCES = decoded_tuple[0], decoded_tuple[1]
            #print(type(result), result)

            #self.state.CONTEXT_TEXT, self.state.SOURCES = pass_values_to_main_code()
            # print(result.pydantic, result.raw)
            print("CONTEXT_TEXT:", self.state.CONTEXT_TEXT)
            print("SOURCES:", self.state.SOURCES)

# -----------------------------


    @listen(Information_Retrieving)
    def Web_Search(self):

        if str(self.state.INFORMATION_RETRIEVER_OUTPUT) != 'RAG':
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
            if not (result[0] == '(' and result [-1] == ')'):
                left_most = result.find('(')
                right_most = result.rfind(')')
                result = result[left_most:right_most]
            
            print(result)

            decoded_tuple_form = ast.literal_eval(str(result))
            # print(type(decoded_tuple_form), decoded_tuple_form)
            # print(type(decoded_tuple_form[0]), decoded_tuple_form[0])
            # print(type(decoded_tuple_form[1]), decoded_tuple_form[1])
            
            self.state.WEB_SOURCES, self.state.WEB_OUTPUT = decoded_tuple_form[0], decoded_tuple_form[1]

            # self.state.WEB_SEARCHER_OUTPUT = result
            print("WEB_SOURCES:", self.state.WEB_SOURCES)
            print("WEB_OUTPUT:", self.state.WEB_OUTPUT)

# -----------------------------

    @listen(and_(Web_Search, RAG_Retrieving))
    def Summarization(self):
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

        self.state.SUMMARY = result
        print("SUMMARY:", result)


# ------------------------------------------------------------------------------------


def kickoff():
    flow_Agentic_RAG = Agentic_RAG_Flow()
    flow_Agentic_RAG.kickoff()


def plot():
    flow_Agentic_RAG = Agentic_RAG_Flow()
    flow_Agentic_RAG.plot()


if __name__ == "__main__":
    argparse 
    kickoff()
