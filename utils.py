from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_community.vectorstores import Pinecone
from initializations import *
import streamlit as st
import pinecone


class PrepareDocsAndEmbeddings:
    def __init__(self, path_to_directory=pdf_folder_path, index_name="chatbot"):
        self.path_to_directory = path_to_directory
        self.index_name = index_name

    def load_and_split_docs(self, chunk_size=4000, chunk_overlap=0):
        loader = PyPDFDirectoryLoader(self.path_to_directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = loader.load()
        final_docs = text_splitter.split_documents(docs)
        return final_docs

    def get_embeddings(self, prepare_embeddings=False):
        embeddings = HuggingFaceEmbeddings()
        # initialize pinecone
        pinecone.Pinecone(
            api_key=PINECONE_API_KEY,
            environment="us-east-1"
        )
        if prepare_embeddings:
            final_docs = self.load_and_split_docs(chunk_size=4000, chunk_overlap=0)
            vector_db = Pinecone.from_documents(final_docs, embeddings, index_name=self.index_name)
        else:
            vector_db = Pinecone.from_existing_index(index_name=self.index_name, embedding=embeddings)
        return vector_db


class Query:
    def __init__(self, vector_db):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
        self.vector_db = vector_db

    def get_query_answer(self, llm, query):
        template = """
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        {question}
        Answer:
        """
        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            # chain_type="map_reduce",
                                            retriever=self.vector_db.as_retriever(),
                                            input_key="question",
                                            chain_type_kwargs={
                                                "verbose": True,
                                                "prompt": prompt,
                                                "memory": ConversationBufferMemory(
                                                    memory_key="history",
                                                    input_key="question"),
                                            }
                                            )

        response = chain.run(query)
        return response

    def get_conversation_string(self, conversation_string=""):
        for i in range(len(st.session_state['responses']) - 1):
            conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
            conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
        return conversation_string
