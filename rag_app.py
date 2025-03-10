import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

class RAGApplication:
    def __init__(self, directory_path: str = "."):
        self.directory_path = directory_path
        self.vector_store = None
        self.qa_chain = None
        self.initialize_rag()

    def load_documents(self) -> List:
        """Load documents from the specified directory."""
        # Create loaders for Python and Markdown files
        python_loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.py",  # Load Python files
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            use_multithreading=True,
        )
        
        markdown_loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.md",  # Load Markdown files
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            use_multithreading=True,
        )
        
        # Load both types of documents
        python_docs = python_loader.load()
        markdown_docs = markdown_loader.load()
        
        # Combine all documents
        documents = python_docs + markdown_docs
        return documents

    def process_documents(self, documents: List) -> None:
        """Process documents and create vector store."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    def initialize_rag(self) -> None:
        """Initialize the RAG system."""
        # Load and process documents
        documents = self.load_documents()
        self.process_documents(documents)

        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            input_key="question"
        )

        # Create retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            chain_type="stuff",
            verbose=True
        )

    def query(self, question: str) -> dict:
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized")
        
        result = self.qa_chain({"question": question})
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"]
        }

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file")
        return

    # Initialize RAG application
    rag = RAGApplication()
    
    print("RAG Application initialized! You can now ask questions.")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        try:
            result = rag.query(question)
            print(result)
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 