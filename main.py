import torch
import gradio as gr
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model and PDF configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
PDF_PATH = "./file.pdf"


def load_and_split_pdf(pdf_path):
    """Load and split PDF into manageable chunks."""
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_documents(data)


def setup_embeddings():
    """Configure embeddings for vector store."""
    return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': DEVICE}, encode_kwargs={'normalize_embeddings': True})


def setup_model():
    """Configure and load the language model."""
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if DEVICE == "cuda" else None

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config)

    # Create pipeline and LLM
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, top_k=40, do_sample=True)

    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm)


def create_rag_chain(model):
    """Create the Retrieval-Augmented Generation (RAG) chain."""
    RAG_TEMPLATE = ("Context information is below.\n"
                    "---------------------\n"
                    "{context}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step "
                    "to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
                    "Query: {question}")

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return RunnablePassthrough.assign(context=lambda input: format_docs(input["context"])) | rag_prompt | model | StrOutputParser()


def run_inference(message: str, history: list):
    """Run inference on the input message."""
    # Perform similarity search
    docs = vectorstore.similarity_search(message)

    # Generate output
    output = chain.invoke({"context": docs, "question": message})

    # Clean up output (remove potential model-specific tokens)
    output = output.split('<|im_start|>assistant')[-1]

    return output


def main():
    """Main function to set up and launch the chatbot."""
    global vectorstore, chain

    # Load and split PDF
    all_splits = load_and_split_pdf(PDF_PATH)

    # Setup embeddings and vector store
    local_embeddings = setup_embeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    # Setup model and RAG chain
    model = setup_model()
    chain = create_rag_chain(model)

    # Launch Gradio interface
    gr.ChatInterface(fn=run_inference, type="messages").launch()


if __name__ == '__main__':
    main()
    