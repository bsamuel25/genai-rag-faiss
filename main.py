import os
import asyncio
import glob
from dotenv import load_dotenv
import numpy as np
import plotly.graph_objects as go

# imports for langchain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sklearn.manifold import TSNE

import streamlit as st

# set page configuration
st.set_page_config(page_title="GenAI - RAG Demo", page_icon="🤖", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

if not st.session_state.messages:
    st.session_state.messages.append(
        {"role": "assistant", 
         "content": "How can I help you today?"}
         )

# Load environment variables in a file called .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

def get_embedding():
    embeddings = OpenAIEmbeddings()
    return embeddings

def load_documents():
    # Read in documents using LangChain's loaders
    # Take everything in all the sub-folders of our knowledgebase
    folders = glob.glob("knowledge-base/*")

    text_loader_kwargs = {'encoding': 'utf-8'}

    documents = []
    for folder in folders:  
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata['doc_type'] = doc_type
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")

    return chunks

def embedding_documents(chunks):
    embeddings = get_embedding()

    # Create the FAISS vector store to use
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    total_vectors = vectorstore.index.ntotal
    dimensions = vectorstore.index.d
    print(f"FAISS vector store created with {total_vectors} vectors of dimension {dimensions}")
    
    return vectorstore

def query_vectorstore(vectorstore, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return results

def visualize_vectors(vectorstore):
    # This function is a placeholder for any visualization logic you might want to add
    vectors = []
    documents = []
    doc_types = []
    colors = []
    color_map = {'products':'blue', 'employees':'green', 'contracts':'red', 'company':'orange'}

    total_vectors = vectorstore.index.ntotal
    for i in range(total_vectors):
        vectors.append(vectorstore.index.reconstruct(i))
        doc_id = vectorstore.index_to_docstore_id[i]
        document = vectorstore.docstore.search(doc_id)
        documents.append(document.page_content)
        doc_type = document.metadata['doc_type']
        doc_types.append(doc_type)
        colors.append(color_map[doc_type])
        
    vectors = np.array(vectors)

    # Reduce the dimensionality of the vectors to 2D using t-SNE
    # (t-distributed stochastic neighbor embedding)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)  

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D FAISS Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()


def create_conversation_agent(vectorstore):
    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=model_name, api_key=api_key)

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # the retriever is an abstraction over the VectorStore that will be used during RAG
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    # putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    #query = "Can you describe Insurellm in a few sentences"
    #result = conversation_chain.invoke({"question":query})
    #print(result["answer"])

    return conversation_chain

async def main(conversation_agent):
    st.title("🤖 GenAI - RAG Demo")
    st.write("Ask questions about the knowledge base documents.")

    user_input = st.text_input("Your question:", key="input")

    if user_input:
        with st.spinner("Generating response..."):
            result = await asyncio.to_thread(conversation_agent.invoke, {"question": user_input})
            answer = result["answer"]

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

if __name__ == "__main__":
    chunks = load_documents()
    vectorstore = embedding_documents(chunks)

    # visualization (if needed)
    #visualize_vectors(vectorstore)
    conversation_agent = create_conversation_agent(vectorstore=vectorstore)
    asyncio.run(main(conversation_agent))
