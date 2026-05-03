# GenAI RAG Demo

A comprehensive Retrieval-Augmented Generation (RAG) demonstration using LangChain and FAISS vector database. This project showcases how to build an intelligent conversational AI system that can answer questions about a knowledge base containing company information, contracts, employees, and products.

## 🚀 Features

- **Document Processing**: Automatically loads and processes Markdown documents from a structured knowledge base
- **Vector Embeddings**: Uses OpenAI embeddings to convert text chunks into high-dimensional vectors
- **FAISS Vector Store**: Efficient similarity search using Facebook AI Similarity Search (FAISS)
- **Conversational AI**: Interactive chat interface powered by GPT models
- **Streamlit UI**: Modern web interface for easy interaction
- **Visualization**: 2D t-SNE visualization of document embeddings
- **Memory Management**: Conversation history tracking for contextual responses

## 📋 Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GenAI_RAG
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or if using pyproject.toml
   pip install -e .
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

## ⚙️ Setup

1. **Knowledge Base Structure**
   The system expects a `knowledge-base/` directory with the following structure:
   ```
   knowledge-base/
   ├── company/
   │   ├── about.md
   │   ├── careers.md
   │   └── overview.md
   ├── contracts/
   │   ├── Contract with Apex Reinsurance for Rellm.md
   │   └── ... (other contract files)
   ├── employees/
   │   ├── Alex Chen.md
   │   └── ... (other employee files)
   └── products/
       ├── Carllm.md
       ├── Homellm.md
       └── ... (other product files)
   ```

2. **Document Format**
   All documents should be in Markdown (.md) format. The system automatically categorizes documents based on their folder names.

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run main.py
   ```

2. **Access the interface**
   Open your browser to `http://localhost:8501`

3. **Ask questions**
   Type your questions in the text input field. The system will:
   - Search for relevant documents in the knowledge base
   - Retrieve the most similar content
   - Generate a contextual response using the LLM

### Example Queries

- "What products does the company offer?"
- "Tell me about our contract with Apex Reinsurance"
- "Who is Alex Chen and what do they do?"
- "What are the company career opportunities?"

### Jupyter Notebook

Explore the implementation details in `notebooks/RAG_Langchain_FAISS.ipynb`, which includes:
- Step-by-step document processing
- Vector embedding creation
- FAISS index construction
- Interactive visualization of embeddings

## 📁 Project Structure

```
GenAI_RAG/
├── main.py                 # Main Streamlit application
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # This file
├── knowledge-base/        # Document storage
│   ├── company/          # Company information
│   ├── contracts/        # Contract documents
│   ├── employees/        # Employee profiles
│   └── products/         # Product descriptions
├── notebooks/            # Jupyter notebooks
│   └── RAG_Langchain_FAISS.ipynb
└── requirements.txt      # Python dependencies (if not using pyproject.toml)
```

## 🔍 How It Works

### 1. Document Loading
- Scans the `knowledge-base/` directory recursively
- Loads all Markdown files using LangChain's `DirectoryLoader`
- Adds metadata (document type) based on folder structure

### 2. Text Chunking
- Splits documents into manageable chunks (1000 characters with 200 character overlap)
- Preserves document metadata for categorization

### 3. Vector Embeddings
- Converts text chunks to high-dimensional vectors using OpenAI embeddings
- Stores vectors in FAISS for efficient similarity search

### 4. Retrieval-Augmented Generation
- User query → Vector similarity search → Relevant documents retrieved
- Retrieved context + user query → LLM generates response
- Conversation memory maintains context across interactions

### 5. Visualization
- Optional t-SNE dimensionality reduction for 2D visualization
- Color-coded points representing different document types
- Interactive Plotly charts for exploration

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: GPT model to use (default: gpt-4o-mini)

### Customization Options

- **Chunk Size**: Modify `chunk_size` in `CharacterTextSplitter` (default: 1000)
- **Overlap**: Adjust `chunk_overlap` for better context preservation (default: 200)
- **Similarity Search**: Change `k` parameter for number of retrieved documents (default: 5)
- **Temperature**: Adjust LLM creativity in `ChatOpenAI` (default: 0.7)

## 📦 Dependencies

Key libraries used:
- **LangChain**: Framework for building LLM applications
- **FAISS**: Efficient similarity search and clustering
- **Streamlit**: Web application framework
- **OpenAI**: GPT models and embeddings
- **Plotly**: Data visualization
- **scikit-learn**: t-SNE for dimensionality reduction
- **python-dotenv**: Environment variable management

See `pyproject.toml` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Vector search using [FAISS](https://github.com/facebookresearch/faiss)
- UI framework [Streamlit](https://streamlit.io/)

---

**Note**: This is a demonstration project. Ensure you have appropriate API access and comply with OpenAI's terms of service when deploying.
