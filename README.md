# AI Report Summarizer

**AI Report Summarizer** is a Streamlit-based application that automatically extracts text and captions from PDF and image files and generates structured, factual summaries using Azure Cognitive Services and Azure OpenAI.  

- Processes PDFs and images seamlessly
- Extracts OCR text and visual captions
- Generates multi-page summaries with context
- Integrates with Azure Blob Storage for file hosting
- Supports large documents with automated page handling

---

## Why the Project is Useful

This project is designed to save time and reduce manual work for professionals who need to summarize reports, scanned documents, or visual data. Key benefits:

- **Automated OCR and analysis:** No manual transcription required  
- **Structured summaries:** Combines text and image context for factual reporting  
- **Cloud-ready:** Uses Azure services for reliability and scalability  
- **Extensible:** Modular code allows adding new AI models or processing logic  

---

## Getting Started

### Prerequisites

- Python 3.11+
- Azure subscription with:
  - Computer Vision resource
  - OpenAI resource
  - Blob Storage container

### Installation

1. Clone the repository:

    git clone - https://github.com/yourusername/ai-report-summarizer.git
   
    cd ai-report-summarizer

3. Create and activate a virtual environment:

    python -m venv .venv
   
    source .venv/bin/activate  # Linux/Mac
   
    .venv\Scripts\activate     # Windows

5. Install dependencies:

    pip install -r requirements.txt

6. Configure Azure credentials in a .env file:

    COMPUTER_VISION_ENDPOINT=https://<your-endpoint>.cognitiveservices.azure.com/
    COMPUTER_VISION_KEY=<your-key>

    OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
    OPENAI_KEY=<your-key>
    OPENAI_DEPLOYMENT=<deployment-name>

    AZURE_STORAGE_ENDPOINT=https://<your-storage-account>.blob.core.windows.net/
    AZURE_STORAGE_KEY=<your-key>
    CONTAINER_NAME=<container-name>
    BLOB_NAME=<optional-blob-name>

7. Running the App

    streamlit run app.py

8. Upload a PDF or image via the sidebar

    - Files are automatically validated, uploaded, and processed

    - Multi-page summaries are displayed in the main panel

    - Azure OpenAI generates factual summaries from both text and image content

9. Testing

    pytest test_app.py

    pytest test_performance.py --benchmark-only
