"""
Streamlit Frontend for AI Report Summarizer

This module provides the user interface for:
- Selecting a PDF/image file OR providing a direct URL
- Uploading the file to Azure Blob Storage
- Running OCR on all pages
- Passing OCR + image URLs to OpenAI for interpretation
- Displaying a multi-page summary

UX improvements:
- Inputs moved to sidebar
- URL field clears itself on Enter (kept exactly as implemented)
- Automatic execution once file or URL is provided
- Removed Run Analysis button
"""
import os
import requests
import streamlit as st

from text_extractor import TextExtractor
from openai_client import Interpreter
from storage import Storage
from file_utils import (
    process_file,
    image_to_png_bytes,
    EmptyFileError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    CorruptedFileError,
)


def main():
    """
    Main Streamlit app function.

    Responsibilities:
    - Render sidebar widgets
    - Validate inputs
    - Upload files to Azure Blob Storage
    - Run OCR on all pages
    - Generate cleaned text for OpenAI interpretation
    - Display multi-page summary
    """
    st.title("AI Report Summarizer")

    with st.sidebar:   
        uploaded_file = st.file_uploader(
            "Upload a PDF or image",
            type=["pdf", "png", "jpg", "jpeg"]
        )

    should_run = bool(uploaded_file)
    if not should_run:
        return

    try:
        if uploaded_file:
            file_bytes = uploaded_file.read()
            original_extension = (os.path.splitext(uploaded_file.name)[1]
                                  .lstrip("."))
        else:
            st.error("No valid file.")
            return

        # Upload to Azure Blob Storage
        with st.spinner("Uploading file to Azure Blob Storage..."):
            storage = Storage()
            blob_url = storage.upload_bytes_and_get_url(
                file_bytes,
                extension=original_extension or None
            )

        st.session_state.blob_url = blob_url
        st.success(f"Uploaded successfully: {blob_url}")

        # Process file and extract OCR
        st.info("Processing file...")
        processed = process_file(file_bytes)
        pages = processed if isinstance(processed, list) else [processed]

        extractor = TextExtractor()
        cleaned_pages = []

        for idx, page in enumerate(pages, start=1):
            st.write(f"OCR on page {idx}...")

            if isinstance(page, bytes):
                img_bytes = page
            else:
                img_bytes = image_to_png_bytes(page)

            ocr_result = extractor.analyze_img(img_bytes)
            cleaned_pages.append(extractor.cleaned_result(ocr_result))

        # Generate summary
        st.info("Generating summary using OpenAI...")
        interpreter = Interpreter()
        summary = ""

        for idx, cleaned_page in enumerate(cleaned_pages, start=1):
            st.write(f"Interpreting page {idx}...")

            system, prompt = interpreter.build_interpretation_prompt(
                cleaned_page,
                image_url=st.session_state.blob_url
            )

            page_summary = interpreter.interpret_data(system, prompt)
            summary += f"\n\n### Page {idx}\n{page_summary}"

        st.subheader("Summary")
        st.write(summary)

    except (EmptyFileError, UnsupportedFileTypeError,
            FileTooLargeError, CorruptedFileError) as e:
        st.error(f"File error: {str(e)}")

    except requests.RequestException as e:
        st.error(f"Failed to download URL: {str(e)}")

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
