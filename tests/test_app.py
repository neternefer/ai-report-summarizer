"""
Unit Tests for AI Report Summarizer

"""
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import unittest
import io
import pytest
from PIL import Image

from file_utils import (
    detect_file_type,
    validate_file,
    process_file,
    EmptyFileError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    CorruptedFileError,
    AZURE_MAX_FILE_MB
)
from storage import Storage
from text_extractor import TextExtractor
from openai_client import Interpreter
import config


class TestApp(unittest.TestCase):
    """Unit tests for AI Report Summarizer."""

    def test_detect_file_type_pdf(self):
        assert detect_file_type(b"%PDF-1.4\n%...") == "pdf"

    def test_detect_file_type_image(self):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), "red").save(buf, "PNG")
        assert detect_file_type(buf.getvalue()) == "image"

    def test_detect_file_type_empty(self):
        with pytest.raises(EmptyFileError):
            detect_file_type(b"")

    def test_detect_file_type_unsupported(self):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(b"not a PDF or image")

    def test_validate_file_image_too_large(self):
        large_bytes = b"\x00" * ((AZURE_MAX_FILE_MB + 1) * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            validate_file(large_bytes, "image")

    @patch("file_utils.fitz.open", side_effect=Exception("corrupted PDF"))
    def test_validate_file_pdf_corrupted(self, _):
        with pytest.raises(CorruptedFileError):
            validate_file(b"%PDF-1.4\n...", "pdf")

    def test_validate_file_jpg_corrupted(self):
        with pytest.raises(CorruptedFileError):
            validate_file(b"\xFF\xD8\xFF\xE0" + b"invalid data", "image")

    def test_process_file_image_returns_bytes(self):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, "PNG")
        assert isinstance(process_file(buf.getvalue()), bytes)

    def test_process_file_pdf_returns_list(self):
        sample_pdf_path = "tests/sample.pdf"
        with open(sample_pdf_path, "rb") as f:
            images = process_file(f.read())
        assert isinstance(images, list)
        assert all(isinstance(img, Image.Image) for img in images)

    @patch.object(config.AzureConfig, "get_blob_service_client")
    def test_upload_bytes_and_get_url(self, mock_get_client):
        mock_blob_client = MagicMock()
        mock_blob_client.url = "https://fake.blob/url"
        mock_service_client = MagicMock()
        mock_service_client.get_blob_client.return_value = mock_blob_client
        mock_get_client.return_value = mock_service_client

        storage = Storage()
        url = storage.upload_bytes_and_get_url(b"dummy", extension="png")
        mock_blob_client.upload_blob.assert_called_once_with(
            b"dummy", overwrite=True
            )
        assert url == "https://fake.blob/url"

    def test_cleaned_result_with_caption_and_text(self):
        extractor = TextExtractor()
        result = SimpleNamespace(
            caption=SimpleNamespace(text="A cat"),
            read=SimpleNamespace(blocks=[SimpleNamespace(
                lines=[{"text": "hello"}])]
                )
        )
        cleaned = extractor.cleaned_result(result)
        assert cleaned["caption"] == "A cat"
        assert cleaned["text_lines"] == ["hello"]

    def test_cleaned_result_no_caption_or_text(self):
        extractor = TextExtractor()
        result = MagicMock()
        result.caption = None
        result.read.blocks = []
        cleaned = extractor.cleaned_result(result)
        assert cleaned["caption"] == "No Caption detected"
        assert cleaned["text_lines"] == ["No text lines detected"]

    @patch("openai_client.AzureConfig.get_openai_client")
    def test_interpret_data(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="summary text"))]
        )
        mock_get_client.return_value = mock_client

        inter = Interpreter()
        result = inter.interpret_data("sys", "prompt")
        assert result == "summary text"
