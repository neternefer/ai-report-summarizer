"""
Performance benchmarks for AI Report Summarizer.

Benchmarks:
- process_file: speed of processing large image inputs
- TextExtractor.cleaned_result: speed of processing OCR results
- Interpreter.build_interpretation_prompt: speed of building large prompts

"""
import io
from PIL import Image
from file_utils import process_file
from text_extractor import TextExtractor
from openai_client import Interpreter


def make_test_image(size=(2000, 2000)) -> bytes:
    """
    Generates a large in-memory RGB image for benchmarking.

    Args:
        size (tuple[int, int]): Width and height of the image.

    Returns:
        bytes: PNG-encoded image bytes.
    """
    img = Image.new("RGB", size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_process_file_performance(benchmark):
    """
    Benchmark the processing speed of process_file for a large image.

    Measures how long it takes to convert an image into the format
    expected by the app (bytes or PIL Images for PDFs).

    Args:
        benchmark: pytest-benchmark fixture.
    """
    img_bytes = make_test_image()

    def run():
        return process_file(img_bytes)  # No filename argument needed

    result = benchmark(run)
    assert result is not None


def test_text_extraction_performance(benchmark, monkeypatch):
    """
    Benchmark TextExtractor.cleaned_result speed with OCR mocked.

    We mock analyze_img to avoid calling external services and focus
    on the performance of cleaned_result.

    Args:
        benchmark: pytest-benchmark fixture.
        monkeypatch: pytest fixture for patching methods.
    """
    extractor = TextExtractor()

    # Create a fake OCR result matching cleaned_result expectations
    fake_result = type("MockResult", (), {})()
    fake_result.caption = type("C", (), {"text": "caption"})()
    fake_result.read = type("R", (), {"blocks": []})()

    # Mock the analyze_img method to return the fake result
    monkeypatch.setattr(extractor, "analyze_img", lambda x: fake_result)

    def run():
        return extractor.cleaned_result(fake_result)

    cleaned = benchmark(run)
    assert "caption" in cleaned
    assert "text_lines" in cleaned


def test_prompt_building_performance(benchmark):
    """
    Benchmark the speed of building large OpenAI prompts.

    Args:
        benchmark: pytest-benchmark fixture.
    """
    inter = Interpreter()
    cleaned = {
        "caption": "A" * 2000,
        "text_lines": ["line " + str(i) for i in range(2000)]
    }

    def run():
        return inter.build_interpretation_prompt(cleaned, "http://fake")

    sys_msg, prompt = benchmark(run)
    assert isinstance(sys_msg, str)
    assert isinstance(prompt, str)
