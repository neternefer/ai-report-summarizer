"""
File Utilities Module

Handles file type detection, validation, and PDF processing.

Provides:
- Custom exceptions for empty or unsupported files
- File type detection for PDFs and images
- Centralized file validation for supported formats
- Conversion of PDF pages to images (conversion only, no validation)
"""
import io
from typing import List, Union
from PIL import Image, UnidentifiedImageError
import fitz

# Azure file size constraints
AZURE_MAX_IMAGE_WIDTH = 10000
AZURE_MAX_IMAGE_HEIGHT = 10000
AZURE_MAX_FILE_MB = 500
AZURE_MAX_PDF_PAGES = 2000


class CorruptedFileError(Exception):
    """
    Raised when a file appears corrupted or cannot be properly read.

    Usage:
        raise CorruptedFileError("File is corrupted")
    """


class FileTooLargeError(Exception):
    """
    Raised when a file exceeds the allowed size limit.

    Usage:
        raise FileTooLargeError("File is too large")
    """


class EmptyFileError(Exception):
    """
    Raised when a file provided to a utility function is empty.

    Usage:
        raise EmptyFileError("File is empty")
    """


class UnsupportedFileTypeError(Exception):
    """
    Raised when a file format is unsupported or unrecognized by the
    utility functions.

    Usage:
        raise UnsupportedFileTypeError("File format not supported")
    """


def detect_file_type(file_bytes: bytes) -> str:
    """
    Detect the type of a file from its bytes content.

    Supported file types:
    - PDF
    - Image (any format supported by Pillow)

    Args:
        file_bytes (bytes): Raw content of the file.

    Raises:
        EmptyFileError: If the file is empty.
        UnsupportedFileTypeError: If the file is not a recognized PDF or image.

    Returns:
        str: "pdf" if file is PDF, "image" if file is an image.
    """
    if not file_bytes:
        raise EmptyFileError("File is empty")

    if file_bytes.startswith(b"%PDF"):
        return "pdf"

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        return "image"
    except UnidentifiedImageError as err:
        raise UnsupportedFileTypeError("Unsupported image format") from err
    except Exception as e:
        raise UnsupportedFileTypeError(f"Error processing file: {e}") from e


def validate_file(file_bytes: bytes, file_type: str):
    """
    Validates a file based on its type.

    Centralized validation performs:
    - File size limit check
    - Format-specific corruption detection
    - PDF page count check
    - Image dimension limits

    Args:
        file_bytes (bytes): Raw content of the file.
        file_type (str): Type of the file ("pdf" or "image").

    Raises:
        FileTooLargeError: If file or its rendered pages exceed size.
        CorruptedFileError: If the file cannot be opened or parsed.
        UnsupportedFileTypeError: If the file type is not supported.
    """
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > AZURE_MAX_FILE_MB:
        raise FileTooLargeError(
            f"{file_type.upper()} exceeds max allowed size of "
            f"{AZURE_MAX_FILE_MB} MB."
        )

    if file_type == "pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptedFileError("PDF appears corrupted.") from e

        # Page count validation
        page_count = len(doc)
        if page_count > AZURE_MAX_PDF_PAGES:
            raise FileTooLargeError(
                f"{page_count} pages exceeds max number of "
                f"{AZURE_MAX_PDF_PAGES}."
            )

        # Validate every page's dimensions BEFORE conversion
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(alpha=False)
            if (pix.width > AZURE_MAX_IMAGE_WIDTH or
               pix.height > AZURE_MAX_IMAGE_HEIGHT):
                raise FileTooLargeError(
                    f"PDF page {idx} renders to {pix.width}x{pix.height}, "
                    f"which exceeds max allowed "
                    f"{AZURE_MAX_IMAGE_WIDTH}x{AZURE_MAX_IMAGE_HEIGHT}."
                )

    elif file_type == "image":
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()

                if (img.width > AZURE_MAX_IMAGE_WIDTH or
                   img.height > AZURE_MAX_IMAGE_HEIGHT):
                    raise FileTooLargeError(
                        f"Image dimensions {img.width}×{img.height} exceed "
                        f"Azure OCR limits of "
                        f"{AZURE_MAX_IMAGE_WIDTH}×{AZURE_MAX_IMAGE_HEIGHT}."
                    )
        except UnidentifiedImageError as err:
            raise CorruptedFileError("Image is corrupted.") from err
        except Exception as e:
            raise CorruptedFileError("Image validation failed.") from e

    else:
        raise UnsupportedFileTypeError("Unsupported format.")


def image_to_png_bytes(img: Image.Image) -> bytes:
    """
    Convert a PIL Image to PNG bytes.

    Args:
        img (PIL.Image.Image): The image to convert.

    Returns:
        bytes: Encoded PNG bytes.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def process_file(file_bytes: bytes) -> Union[bytes, List[Image.Image]]:
    """
    Detects, validates, and converts uploaded files to a format suitable
    for OCR.

    PDF → list of PIL images
    Image → return bytes (unchanged)

    Args:
        file_bytes (bytes): Raw content of the file.

    Returns:
        bytes | List[PIL.Image.Image]:
            - Raw image bytes if input was an image
            - List of PIL Images if input was a PDF

    Raises:
        EmptyFileError, UnsupportedFileTypeError,
        FileTooLargeError, CorruptedFileError
    """
    file_type = detect_file_type(file_bytes)
    validate_file(file_bytes, file_type)

    if file_type == "pdf":
        return pdf_bytes_to_images(file_bytes)

    return file_bytes


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 150,
    max_width: int = 1200
     ) -> List[Image.Image]:
    """
    Convert validated PDF bytes to a list of PIL images.

    NOTE:
    All validation has already occurred in validate_file().
    This function ONLY performs conversion.

    Args:
        pdf_bytes (bytes): Raw PDF content.
        dpi (int): Rendering resolution.
        max_width (int): Optional resize threshold for memory efficiency.

    Returns:
        List[PIL.Image.Image]: One image per PDF page.

    Raises:
        CorruptedFileError: If PDF cannot be opened.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise CorruptedFileError("PDF appears corrupted.") from e

    images = []

    for page in doc:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Resize oversized images for performance
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        images.append(img)

    return images
