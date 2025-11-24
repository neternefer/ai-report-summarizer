"""
Text Extractor Module

Handles preprocessing and analysis of images and PDFs for OCR and visual data
extraction.

Provides:
- Analyze images using Azure Computer Vision features
- Extract and clean OCR results for downstream processing
- Provides a simple interface for obtaining structured analysis results
"""
from typing import Dict, List
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.exceptions import HttpResponseError
from config import AzureConfig


class TextExtractor:
    """
    Provides methods for analyzing images with Azure Computer Vision
    and extracting structured text and visual features.
    """
    def __init__(self):
        self.client = AzureConfig.get_computer_vision_client()

    def analyze_img(self, image_bytes: bytes) -> object:
        """
        Analyze an image using Azure Computer Vision.

        Args:
            image_bytes (bytes): Raw image content.

        Returns:
            object: Azure Vision analysis result object containing captions
                    and OCR text.

        Raises:
            RuntimeError: If Azure Vision analysis fails.
        """
        visual_features = [
                VisualFeatures.CAPTION,
                VisualFeatures.READ
            ]
        try:
            result = self.client.analyze(
                image_data=image_bytes,
                visual_features=visual_features
            )
            return result
        except HttpResponseError as e:
            raise RuntimeError(f"Azure Vision error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"OCR failure: {str(e)}") from e
                          
    def cleaned_result(self, result: object) -> Dict[str, List[str] | str]:
        """
        Extracts and cleans the OCR and caption results from an analysis
        result.

        Args:
            result (object): Azure Vision analysis result from analyze_img.

        Returns:
            Dict[str, List[str] | str]: A dictionary with:
                - 'caption': the extracted image caption or a default message
                - 'text_lines': a list of OCR text lines or a default message
        """
        cleaned = {}
        if result.caption is not None:
            cleaned['caption'] = result.caption.text
        else:
            cleaned['caption'] = "No Caption detected"
        text_lines = []
        if result.read is not None and result.read.blocks:
            for line in result.read.blocks[0].lines:
                text_lines.append(line["text"])
        else:
            text_lines.append("No text lines detected")
        cleaned["text_lines"] = text_lines
        return cleaned
