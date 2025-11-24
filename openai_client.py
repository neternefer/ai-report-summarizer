"""
OpenAI Interpreter for Chart and Text Analysis

Wraps Azure OpenAI API calls to produce structured summaries from image OCR
and captions.
"""
from config import AzureConfig


class Interpreter:
    """
    Handles prompt building and interpretation of OCR and image data
    using Azure OpenAI chat completion API.
    """

    def __init__(self):
        """Initialize the Interpreter with an Azure OpenAI client."""
        self.client = AzureConfig.get_openai_client()

    def build_interpretation_prompt(self, cleaned: dict,
                                    image_url: str) -> tuple[str, str]:
        """
        Builds a system message and user prompt for OpenAI based on OCR
        results and image URL.

        Args:
            cleaned (dict): Dictionary with keys 'caption' and 'text_lines'.
            image_url (str): URL of the uploaded image/PDF page.

        Returns:
            tuple[str, str]: (system_message, user_prompt)
        """
        caption = cleaned.get("caption", "No caption detected")
        text = "\n".join(cleaned.get("text_lines", ["No text detected"]))

        system_message = (
            "You are a helpful data-analysis assistant.\n"
            "Your input consists of:\n"
            "1. OCR text extracted from an image\n"
            "2. A URL pointing to the image itself\n\n"
            "Your task is to:\n"
            "- interpret both the visual content (from URL) and the OCR text\n"
            "- produce a structured, factual summary\n"
            "- Do NOT hallucinate values that aren't present in the data"
        )

        prompt = (
            f"Caption: {caption}\n"
            f"Text: {text}\n"
            f"Image URL: {image_url}\n\n"
            "Use both the OCR text and visual information from the image URL "
            "to create a factual summary."
        )

        return system_message, prompt

    def interpret_data(self, system_message: str, prompt: str) -> str:
        """
        Sends the system message and user prompt to Azure OpenAI
        chat completion and retrieves the summary.

        Args:
            system_message (str): System-level instructions for the model.
            prompt (str): User-level content containing OCR text and image URL.

        Returns:
            str: Generated summary text from OpenAI.
        """
        response = self.client.chat.completions.create(
            model=AzureConfig.OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        content = response.choices[0].message.content
        return content
