"""
Azure Blob Storage Interface

Provides simplified upload and access methods for user-uploaded files.

Responsibilities:
- Initialize BlobServiceClient for the configured storage account
- Upload bytes and automatically generate blob names
- Return accessible blob URLs suitable for OpenAI completions
"""
import uuid
from config import AzureConfig


class Storage:
    """
    Handles uploading files to Azure Blob Storage and generating
    accessible URLs.
    """

    def __init__(self):
        self.service_client = AzureConfig.get_blob_service_client()

    def _generate_uuid_name(self, extension: str | None) -> str:
        """
        Generates a UUID-based blob filename.

        Args:
            extension (str | None): Optional file extension (e.g., 'png').

        Returns:
            str: Generated filename.
        """
        ext = f".{extension.lstrip('.')}" if extension else ""
        return f"{uuid.uuid4()}{ext}"

    def upload_bytes_and_get_url(self, file_bytes: bytes,
                                 extension: str | None = None) -> str:
        """
        Uploads bytes to Azure Blob Storage with a UUID-based filename and
        returns the accessible URL.

        Args:
            file_bytes (bytes): Content to upload.
            extension (str | None): Optional extension without the dot
                (e.g., 'png', 'jpg', 'pdf').

        Returns:
            str: Public URL of uploaded blob.
        """
        filename = self._generate_uuid_name(extension)

        client = self.service_client.get_blob_client(
            container=AzureConfig.CONTAINER_NAME,
            blob=filename
        )

        client.upload_blob(file_bytes, overwrite=True)
        return client.url
