"""
Azure Configuration Module

Handles centralized configuration management for Azure services:
- Computer Vision
- OpenAI
- Document Intelligence
- Blob Storage

Provides:
- Environment variable loading
- Credentials validation
- Client initialization
- Full configuration check
"""
import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.exceptions import AzureError, HttpResponseError
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

load_dotenv()


class AzureConfig:
    """
    Centralized Azure Configuration and Client Manager.

    Handles environment variables, credential validation, and client creation
    for Azure services used in the application.

    Responsibilities:
    - Validate configurations for Computer Vision, OpenAI and Blob Storage
    - Provide initialized clients for each service
    - Centralize credentials and endpoint management
    """
    COMPUTER_VISION_ENDPOINT = os.getenv('COMPUTER_VISION_ENDPOINT')
    COMPUTER_VISION_KEY = os.getenv('COMPUTER_VISION_KEY')

    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")

    AZURE_STORAGE_ENDPOINT = os.getenv("AZURE_STORAGE_ENDPOINT")
    AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
    AZURE_STORAGE_CONNECTION = os.getenv("AZURE_STORAGE_CONNECTION")

    BLOB_NAME = os.getenv("BLOB_NAME")
    CONTAINER_NAME = os.getenv("CONTAINER_NAME")

    @staticmethod
    def validate_computer_vision_config():
        """
        Validate that the Azure Computer Vision configuration is correctly set.

        Checks that endpoint and key are present and endpoint starts with 
        'https://'.

        Raises:
            ValueError: If configuration is missing or invalid.

        Returns:
            bool | None: True if configuration is valid, None if validation 
            fails.
        """
        if (not AzureConfig.COMPUTER_VISION_ENDPOINT and not 
                AzureConfig.COMPUTER_VISION_KEY):
            raise ValueError("Please set Computer Vision credentials.")
        if not AzureConfig.COMPUTER_VISION_ENDPOINT.startswith('https://'):
            raise ValueError("Computer Vision endpoint starts with https://")
        return True
    
    @staticmethod
    def validate_openai_config():
        """
        Validate that the Azure OpenAI configuration is correctly set.

        Checks that endpoint and key are present.

        Raises:
            ValueError: If configuration is missing or invalid.

        Returns:
            bool | None: True if configuration is valid, None if validation
            fails.
        """
        if not AzureConfig.OPENAI_ENDPOINT and not AzureConfig.OPENAI_KEY:
            raise ValueError("Please set Azure OpenAI credentials.")
        return True
    
    @staticmethod
    def validate_blob_service_config():
        """
        Validate that the Blob Service configuration is correctly set.

        Checks for storage endpoint, key, or connection string.

        Raises:
            ValueError: If configuration is missing or invalid.

        Returns:
            bool | None: True if configuration is valid, None if validation
            fails.
        """
        if (not AzureConfig.AZURE_STORAGE_ENDPOINT and not
                AzureConfig.AZURE_STORAGE_KEY):
            raise ValueError("Please set Blob Service credentials.")
        return True
    
    @staticmethod
    def validate_blob_storage_config():
        """
        Validate that the Azure Blob Storage configuration is correctly set.

        Checks for storage endpoint, key, or connection string.

        Raises:
            ValueError: If configuration is missing or invalid.

        Returns:
            bool | None: True if configuration is valid, None if validation
            fails.
        """
        if not AzureConfig.CONTAINER_NAME and not AzureConfig.BLOB_NAME:
            raise ValueError("Please set Blob Storage credentials.")
        return True
    
    @staticmethod
    def get_computer_vision_client():
        """
        Initializes and returns an Azure Computer Vision client.

        The method validates the configuration first. If initialization fails,
        the exception is caught and printed, and None is returned.

        Returns:
            ImageAnalysisClient | None: Initialized client instance, 
            or None if initialization fails.
        """
        try:
            AzureConfig.validate_computer_vision_config()
            endpoint = AzureConfig.COMPUTER_VISION_ENDPOINT
            credential = AzureKeyCredential(AzureConfig.COMPUTER_VISION_KEY)
            client = ImageAnalysisClient(
                endpoint=endpoint, 
                credential=credential
                )
            return client
        except ValueError as e:
            print("Configuration error:", e)
        except HttpResponseError as e:
            print("HTTP error:", e)
        except AzureError as e:
            print("Azure SDK error:", e)
        return None

    @staticmethod
    def get_openai_client():
        """
        Initializes and returns an Azure OpenAI client.

        The method validates the configuration first. If initialization fails,
        the exception is caught and printed, and None is returned.

        Returns:
            AzureOpenAIClient | None: Initialized client instance, 
            or None if initialization fails.
        """
        try:
            AzureConfig.validate_openai_config()
            client = AzureOpenAI(
                api_key=AzureConfig.OPENAI_KEY,
                azure_endpoint=AzureConfig.OPENAI_ENDPOINT,
                azure_deployment=AzureConfig.OPENAI_DEPLOYMENT,
                api_version="2024-12-01-preview"
            )
            return client
        except ValueError as e:
            print("Configuration error:", e)
        except HttpResponseError as e:
            print("HTTP error:", e)
        except AzureError as e:
            print("Azure SDK error:", e)
        return None
             
    @staticmethod
    def get_blob_service_client():
        """
        Initializes and returns an Blob Service client.

        The method validates the configuration first. If initialization fails,
        the exception is caught and printed, and None is returned.

        Returns:
            BlobServiceClient | None: Initialized client instance, 
            or None if initialization fails.
        """
        try:
            endpoint = AzureConfig.AZURE_STORAGE_ENDPOINT
            blob_credential = AzureConfig.AZURE_STORAGE_KEY
            client = BlobServiceClient(
                account_url=endpoint,
                credential=blob_credential)
            return client
        except ValueError as e:
            print("Configuration error:", e)
        except HttpResponseError as e:
            print("HTTP error:", e)
        except AzureError as e:
            print("Azure SDK error:", e)
        return None
            
    @staticmethod
    def get_blob_client():
        """
        Initializes and returns a BlobClient for a specific container and blob.

        Notes:
            The blob name can be dynamic; this code uses a placeholder for
            presentation purposes.

        Returns:
            BlobClient | None: Initialized client instance, 
            or None if initialization fails.
        """
        try:
            service_client = AzureConfig.get_blob_service_client()
            container = AzureConfig.CONTAINER_NAME
            blob_name = AzureConfig.BLOB_NAME
            client = service_client.get_blob_client(
                container=container,
                blob=blob_name
            )
            return client
        except ValueError as e:
            print("Configuration error:", e)
        except HttpResponseError as e:
            print("HTTP error:", e)
        except AzureError as e:
            print("Azure SDK error:", e)
        return None


def check_all_configs():
    """
    Perform a complete validation of all Azure service configurations.

    Validates configuration and attempts client initialization for:
    - Computer Vision
    - OpenAI
    - Blob Storage (single blob client)
    - Blob Storage service client

    Returns:
        bool | None: True if all services are configured correctly and 
        clients initialized.
    """
    configs = [
        ("Vision", AzureConfig.get_computer_vision_client),
        ("OpenAI", AzureConfig.get_openai_client),
        ("Azure Storage Blob", AzureConfig.get_blob_client),
        ("Service Blob", AzureConfig.get_blob_service_client)
    ]
    
    all_initialized = True
    for id, (name, client_function) in enumerate(configs, start=1):
        print(f"{id}. {name} Configuration:")
        try:
            client_function()
            print("OK")
        except ValueError as e:
            print(f"Error: {e}")
            all_initialized = False

    if all_initialized:
        print("All client initialized successfully\n")
    else:
        print("Invalid configuration - not all clients initialized\n")
    
    return all_initialized


if __name__ == "__main__":
    """
    Run configuration check when module is executed directly
    """
    print("Configuration Checker")
    print("=" * 50)
    
    if check_all_configs():
        print("\n✓ Configuration is ready for use!")
    else:
        print("\n✗ Configuration incomplete. Please check .env file.")
        
