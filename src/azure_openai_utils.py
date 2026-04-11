# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Shared Azure OpenAI client setup for the Nelson Mandela Foundation archive pipeline.

Provides a single ``setup_azure_openai()`` helper so that every script in this
repository authenticates to Azure OpenAI in a consistent way using Entra ID
(DefaultAzureCredential) rather than API keys.
"""

import os
import logging

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def setup_azure_openai(api_version="2025-01-01-preview"):
    """Create and return an authenticated AzureOpenAI client.

    Parameters
    ----------
    api_version : str, optional
        Azure OpenAI API version to use (default: ``"2025-01-01-preview"``).

    Returns
    -------
    openai.AzureOpenAI
        A ready-to-use client instance.
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    logger.info("Azure OpenAI client initialized successfully")
    return client
