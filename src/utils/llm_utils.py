import os
from typing import Dict, Any, Optional
import logging
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    """Structure for LLM response."""
    analysis: str = Field(description="Analysis of the data")
    recommendations: list = Field(description="List of recommendations")
    confidence: float = Field(description="Confidence score between 0 and 1")

def call_llm(prompt: str, config: Optional[Dict] = None) -> Dict:
    """
    Call LLM with the given prompt using Azure OpenAI.
    
    Args:
        prompt (str): Prompt to send to LLM
        config (dict, optional): Configuration for LLM
        
    Returns:
        dict: LLM response
    """
    try:
        # Get Azure OpenAI configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION")
        azure_endpoint = os.getenv("OPENAI_ENDPOINT")
        deployment_name = os.getenv("OPENAI_DEPLOYMENT")
        
        if not all([api_key, api_version, azure_endpoint, deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        # Initialize parser
        parser = PydanticOutputParser(pydantic_object=LLMResponse)
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a data analysis expert. Analyze the following data and provide insights."),
            ("human", "{input}\n{format_instructions}")
        ])
        
        # Format prompt
        formatted_prompt = prompt_template.format_messages(
            input=prompt,
            format_instructions=parser.get_format_instructions()
        )
        
        # Initialize Azure OpenAI chat model with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                chat = AzureChatOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    deployment_name=deployment_name,
                    model="gpt-4o",
                    temperature=config.get('temperature', 0.3) if config else 0.3,
                    max_retries=2,
                    timeout=30
                )
                
                # Get response
                response = chat.invoke(formatted_prompt)
                
                # Parse response
                parsed_response = parser.parse(response.content)
                
                return {
                    'analysis': parsed_response.analysis,
                    'recommendations': parsed_response.recommendations,
                    'confidence': parsed_response.confidence
                }
                
            except Exception as e:
                if "403" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Azure OpenAI 403 error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise
        
        # If we get here, all retries failed
        raise Exception("Failed to connect to Azure OpenAI after multiple retries")
        
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        # Return a default response instead of raising
        return {
            'analysis': "Unable to analyze data due to LLM connection error.",
            'recommendations': ["Please check your Azure OpenAI configuration and internet connection."],
            'confidence': 0.0
        }

def format_llm_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given variables.
    
    Args:
        template (str): The prompt template
        **kwargs: Variables to format the template with
        
    Returns:
        str: The formatted prompt
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing key in prompt template: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        raise 