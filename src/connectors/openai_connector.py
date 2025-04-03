from openai import AzureOpenAI
import openai
import certifi
import httpx

import logging
import os
from time import sleep

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)

last_string = "<|END_OF_ANSWER|>"

def predict_with_gpt_4(prompt):
    # Get API key from environment variable
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Set up retry mechanism
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Make the API call to GPT-4
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies PII entities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            retry_count += 1
            if retry_count == max_retries:
                _logger.error(f"Failed after {max_retries} retries: {str(e)}")
                raise
            _logger.warning(f"API error: {str(e)}. Retrying ({retry_count}/{max_retries})...")
            sleep(2 ** retry_count)  # Exponential backoff
    response_str = response.choices[0].message.content
    return response_str