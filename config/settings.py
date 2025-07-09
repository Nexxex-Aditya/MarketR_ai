import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_base=os.getenv("OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
)
