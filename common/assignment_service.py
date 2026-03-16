import os

import requests
from dotenv import load_dotenv

from common.logger import get_logger

load_dotenv()

log = get_logger(__name__)


class AssignmentService:
    def __init__(self, verify_url: str | None = None):
        self.api_key = os.getenv("AIDEVS_API_KEY")
        if not self.api_key:
            raise ValueError("AIDEVS_API_KEY is not set in environment")

        self.verify_url = verify_url or os.getenv("AIDEVS_VERIFY_URL")
        if not self.verify_url:
            raise ValueError("AIDEVS_VERIFY_URL is not set in environment or constructor")

    def send(self, task: str, answer) -> dict:
        payload = {
            "apikey": self.api_key,
            "task": task,
            "answer": answer,
        }
        log.info(f"Sending payload: {payload}")
        response = requests.post(self.verify_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
