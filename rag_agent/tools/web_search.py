from typing import List, Dict
import os
import requests
import json
import trafilatura
from bs4 import BeautifulSoup
import re

class WebSearch:
    def __init__(
        self,
        base_url: str = "https://ydc-index.io/v1/search"):
        self.base_url = base_url
        self.api_key = os.getenv("YOU_API_KEY")


    def extract_data(self, URL: str) -> str:
        downloaded = trafilatura.fetch_url(URL)
        if not downloaded:
            return None

        # clean_text = trafilatura.extract(
        #     downloaded,
        #     include_comments=False,
        #     include_tables=False,
        #     include_links=False,
        #     favor_recall=False
        # )

        return downloaded

    def web_search(self, query: str, results_to_extract_count: int = 10) -> Dict:
        """Searches the web for relevant information and extracts clean text.

        This tool is used when internal knowledge retrieval does not sufficiently
        answer the user's query and external, up-to-date information is required.

        Args:
            query: The search query derived from the user's question.
            results_to_extract_count: Number of web results to retrieve and process.

        Returns:
            Success:
            {
                "status": "success",
                "query": "...",
                "results": [
                    {
                        "title": "...",
                        "url": "...",
                    }
                ]
            }

            Error:
            {
                "status": "error",
                "error_message": "..."
            }
        """

        # 🔐 Replace with your actual You.com API key
        API_KEY = "ydc-sk-988fe646a127e2ca-zHOgmT2slT02L28HZttsS5FuHH8VH3Nk-2c7423e1"

        # if not self.api_key:
        #     return {
        #         "status": "error",
        #         "error_message": "Missing YOU_API_KEY environment variable"
        #     }


        # 🌐 Endpoint
        # URL = "https://ydc-index.io/v1/search"

        # 🔎 Query parameters (same as curl)
        params = {
            "query": query,
            "count": results_to_extract_count,
        }

        # 📤 Headers
        headers = {
            "X-API-Key": API_KEY
        }

        # 🚀 Send GET request
        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            return {
                "status": "error",
                "error_message": str(e)
            }

        # ✅ Status
        # print("Status Code:", response.status_code)

        # 📄 Pretty-print JSON response

        if response.status_code != 200:
            # print(response.text)
            return {
                "status": "error",
                "error_message": response.text
            }

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error_message": "Failed to parse JSON response",
            }

        results = []
        results_data = data.get("results", {})

        for item in results_data.get("web", []):
            url = item.get("url")
            if not url:
                continue
            
            results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                })

            # extracted_text = self.extract_data(url)

            # if not extracted_text:
            #     results.append({
            #         "title": item.get("title"),
            #         "url": item.get("url"),
            #         "html": "NOTHING EXTRACTED"
            #     })
            # else:
            #     results.append({
            #         "title": item.get("title"),
            #         "url": item.get("url"),
            #         "html": extracted_text
            #     })



        return {
            "status": "success",
            "query": query,
            "results": results
        }
