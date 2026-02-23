# ==========================================
# web_scraper.py
# ==========================================
import requests
import re
from bs4 import BeautifulSoup

class SimpleWebScraper:
    """Fetches and cleans webpage text given a list of URLs."""

    def scrape(self, urls):
        results = {}
        for url in urls:
            try:
                response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove scripts, styles, navbars, etc.
                    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                        tag.decompose()

                    title = soup.title.string.strip() if soup.title else ""
                    text = " ".join(soup.stripped_strings)
                    text = re.sub(r"\s+", " ", text)
                    if len(text) < 500:  # skip trivial junk pages
                        continue
                    results[url] = {"title": title, "text": text[:10000]}  # cap text length for safety

                    print(f"✅ Scraped: {url}")
                    print(f"   Title: {title[:80]}...")
                else:
                    print(f"[Warning] {url} returned status {response.status_code}")
            except Exception as e:
                print(f"[Error] Failed to scrape {url}: {e}")
        return results
