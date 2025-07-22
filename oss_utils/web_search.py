import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo web search and return a list of results."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.post(url, data=params, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.find_all("a", class_="result__a", limit=max_results):
        title = result.get_text()
        link = result["href"]
        results.append({"title": title, "url": link})
    return results

# Pluggable backend
class WebSearchBackend:
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        return duckduckgo_search(query, max_results)

# In the future, you can add Google/Bing backends and switch here 