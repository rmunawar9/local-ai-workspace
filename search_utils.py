import urllib.request
import urllib.parse


def execute_fallback_search(query_string):
    try:
        encoded_query = urllib.parse.quote(query_string)
        url = f"https://duckduckgo.com{encoded_query}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )

        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode("utf-8")

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        results = []
        for index, row in enumerate(soup.find_all("a", class_="result__url")):
            if index >= 3:
                break
            snippet_element = row.find_next("span", class_="result__snippet")
            snippet_text = (
                snippet_element.text.strip()
                if snippet_element
                else "View details online."
            )
            link_url = row.get("href", "").strip()

            if "uddg=" in link_url:
                raw_url_part = link_url.split("uddg=")[-1]
                link_url = urllib.parse.unquote(raw_url_part)

            results.append(f"- Data: '{snippet_text}'\n  Source URL: {link_url}")

        return (
            "\n\n".join(results) if results else "No direct matching web records found."
        )
    except Exception as err:
        return f"Scraper notice: {err}"
