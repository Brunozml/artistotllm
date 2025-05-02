import requests
from bs4 import BeautifulSoup
import os
import time
from pathlib import Path
import re

def clean_filename(title):
    """Convert title to a valid filename."""
    return re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '_')

def scrape_essay(url):
    """Scrape a single essay's content."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Most essays have their content in font tags
        content = soup.find('font')
        if content:
            return content.get_text()
        return soup.get_text()  # Fallback to full text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    # Create raw directory if it doesn't exist
    raw_dir = Path('data/raw/pg_essays')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Scrape main articles page
    base_url = 'http://www.paulgraham.com'
    response = requests.get(f'{base_url}/articles.html')
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links that point to essays
    links = soup.find_all('a')
    essay_count = 0
    
    for link in links:
        href = link.get('href')
        if not href or not href.endswith('.html'):
            continue
            
        title = link.get_text().strip()
        if not title:  # Skip links without text
            continue
            
        # Construct full URL if needed
        if not href.startswith('http'):
            url = f'{base_url}/{href}'
        else:
            url = href
            
        print(f"Scraping: {title}")
        content = scrape_essay(url)
        
        if content:
            filename = clean_filename(title)
            filepath = raw_dir / f"{filename}.txt"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            essay_count += 1
            # Be nice to the server
            time.sleep(1)
    
    print(f"\nFinished scraping {essay_count} essays.")

if __name__ == '__main__':
    main()
