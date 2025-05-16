#!/usr/bin/env python
"""
Script to ethically scrape medical transcription samples from MTSamples.com.
Respects rate limits and follows good web citizenship practices.
This version disables SSL verification (use with caution).

IMPORTANT: Check the website's terms of service before running this script.
Only run if scraping is allowed by the website's terms.
"""
import os
import time
import random
import argparse
import warnings
import requests
from bs4 import BeautifulSoup
import json
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrape_mt_samples(output_dir, max_samples=50, delay=1):
    """
    Scrape samples from MTSamples.com and save to files.
    SSL verification is disabled in this script version.
    
    Args:
        output_dir: Directory to save samples
        max_samples: Maximum number of samples to scrape per category
        delay: Delay between requests (seconds) to avoid overloading the server
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Categories to scrape
    categories = [
        'discharge-summary', 
        'soap-notes', 
        'consult-h-p',
        'emergency-room-reports',
        'progress-notes',
        'radiology-reports'
    ]
    
    # Track all samples and metadata
    all_samples = []
    sample_count = 0
    
    # User agents to rotate (to be a good citizen)
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    
    print(f"Starting to scrape MTSamples.com (SSL verification disabled)...")
    
    # Check if robots.txt allows scraping
    try:
        robots_response = requests.get("https://www.mtsamples.com/robots.txt", verify=False)
        if robots_response.status_code == 200 and "Disallow: /" in robots_response.text:
            print("WARNING: The robots.txt file may disallow scraping. Please check manually.")
            proceed = input("Do you want to proceed anyway? (yes/no): ")
            if proceed.lower() != "yes":
                print("Aborting scrape operation.")
                return
    except requests.exceptions.SSLError:
        print("SSL Error while checking robots.txt - continuing without verification")
    except Exception as e:
        print(f"Error checking robots.txt: {e}")
    
    # Process each category
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Get category page
        url = f"https://www.mtsamples.com/site/pages/browse.asp?type={category}&page=1"
        headers = {'User-Agent': random.choice(user_agents)}
        
        try:
            response = requests.get(url, headers=headers, verify=False)
            response.raise_for_status()  # Raise exception for bad status codes
        except requests.exceptions.RequestException as e:
            print(f"Error accessing category {category}: {e}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find sample links
        sample_links = []
        
        # This is a placeholder selector - you'll need to inspect the website to find the correct one
        links = soup.find_all('a')
        for link in links:
            href = link.get('href', '')
            if 'sample' in href and '.asp' in href:
                sample_links.append(href)
        
        print(f"Found {len(sample_links)} potential sample links")
        
        # Limit the number of samples per category
        samples_to_process = min(max_samples, len(sample_links))
        
        # Process each sample
        for i, link in enumerate(sample_links[:samples_to_process]):
            # Construct full URL if needed
            if not link.startswith('http'):
                if link.startswith('/'):
                    sample_url = f"https://www.mtsamples.com{link}"
                else:
                    sample_url = f"https://www.mtsamples.com/{link}"
            else:
                sample_url = link
            
            print(f"Fetching sample {i+1}/{samples_to_process}: {sample_url}")
            
            # Get sample page
            headers = {'User-Agent': random.choice(user_agents)}
            
            try:
                sample_response = requests.get(sample_url, headers=headers, verify=False)
                sample_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error accessing sample {sample_url}: {e}")
                continue
            
            sample_soup = BeautifulSoup(sample_response.text, 'html.parser')
            
            # Extract sample text - this selector needs to be adjusted based on site structure
            sample_text = ""
            # Try different potential selectors
            for selector in ['.sample-content', '#sample-content', '.report-content', '#report-content']:
                content_elem = sample_soup.select_one(selector)
                if content_elem:
                    sample_text = content_elem.get_text(strip=True)
                    break
            
            # If no selector worked, try a more general approach
            if not sample_text:
                # Look for the largest text block
                paragraphs = sample_soup.find_all('p')
                if paragraphs:
                    # Find paragraph with most text
                    sample_text = max(paragraphs, key=lambda p: len(p.get_text())).get_text(strip=True)
            
            # Extract title
            title = "Unknown Sample"
            title_elem = sample_soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True).replace("MT Sample: ", "")
            
            # If still no content, skip this sample
            if not sample_text or len(sample_text) < 100:  # At least 100 chars to be valid
                print(f"Could not extract meaningful content from {sample_url}")
                continue
            
            # Extract metadata if available
            metadata = {
                "category": category,
                "title": title,
                "url": sample_url,
                "scrape_date": time.strftime("%Y-%m-%d")
            }
            
            # Save sample text
            sample_filename = f"{category}_{i+1}.txt"
            with open(os.path.join(output_dir, sample_filename), "w", encoding="utf-8") as f:
                f.write(sample_text)
            
            # Save metadata
            metadata_filename = f"{category}_{i+1}_meta.json"
            with open(os.path.join(output_dir, metadata_filename), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Add to overall collection
            all_samples.append({
                "filename": sample_filename,
                "metadata": metadata,
                "length": len(sample_text)
            })
            
            sample_count += 1
            
            # Be nice to the server
            time.sleep(delay + random.random())
    
    # Save index of all samples
    index_file = os.path.join(output_dir, "sample_index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\nScraping complete. Collected {sample_count} samples across {len(categories)} categories.")
    print(f"Samples saved to {output_dir}")
    print(f"Sample index saved to {index_file}")


def main():
    """Parse arguments and run scraper."""
    parser = argparse.ArgumentParser(description="Scrape medical transcription samples (SSL verification disabled)")
    
    parser.add_argument("--output_dir", type=str, default="mt_samples",
                      help="Directory to save samples")
    parser.add_argument("--max_samples", type=int, default=10,
                      help="Maximum samples per category")
    parser.add_argument("--delay", type=float, default=2.0,
                      help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    # Print warning and get confirmation
    print("=" * 80)
    print("WARNING: Web scraping may be against a website's terms of service.")
    print("Make sure you have checked the website's terms and have permission to scrape.")
    print("This script is provided for educational purposes only.")
    print("SSL VERIFICATION IS DISABLED - this comes with security risks.")
    print("Use at your own risk and responsibility.")
    print("=" * 80)
    
    consent = input("I have checked the terms of service and wish to proceed (yes/no): ")
    if consent.lower() != "yes":
        print("Scraping aborted.")
        return
    
    scrape_mt_samples(args.output_dir, args.max_samples, args.delay)


if __name__ == "__main__":
    main()