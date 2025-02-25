import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib
import os
from utils.download import download
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

# generate a set of stop_words from pre-definded text file
def generate_stop_word_set(filename="stop_words.txt"):
    tokens = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                for word in words:
                    tokens.add(word)
    except FileNotFoundError:
        print(f"The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return tokens

def tokenizer(text):
    tokens = []
    t = ""
    for char in text:
        char = char.lower()
        if 'a' <= char <= 'z' or '0' <= char <= '9':
            t += char
        else:
            if t:
                tokens.append(t)  
                t = ''
    if t:
        tokens.append(t)
    return tokens


def update_WordFrequencies(tokens, wordFreq):
    for token in tokens:
        if token not in wordFreq:
            wordFreq[token] = 1
        else:
            wordFreq[token] += 1
    return wordFreq

def sum_columns(matrix):
    if not matrix or not matrix[0]:
        return []
    num_columns = len(matrix[0])
    column_sums = [0] * num_columns
    for row in matrix:
        for i in range(num_columns):
            column_sums[i] += row[i]

    return column_sums

# generate an unique hash value for words
def get_unique_256bits_hash_value_for_word(word):
    # Memmemorizing the hash values to avoid recomputation
    global hash_value_for_words
    if not word or len(word) == 0:
        return None 
    if word in hash_value_for_words:
        return hash_value_for_words[word]
    
    word_bytes = word.encode('utf-8')
    sha256_hash = hashlib.sha256()
    sha256_hash.update(word_bytes)
    hash_value = sha256_hash.hexdigest()  # Hexadecimal hash string
    
    hash_value_for_words[word] = hash_value
    return hash_value

# Function to generate fingerprint for a URL
def generate_fingerprint_for_text(text):
    tokens = tokenizer(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    frequencies = {}
    update_WordFrequencies(filtered_tokens, frequencies)
    
    interpreted_values = []

    for token, frequency in frequencies.items():
        hash_value = get_unique_256bits_hash_value_for_word(token)
        if hash_value is None:
            continue

        # Convert hex hash to binary string
        bits = ''.join(f'{int(c, 16):04b}' for c in hash_value)

        value = [1 * frequency if bit == '1' else -1 * frequency for bit in bits]
        interpreted_values.append(value)

    sum_values = sum_columns(interpreted_values)
    
    # Generate fingerprint based on sum values
    fingerprint = ''.join('1' if bit_sum > 0 else '0' for bit_sum in sum_values)

    return fingerprint
    
def distance(hash1, hash2, bits):
    distance = 0
    for i in range(bits):
        if hash1[i] != hash2[i]:
            distance += 1
    return distance
    
def detect_similarity(text, bits =256, threshold=20):
    global fingerprints
    page_hash = generate_fingerprint_for_text(text)
    if len(page_hash) < bits: 
        return True
    for other_hash in fingerprints:
            if distance(page_hash, other_hash, bits) <= threshold:
                return True
    fingerprints.add(page_hash)
    return False

def remove_fragment(url):
    if url:
        return url.split("#")[0]
    else:
        return None
    
# detect pages with no infomation or long page
def is_grabage_or_high(text):
    garbage_pattern = r'\ufffd'
    if re.search(garbage_pattern , text) or len(text) > 100000:
        return True
    return False


def get_robots_parser(url, config, logger, frontier):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    decode_domain = parsed_url.netloc.lower()
    if isinstance(decode_domain, bytes):  # Convert bytes to string if needed
        decode_domain = decode_domain.decode("utf-8")
    if isinstance(scheme, bytes):
        scheme = scheme.decode("utf-8")
    domain = f"{scheme}://{decode_domain}"
    if domain in robots_parsers:
        return robots_parsers[domain]  # Use cached parser if available

    # Use the download function to fetch robots.txt
    robots_url = f"{domain}/robots.txt"
    response = download(robots_url, config, logger) # fecth the robots.txt through downloaded.

    if response and response.raw_response and response.raw_response.content:
        try:
            # Parse the robots.txt content manually
            robots_content = response.raw_response.content.decode("utf-8")
            rules = parse_robots_txt(robots_content)  # Custom parser
            robots_parsers[domain] = rules  # Cache the parsed rules

            # Extract sitemap URLs from robots.txt
            sitemap_urls = get_sitemap_urls_from_robots(robots_content)
            for sitemap_url in sitemap_urls:
                process_sitemap(sitemap_url, config, logger, frontier)  # Process sitemap'''
            
            return rules
        except Exception as e:
            return None
    else:
        return None

def parse_robots_txt(content):
    rules = {"allow": [], "disallow": []}
    user_agent = "*"  # Default user-agent (applies to all crawlers)

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse User-agent directive
        if line.lower().startswith("user-agent:"):
            user_agent = line.split(":", 1)[1].strip()
            continue

        # Parse Allow and Disallow directives
        if user_agent == "*":  # Only process rules for the default user-agent
            if line.lower().startswith("allow:"):
                path = line.split(":", 1)[1].strip()
                rules["allow"].append(path)
            elif line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                rules["disallow"].append(path)

    return rules

def is_allowed_by_robots(url, rules):
    if not rules:
        return True  # Assume allowed if no rules are available

    parsed_url = urlparse(url)
    path = parsed_url.path

    # Check allow rules
    for allowed_path in rules["allow"]:
        if path.startswith(allowed_path):
            return True  # URL is allowed

    # Check disallow rules first
    for disallowed_path in rules["disallow"]:
        if path.startswith(disallowed_path):
            return False  # URL is disallowed

    # Default to allowed if no specific rule matches
    return True

def get_sitemap_urls_from_robots(robots_content):
    sitemap_urls = []
    for line in robots_content.splitlines():
        line = line.strip()
        if line.lower().startswith("sitemap:"):
            sitemap_url = line.split(":", 1)[1].strip()
            sitemap_urls.append(sitemap_url)
    return sitemap_urls

def process_sitemap(sitemap_url, config, logger, frontier):
    response = download(sitemap_url, config, logger)
    if response and response.raw_response and response.raw_response.content:
        sitemap_content = response.raw_response.content.decode("utf-8")
        try:
            # Check if it's a sitemap index
            if "sitemapindex" in sitemap_content.lower():
                sitemap_urls = parse_sitemap_index(sitemap_content)
                for url in sitemap_urls:
                    process_sitemap(url, config, logger, frontier)  # Recursively process nested sitemaps
            else:
                urls = parse_xml_sitemap(sitemap_content)
                for url in urls:
                    if is_valid(url):  # Check if the URL is valid
                        frontier.add_url(remove_fragment(url))  # Add the URL to the crawl queue
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")

def parse_xml_sitemap(sitemap_content):
    urls = []
    root = ET.fromstring(sitemap_content)
    for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        urls.append(elem.text)
    return urls

def parse_sitemap_index(sitemap_content):
    sitemap_urls = []
    root = ET.fromstring(sitemap_content)
    for sitemap in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
        loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
        if loc:
            sitemap_urls.append(loc)
    return sitemap_urls

# local variables
fingerprints = set()
hash_value_for_words = {} 
stop_words = generate_stop_word_set()
word_frequencies = {}
longest_content = {
    'url': None,
    'word_count': 0
}
unique_urls = set()
subdomains = {}
robots_parsers = {} # Dictionary to store robots.txt parsers for different domains


def scraper(url, resp, config, logger, frontier):
    if not resp or not resp.raw_response:
        return []

     # Handle redirects
    if resp.status in {301, 302, 307, 308}:  # Check for redirect status codes
        new_url = resp.raw_response.headers.get("Location")  # Get the new URL
        if new_url:
            new_url = remove_fragment(new_url)  # Remove fragments
            if is_valid(new_url):
                return [new_url]  # Return the new URL for crawling
        return []
    
    if resp.status != 200:
        return []

    if not resp.raw_response.content:
        return []
    
    if "text/html" not in resp.raw_response.headers.get("Content-Type", "").lower():
        return []
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()  # Ensure it's a string
        if isinstance(domain, bytes):  # Convert bytes to string if needed
            domain = domain.decode("utf-8")
        domain_parts = domain.split('.')
        if len(domain_parts) > 3 and '.'.join(domain_parts[-3:]) == 'ics.uci.edu':
            if domain not in subdomains:
                subdomains[domain] = 1
            else:
                subdomains[domain] += 1

    except TypeError:
        print ("TypeError for ", parsed)
        raise
    
    
    soup = BeautifulSoup(resp.raw_response.content, 'html.parser')    

    for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav', 'aside', 'form', 'button']):
        tag.extract() 

    text = soup.get_text(separator=" ")
    
    # check for grabage contents and too high textual information
    if is_grabage_or_high(text):
        return []
    # chech similarity with havs seen urls
    if detect_similarity(text):
        return []

    tokens = tokenizer(text)

    # add url to uniqur_urls after similarity check
    # check for the value of information. A high value page must contains 100 meaningful words at least
    if len(tokens) > 50:
        unique_urls.add(url)

    # update the url with longest content
    content_size = len(tokens)
    if content_size > longest_content['word_count']:
        longest_content['url'] = url
        longest_content['word_count'] = content_size

    meaningful_tokens = [token for token in tokens if token not in stop_words and len(token)>2]
    
    update_WordFrequencies(meaningful_tokens, word_frequencies)
    
    allowed_links = []
    links = extract_next_links(url, resp)
    # Get robots.txt parser
    for link in links:
        rule = get_robots_parser(link, config, logger, frontier)
        if is_allowed_by_robots(link, rule):  # Check if crawling is allowed
            allowed_links.append(link)
    return [link for link in allowed_links if is_valid(link)]

def extract_next_links(url, resp):
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    if resp.status != 200 or not resp.raw_response:
        return []

    try:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(resp.raw_response.content, "html.parser")
        links = [remove_fragment(a['href']) for a in soup.find_all('a', href=True)]

        # Extract links from other elements that might contain clickable links
        for element in soup.find_all(['button', 'div', 'span', 'li', 'img', 'form']):
            if element.get('onclick'):  # Handle JavaScript onclick events
                onclick = element.get('onclick')
                # Extract URLs from common JavaScript patterns (e.g., window.location.href)
                js_urls = re.findall(r"window\.location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick)
                links.extend(js_urls)

            if element.get('data-href'):  # Handle data-href attributes
                links.append(remove_fragment(element.get('data-href')))

        # Normalize and filter URLs, transform relative to absolute URL
        base_url = resp.raw_response.url
        normalized_links = []
        for link in links:
            try:
                # Check if the link is already an absolute URL
                parsed_link = urlparse(link)
                if parsed_link.scheme and parsed_link.netloc:
                    # The link is already absolute, so no transformation is needed
                    full_url = link
                else:
                    # The link is relative, so transform it to an absolute URL
                    full_url = urljoin(base_url, link)
                # Add the normalized URL to the list
                normalized_links.append(full_url)
            except Exception as e:
                print(f"Error normalizing URL {link}: {e}")

        return normalized_links

    except Exception as e:
        print(f"Error extracting links from {url}: {e}")
        return []


def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()  # Ensure it's a string

        if isinstance(domain, bytes):  # Convert bytes to string if needed
            domain = domain.decode("utf-8")

        allowed_domains = [".ics.uci.edu", ".cs.uci.edu", ".informatics.uci.edu", ".stat.uci.edu"]

        if parsed.scheme not in set(["http", "https"]):
            return False

        if not any(domain.endswith(allowed) for allowed in allowed_domains):
            return False
        
        # Avoid Crawling Login & Logout Pages
        if re.search(r"(login|logout|signup|register|youtube|facebook|twitter)", url.lower()):
            return False
        
        # Avoid Crawling Query Parameters That Suggest Duplication
        if re.search(r"(action|utm_|sessionid|ref)", parsed.query.lower()):
            return False
        
        # Prevent Infinite Crawling of Calendar & Search Pages
        if re.search(r"(calendar|search|results|event)", parsed.path.lower()): # delete the |event|date=, need to focus on this if the crawler run into the trap.
            return False
        
        if re.search(r"(.pdf|.zip|.mat|.odc|.bed|.bw|.bigwig|.ppsx|.tsv|.sql|.npy|.gctx|.bam|.nb)", url.lower()):
            return False
        
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower())

    except TypeError:
        print ("TypeError for ", parsed)
        raise

def report():
    if not os.path.exists("Logs"):
        os.makedirs("Logs")

    # Write the total number of unique URLs
    with open("Logs/unique_urls.txt", "w") as f:
        f.write(f"Total unique URLs: {len(unique_urls)}\n")
        f.write(f"longest page: {longest_content['url']}, word count: {longest_content['word_count']}\n")
        sorted_subdomains = dict(sorted(subdomains.items(), key=lambda x: (-x[1], x[0])))
        f.write("++++++++++++ ics.uci.edu subdomains ++++++++++++++++++ \n")
        for key, value in sorted_subdomains.items():
            f.write(f"https://{key},{value}\n")
        sorted_wordFreq = dict(sorted(word_frequencies.items(), key=lambda x: (-x[1], x[0])))
        count = 0
        f.write("++++++++++++ word frequency ++++++++++++++++++ \n")
        for key, value in sorted_wordFreq.items():
            f.write(f"{key}\t{value}\n")
            count += 1
            if count == 50:
                break