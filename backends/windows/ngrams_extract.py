import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from collections import Counter
import nltk
import string

# Download necessary NLTK data if you haven't already
# Uncomment these lines on first run
# nltk.download('punkt')
# nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    """
    Cleans the extracted text by removing excessive whitespace,
    URLs, references, and other common academic paper artifacts.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove citations like [1], [2,3], etc.
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    
    # Remove references section (often starts with "References")
    if "References" in text:
        text = text.split("References")[0]
    
    # Remove headers/footers that often contain page numbers
    text = re.sub(r'\b(Page|pg\.)\s*\d+\s*of\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove common academic paper sections that aren't useful for content extraction
    for section in ["Abstract", "Introduction", "Conclusion", "Acknowledgments", "Appendix"]:
        text = re.sub(fr'\b{section}\b', '', text)
    
    # Clean up excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_valid_ngram(ngram, custom_stopwords):
    """
    Determines if an n-gram is valid and meaningful.
    
    Args:
        ngram (str): The n-gram to check
        custom_stopwords (set): Set of custom stopwords
        
    Returns:
        bool: True if the n-gram is valid, False otherwise
    """
    # Check if the n-gram contains any of the custom stopwords
    if any(word in custom_stopwords for word in ngram.split()):
        return False
    
    # Check if the n-gram is just a number or date
    if re.match(r'^[\d\s\.\-\/]+$', ngram):
        return False
    
    # Check if the n-gram is too short (less than 3 characters)
    if len(ngram) < 3:
        return False
    
    # Check if n-gram is made up solely of punctuation and whitespace
    if all(char in string.punctuation + ' ' for char in ngram):
        return False
    
    # Avoid n-grams that contain a lot of punctuation or non-alphabetic chars
    punct_count = sum(1 for char in ngram if char in string.punctuation)
    if punct_count > len(ngram) / 3:  # More than 1/3 is punctuation
        return False
    
    return True

def extract_ngrams(text, n_values=[1, 2, 3], top_n=3):
    """
    Extracts the top n-grams for multiple values of n with better filtering.
    
    Args:
        text (str): The text to extract n-grams from
        n_values (list): List of n values to extract (e.g., [1, 2, 3] for unigrams, bigrams, trigrams)
        top_n (int): Number of top n-grams to extract for each n value
        
    Returns:
        dict: Dictionary mapping n to list of top n-grams
    """
    # Tokenize the text
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Get stopwords and extend with custom academic paper terms
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords - terms that are common in academic papers but not informative
    custom_stopwords = stop_words.union({
        # General paper vocabulary
        'paper', 'figure', 'table', 'et', 'al', 'arxiv', 'url', 'https', 'org', 'abs',
        'section', 'review', 'using', 'used', 'use', 'method', 'work', 'approach',
        'proposed', 'propose', 'results', 'result', 'show', 'shows', 'shown',
        # URLs and identifiers
        'http', 'https', 'www', 'com', 'org', 'net', 'doi', 'arxiv', 'fig', 'eq',
        # Citation/reference terms
        'et', 'al', 'pp', 'vol', 'volume', 'journal', 'conference', 'proceedings', 
        'ieee', 'acm', 'springer', 'elsevier', 'publication', 'publications',
        # Numbers/dates/common measurements
        'one', 'two', 'three', 'first', 'second', 'third', 'fourth', 'fifth',
        'i', 'ii', 'iii', 'iv', 'v', 'january', 'february', 'march', 'april', 'may',
        'june', 'july', 'august', 'september', 'october', 'november', 'december',
        # Common filler words in academic writing
        'thus', 'hence', 'therefore', 'moreover', 'furthermore', 'additionally',
        'despite', 'although', 'however', 'nevertheless', 'consequently',
        'accordingly', 'indeed', 'notably', 'specifically', 'generally',
        # Symbols and punctuation as words
        'quot', 'amp', 'lt', 'gt', 'lpar', 'rpar', 'lsqb', 'rsqb', 'lcub', 'rcub'
    })
    
    # Tokenize and remove stopwords 
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in custom_stopwords]
    
    # Dictionary to store results
    ngram_results = {}
    
    for n in n_values:
        # For unigrams, we already have filtered tokens
        if n == 1:
            # Count occurrences
            ngram_counter = Counter(filtered_tokens)
            
            # Filter out short words (less than 3 characters) - keep as Counter object
            filtered_counter = Counter()
            for word, count in ngram_counter.items():
                if len(word) > 2 and word.isalpha():
                    filtered_counter[word] = count
            
            # Extract top n_grams - now using the Counter object
            top_ngrams = filtered_counter.most_common(top_n * 3)  # Get more, then filter
            
            # Store as strings, only keeping valid ones
            valid_ngrams = [gram for gram, count in top_ngrams 
                          if is_valid_ngram(gram, custom_stopwords)]
            ngram_results[n] = valid_ngrams[:top_n]
        else:
            # For higher n-grams, generate from filtered tokens
            n_grams_list = list(ngrams(filtered_tokens, n))
            
            # Convert tuples to strings for readability
            n_gram_strings = [' '.join(gram) for gram in n_grams_list]
            
            # Filter out invalid n-grams
            filtered_ngrams = [gram for gram in n_gram_strings 
                             if is_valid_ngram(gram, custom_stopwords)]
            
            # Count occurrences
            ngram_counter = Counter(filtered_ngrams)
            
            # Extract top n_grams
            top_ngrams = ngram_counter.most_common(top_n * 3)  # Get more, then further filter
            
            # Additional filtering for relevance
            valid_ngrams = [gram for gram, count in top_ngrams 
                          if count > 1 and is_valid_ngram(gram, custom_stopwords)]
            
            # Store top valid n-grams, up to top_n
            ngram_results[n] = valid_ngrams[:top_n]
    
    return ngram_results

def extract_keywords_tfidf(text, ngram_range=(1, 1), top_n=10, min_df=2):
    """
    Uses TF-IDF to extract the top_n keywords from the text with better filtering.
    
    Args:
        text (str): The text to extract keywords from
        ngram_range (tuple): Range of n-gram values (min_n, max_n)
        top_n (int): Number of top keywords to extract
        min_df (int): Minimum document frequency for terms to be included
        
    Returns:
        list: List of top keywords
    """
    # Create custom stopword list
    from nltk.corpus import stopwords
    custom_stop_words = set(stopwords.words('english')).union({
        # Add the same extensive custom stopwords as in extract_ngrams function
        'paper', 'figure', 'table', 'et', 'al', 'arxiv', 'url', 'https', 'org', 'abs',
        'section', 'review', 'using', 'used', 'use', 'method', 'work', 'approach',
        'proposed', 'propose', 'results', 'result', 'show', 'shows', 'shown',
        # URLs and identifiers
        'http', 'https', 'www', 'com', 'org', 'net', 'doi', 'arxiv', 'fig', 'eq',
        # Citation/reference terms
        'et', 'al', 'pp', 'vol', 'volume', 'journal', 'conference', 'proceedings', 
        'ieee', 'acm', 'springer', 'elsevier', 'publication', 'publications',
        # Numbers/dates/common measurements
        'one', 'two', 'three', 'first', 'second', 'third', 'fourth', 'fifth',
        'i', 'ii', 'iii', 'iv', 'v', 'january', 'february', 'march', 'april', 'may',
        'june', 'july', 'august', 'september', 'october', 'november', 'december'
    })
    
    # Initialize the TF-IDF Vectorizer with custom stop words and specified n-gram range
    vectorizer = TfidfVectorizer(
        stop_words=list(custom_stop_words),
        ngram_range=ngram_range,
        min_df=min_df,             # Minimum document frequency
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]+[a-zA-Z]\b'  # Require alpha terms at least 3 chars
    )
    
    # Split text into paragraphs to create a meaningful corpus for TF-IDF
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p for p in paragraphs if len(p.strip()) > 100]  # Filter short paragraphs
    
    if len(paragraphs) < 3:  # If not enough paragraphs, create artificial ones
        paragraphs = [text[i:i+1000] for i in range(0, len(text), 1000) if i+1000 <= len(text)]
    
    if not paragraphs:  # Safety check
        paragraphs = [text]
    
    # Fit and transform the paragraphs
    try:
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        
        # Get the feature names and their average TF-IDF scores across documents
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().mean(axis=0)  # Average across paragraphs
        
        # Combine each word/n-gram with its score and sort in descending order
        word_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter for valid words and remove duplicates or contained terms
        filtered_sorted_words = []
        seen_words = set()
        
        for word, score in sorted_words:
            # Skip if invalid or too short
            if not is_valid_ngram(word, custom_stop_words) or len(word) < 3:
                continue
            
            # Skip if this is a substring of an already selected term
            if any(word in seen_term for seen_term in seen_words):
                continue
                
            # Skip if this contains an already selected term (for unigrams in bigrams)
            if any(seen_term in word for seen_term in seen_words):
                continue
                
            filtered_sorted_words.append((word, score))
            seen_words.add(word)
            
            if len(filtered_sorted_words) >= top_n:
                break
        
        # Return the top_n valid words/n-grams as keywords
        top_keywords = [word for word, score in filtered_sorted_words[:top_n]]
        return top_keywords
    
    except Exception as e:
        print(f"Error in TF-IDF extraction: {e}")
        return []

def extract_section_titles(text):
    """
    Attempts to extract section titles from the academic paper.
    These can be good indicators of important topics.
    """
    # Common patterns for section titles in academic papers
    patterns = [
        r'\n\s*\d+\.\s+([A-Z][A-Za-z\s]{3,40})\s*\n',  # Numbered sections: "1. Introduction"
        r'\n\s*([A-Z][A-Z\s]{3,40})\s*\n',             # ALL CAPS SECTIONS
        r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s*\n'  # Title Case Sections
    ]
    
    section_titles = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        section_titles.extend(matches)
    
    # Filter and clean section titles
    cleaned_titles = []
    for title in section_titles:
        title = title.strip()
        # Skip common non-content sections
        if title.lower() in ['abstract', 'introduction', 'conclusion', 'references', 
                           'acknowledgments', 'appendix', 'related work']:
            continue
        if len(title) > 3 and title not in cleaned_titles:
            cleaned_titles.append(title)
    
    return cleaned_titles[:10]  # Return at most 10 section titles

def build_prompt(keywords, ngrams_dict=None, section_titles=None):
    """
    Constructs a prompt using the extracted keywords, n-grams, and section titles.
    """
    prompt = f"Based on the following key terms: {', '.join(keywords)}"
    
    if section_titles and len(section_titles) > 0:
        prompt += f"\n\nPaper sections: {', '.join(section_titles)}"
    
    if ngrams_dict:
        prompt += "\n\nRelevant phrases:"
        for n, grams in ngrams_dict.items():
            if n > 1 and grams:  # Only include phrases (bigrams and up) if we have any
                gram_type = "bigrams" if n == 2 else "trigrams" if n == 3 else f"{n}-grams"
                prompt += f"\n- {gram_type.capitalize()}: {', '.join(grams[:5])}"
    
    prompt += "\n\nPlease provide a comprehensive summary of the document."
    return prompt

def main():
    # Set the path to your PDF document
    pdf_path = "/Users/bhuvanpurohit777/Downloads/2501.18512v1.pdf"  # Replace with your actual PDF file path
    
    try:
        # Step 1: Extract text from the PDF
        print("Extracting text from PDF...")
        raw_text = extract_text_from_pdf(pdf_path)
        
        # Step 2: Clean the text
        print("Cleaning extracted text...")
        clean_doc_text = clean_text(raw_text)
        
        # Step 3: Extract section titles
        print("Extracting section titles...")
        section_titles = extract_section_titles(clean_doc_text)
        
        # Step 4: Extract n-grams using frequency-based approach
        print("Extracting n-grams...")
        ngram_results = extract_ngrams(clean_doc_text, n_values=[1, 2, 3], top_n=3)
        
        # Step 5: Extract keywords using TF-IDF for different n-gram ranges
        print("Extracting TF-IDF keywords...")
        unigram_keywords = extract_keywords_tfidf(clean_doc_text, ngram_range=(1, 1), top_n=3, min_df=2)
        bigram_keywords = extract_keywords_tfidf(clean_doc_text, ngram_range=(2, 2), top_n=3, min_df=2)
        trigram_keywords = extract_keywords_tfidf(clean_doc_text, ngram_range=(3, 3), top_n=3, min_df=1)
        
        # Step 6: Build a prompt incorporating the keywords and n-grams
        print("Building prompt...")
        prompt = build_prompt(bigram_keywords, ngram_results, section_titles)
        
        # Output the results
        print("\nExtracted Section Titles:")
        print(section_titles)
        
        print("\nExtracted Unigrams (TF-IDF):")
        print(unigram_keywords)
        
        print("\nExtracted Bigrams (TF-IDF):")
        print(bigram_keywords)
        
        print("\nExtracted Trigrams (TF-IDF):")
        print(trigram_keywords)
        
        print("\nFrequency-based N-grams:")
        for n, grams in ngram_results.items():
            gram_type = "unigrams" if n == 1 else "bigrams" if n == 2 else "trigrams" if n == 3 else f"{n}-grams"
            print(f"{gram_type.capitalize()}: {grams}")
        
        print("\nGenerated Prompt:")
        print(prompt)
        
    except Exception as e:
        import traceback
        print(f"Error in processing: {e}")
        traceback.print_exc()
