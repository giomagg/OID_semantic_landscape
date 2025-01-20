import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
from keybert import KeyBERT
import pandas as pd
from typing import List, Dict
import textwrap

class ClusterThemeExtractor:
    def __init__(self, max_text_length: int = 900000):
        """
        Initialize the theme extractor with safety limits.
        
        Parameters:
        max_text_length (int): Maximum length for spaCy processing
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = max_text_length
        self.kw_model = KeyBERT()
        
    def chunk_text(self, text: str, chunk_size: int = 800000) -> List[str]:
        """
        Split text into chunks that spaCy can handle safely.
        """
        return textwrap.wrap(text, chunk_size, break_long_words=False)
    
    def process_with_spacy(self, text: str, num_keywords: int) -> List[str]:
        """
        Safely process text with spaCy using chunking.
        """
        chunks = self.chunk_text(text)
        all_entities = []
        
        for chunk in chunks:
            try:
                doc = self.nlp(chunk)
                all_entities.extend([ent.text for ent in doc.ents])
            except Exception as e:
                print(f"Warning: Error processing chunk with spaCy: {e}")
                continue
                
        return [item[0] for item in Counter(all_entities).most_common(num_keywords)]
    
    def process_with_keybert(self, text: str, num_keywords: int, max_ngram: int) -> List[str]:
        """
        Process text with KeyBERT using chunking if necessary.
        """
        chunks = self.chunk_text(text)
        all_keywords = []
        
        for chunk in chunks:
            try:
                keywords = self.kw_model.extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1, max_ngram),
                    stop_words='english',
                    top_n=num_keywords
                )
                all_keywords.extend([k[0] for k in keywords])
            except Exception as e:
                print(f"Warning: Error processing chunk with KeyBERT: {e}")
                continue
                
        return [item[0] for item in Counter(all_keywords).most_common(num_keywords)]
    
    def extract_cluster_themes(self, texts_by_cluster: Dict[int, List[str]], 
                             num_keywords: int = 5, 
                             max_ngram: int = 2) -> Dict[int, Dict]:
        """
        Safely extract themes from clusters with progress tracking.
        """
        cluster_themes = {}
        total_clusters = len(texts_by_cluster)
        
        for idx, (cluster_id, texts) in enumerate(texts_by_cluster.items(), 1):
            print(f"Processing cluster {idx}/{total_clusters} (ID: {cluster_id})")
            
            try:
                # Combine texts with length check
                combined_text = ' '.join(texts)
                print(f"  Combined text length: {len(combined_text)} characters")
                
                # 1. TF-IDF Important Terms
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, max_ngram),
                    max_features=1000,
                    stop_words='english'
                )
                tfidf_matrix = vectorizer.fit_transform([combined_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                top_tfidf = [feature_names[i] for i in tfidf_scores.argsort()[-num_keywords:][::-1]]
                
                # 2. KeyBERT Keywords (with chunking)
                print("  Extracting KeyBERT keywords...")
                top_keybert = self.process_with_keybert(combined_text, num_keywords, max_ngram)
                
                # 3. Named Entity Recognition (with chunking)
                print("  Extracting named entities...")
                top_entities = self.process_with_spacy(combined_text, num_keywords)
                
                # Combine and score themes
                all_themes = pd.DataFrame({
                    'theme': top_tfidf + top_keybert + top_entities,
                    'method': ['tfidf']*num_keywords + 
                             ['keybert']*len(top_keybert) + 
                             ['ner']*len(top_entities)
                })
                
                # Score themes based on frequency
                theme_scores = all_themes['theme'].value_counts()
                final_themes = theme_scores.head(num_keywords).index.tolist()
                
                cluster_themes[cluster_id] = {
                    'main_themes': final_themes,
                    'tfidf_terms': top_tfidf,
                    'keybert_terms': top_keybert,
                    'entities': top_entities
                }
                
                print(f"  Successfully processed cluster {cluster_id}")
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
                cluster_themes[cluster_id] = {
                    'main_themes': ['ERROR'],
                    'tfidf_terms': [],
                    'keybert_terms': [],
                    'entities': []
                }
                continue
        
        return cluster_themes

def generate_cluster_labels(cluster_themes: Dict[int, Dict], max_terms: int = 3) -> Dict[int, str]:
    """Generate visualization labels from themes."""
    labels = {}
    
    for cluster_id, themes in cluster_themes.items():
        top_terms = themes['main_themes'][:max_terms]
        label = ' / '.join(top_terms)
        labels[cluster_id] = label
    
    return labels
