


import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import joblib
import logging
from cachetools import LRUCache
import threading
import time
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollegeChatbot:
    def __init__(self, data_file="qa.json", index_dir="index_data", model_name="all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.response_cache = LRUCache(maxsize=200)
        self.is_first_interaction = True
        self.is_ready = False  # Track if chatbot is fully loaded
        
        logger.info("Starting chatbot initialization...")
        
        # Load DialoGPT for fallback formal responses
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.dialog_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        
        # Load or create index
        if not self._index_exists():
            logger.info("Index not found. Creating new index...")
            self._create_index(data_file)
        else:
            logger.info("Loading existing index...")
            self._load_index()
        
        logger.info("CollegeChatbot initialized successfully")
        self.is_ready = True

    def _index_exists(self):
        """Check if index files exist"""
        required_files = [
            os.path.join(self.index_dir, "embeddings.npy"),
            os.path.join(self.index_dir, "meta.pkl"),
            os.path.join(self.index_dir, "nn.joblib")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _create_index(self, data_file):
        """Create and save the search index"""
        os.makedirs(self.index_dir, exist_ok=True)

        # Load data
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file {data_file} not found. Creating empty dataset.")
            docs = []

        self.questions = [d["question"] for d in docs]
        self.answers = [d["answer"] for d in docs]

        # Build vocabulary for spell correction from both questions and answers
        self.vocab = set()
        for d in docs:
            for text in [d["question"], d["answer"]]:
                for word in text.lower().split():
                    cleaned = word.strip('.,?!:;')
                    if len(cleaned) > 2:
                        self.vocab.add(cleaned)
        
        logger.info(f"Loaded {len(docs)} Q&A items with {len(self.vocab)} unique words in vocab")

        # Load model and compute embeddings
        self.embed_model = SentenceTransformer(self.model_name)
        
        if len(self.questions) > 0:
            embeddings = self.embed_model.encode(self.questions, show_progress_bar=True, 
                                               convert_to_numpy=True, batch_size=64)
            # Normalize for cosine similarity
            self.embeddings = normalize(embeddings, norm="l2", axis=1)
        else:
            # Create empty embeddings for empty dataset
            self.embeddings = np.array([]).reshape(0, 384)  # Default dimension for all-MiniLM-L6-v2

        # Build and save NearestNeighbors index
        if len(self.embeddings) > 0:
            self.nn = NearestNeighbors(n_neighbors=min(5, len(self.questions)), metric="cosine", n_jobs=-1)
            self.nn.fit(self.embeddings)
        else:
            self.nn = None

        # Save index files
        np.save(os.path.join(self.index_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.index_dir, "meta.pkl"), "wb") as f:
            pickle.dump({"questions": self.questions, "answers": self.answers, "vocab": list(self.vocab)}, f)
        
        if self.nn:
            joblib.dump(self.nn, os.path.join(self.index_dir, "nn.joblib"))

        logger.info(f"Index built and saved to {self.index_dir}")

    def _load_index(self):
        """Load precomputed index"""
        self.embeddings = np.load(os.path.join(self.index_dir, "embeddings.npy"))
        with open(os.path.join(self.index_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.questions = meta["questions"]
        self.answers = meta["answers"]
        self.vocab = set(meta.get("vocab", []))
        
        if len(self.embeddings) > 0:
            self.nn = joblib.load(os.path.join(self.index_dir, "nn.joblib"))
        else:
            self.nn = None
            
        self.embed_model = SentenceTransformer(self.model_name)

    def correct_query(self, query):
        """Correct potential typos in the query using word-level similarity to vocab"""
        words = query.lower().split()
        corrected_words = []
        for word in words:
            cleaned = word.strip('.,?!:;')
            punctuation = word[len(cleaned):]
            if len(cleaned) > 2:
                matches = difflib.get_close_matches(cleaned, self.vocab, n=1, cutoff=0.6)
                if matches:
                    corrected_words.append(matches[0] + punctuation)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        corrected_query = ' '.join(corrected_words)
        if corrected_query != query.lower():
            logger.info(f"Corrected query: '{query}' -> '{corrected_query}'")
        return corrected_query

    def find_similar_questions(self, query, k=3):
        """Find similar questions using nearest neighbors"""
        if not self.nn or len(self.questions) == 0:
            return []
            
        # Encode and normalize query
        query_embedding = self.embed_model.encode([query])
        query_embedding = normalize(query_embedding, norm="l2", axis=1)
        
        # Find nearest neighbors
        k = min(k, len(self.questions))
        distances, indices = self.nn.kneighbors(query_embedding, n_neighbors=k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": 1 - dist  # Convert distance to similarity score
            })
        
        return results

    def generate_response(self, query):
        """Generate response based on similarity search"""
        if not self.is_ready:
            return {
                "response": "Chatbot is still initializing. Please try again in a moment.",
                "response_time": "0ms",
                "confidence": "0.00",
                "cache_hit": False
            }

        original_query = query
        query = self.correct_query(query)

        # Check cache first (using corrected query for better cache hits across similar misspellings)
        cache_key = query.lower().strip()
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response["cache_hit"] = True
            return cached_response

        # Find similar questions
        similar_results = self.find_similar_questions(query, k=3)
        
        if not similar_results:
            # Fallback to DialoGPT for formal response
            response = self._generate_fallback_response(query)
            confidence = "0.00"
            response_data = {
                "response": response,
                "response_time": "1.0ms",
                "confidence": confidence,
                "cache_hit": False
            }
            self.response_cache[cache_key] = response_data
            return response_data

        # Get best match
        best_match = similar_results[0]
        confidence = best_match["score"]

        # Apply confidence thresholds
        if confidence > 0.8:
            response = best_match["answer"]
        elif confidence > 0.6:
            response = f"{best_match['answer']}"
        elif confidence > 0.4:
            response = f"{best_match['answer']}"
        else:
            # Fallback to DialoGPT for formal response
            response = self._generate_fallback_response(query)

        response_data = {
            "response": response,
            "response_time": "2.0ms",
            "confidence": f"{confidence:.2f}",
            "cache_hit": False
        }

        # Cache the response
        self.response_cache[cache_key] = response_data
        return response_data

    def _generate_fallback_response(self, query):
        """Generate a formal fallback response using DialoGPT-small"""
        prompt = f"Respond formally to this college inquiry: {query}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output_ids = self.dialog_model.generate(
            input_ids,
            max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def get_statistics(self):
        """Get chatbot statistics"""
        return {
            "dataset_size": len(self.questions),
            "embedding_cache_size": len(self.response_cache),
            "vocab_size": len(self.vocab),
            "index_ready": self.is_ready,
            "has_data": len(self.questions) > 0
        }

# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize chatbot in a separate thread"""
    global chatbot
    try:
        logger.info("🔄 Starting chatbot initialization...")
        start_time = time.time()
        
        chatbot = CollegeChatbot()
        
        load_time = time.time() - start_time
        logger.info(f"✅ Chatbot initialized successfully in {load_time:.2f} seconds")
        logger.info(f"📊 Loaded {len(chatbot.questions)} Q&A pairs")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        chatbot = None

def get_chatbot():
    """Get chatbot instance - returns None if not ready"""
    return chatbot

def is_chatbot_ready():
    """Check if chatbot is ready"""
    return chatbot is not None and chatbot.is_ready

# Start initialization in background thread
init_thread = threading.Thread(target=initialize_chatbot, daemon=True)
init_thread.start()

# Wait a moment and check status
time.sleep(1)
if chatbot and chatbot.is_ready:
    logger.info("🚀 Chatbot is ready! Starting Flask server...")
else:
    logger.info("⏳ Chatbot is initializing in background. Flask server starting...")