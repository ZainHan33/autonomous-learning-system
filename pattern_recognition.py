from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class PatternRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models with GPU optimization
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Enable half precision for RTX 3090
        self.model = self.model.half()  # Convert to FP16
        self.vectorizer = TfidfVectorizer()
        self.patterns = {}
        
        # Optimize CUDA settings for RTX 3090
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    @torch.no_grad()  # Disable gradient tracking for inference
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using GPU-accelerated BERT embeddings
        """
        try:
            # Tokenize with padding
            inputs1 = self.tokenizer(text1, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            inputs2 = self.tokenizer(text2, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            
            # Move inputs to GPU
            inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
            
            # Use torch.amp.autocast for mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Get embeddings
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                
                # Use CLS token embeddings
                emb1 = outputs1.last_hidden_state[:, 0, :]
                emb2 = outputs2.last_hidden_state[:, 0, :]
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(emb1, emb2)
            
            return float(similarity.cpu().item())
            
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return 0.0
    
    def batch_process_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Process multiple texts in batches for efficiency
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
            
    def find_patterns(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Find common patterns using GPU acceleration
        """
        try:
            # Get embeddings for all texts in batches
            with torch.no_grad():
                embeddings = self.batch_process_texts(texts)
            
            # Calculate similarity matrix on GPU
            similarity_matrix = torch.mm(embeddings, embeddings.t())
            
            # Normalize
            norm = embeddings.norm(dim=1).unsqueeze(0)
            similarity_matrix = similarity_matrix / (norm * norm.t())
            
            # Convert to CPU and detach
            similarity_matrix = similarity_matrix.detach().cpu().numpy()
            
            # Group similar texts
            patterns = {}
            threshold = 0.7
            
            for i in range(len(texts)):
                similar_indices = np.where(similarity_matrix[i] > threshold)[0]
                if len(similar_indices) > 1:
                    pattern_key = f"pattern_{i}"
                    patterns[pattern_key] = [texts[j] for j in similar_indices]
                    
            return patterns
            
        except Exception as e:
            print(f"Pattern finding error: {str(e)}")
            return {}
            
    def update_patterns(self, new_text: str):
        """
        Update patterns with new information
        """
        try:
            if not self.patterns:
                self.patterns = {f"pattern_0": [new_text]}
                return
                
            # Convert existing patterns to embeddings
            all_texts = []
            pattern_map = {}
            
            for key, texts in self.patterns.items():
                start_idx = len(all_texts)
                all_texts.extend(texts)
                pattern_map[key] = (start_idx, len(all_texts))
            
            # Add new text
            all_texts.append(new_text)
            
            # Get embeddings for all texts
            with torch.no_grad():
                embeddings = self.batch_process_texts(all_texts)
            
            # Find most similar pattern
            new_text_embedding = embeddings[-1].unsqueeze(0)
            similarities = F.cosine_similarity(new_text_embedding, embeddings[:-1])
            
            max_sim, max_idx = torch.max(similarities, dim=0)
            
            # Add to existing pattern or create new one
            if max_sim > 0.7:
                # Find which pattern the max_idx belongs to
                for key, (start, end) in pattern_map.items():
                    if start <= max_idx < end:
                        self.patterns[key].append(new_text)
                        return
            
            # Create new pattern if no similar ones found
            new_key = f"pattern_{len(self.patterns)}"
            self.patterns[new_key] = [new_text]
            
        except Exception as e:
            print(f"Pattern update error: {str(e)}")
