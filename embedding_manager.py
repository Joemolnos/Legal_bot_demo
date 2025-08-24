import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from config import Config

class EmbeddingManager:
    def __init__(self):
        self.config = Config()
        self.model_name = str(self.config.EMBEDDING_MODEL).strip()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: List[Dict] = []
        self._ensure_directories()
        self._load_model()
    
    def _ensure_directories(self):
        os.makedirs(self.config.EMBEDDINGS_DIR, exist_ok=True)
    
    def _load_model(self):
        try:
            print(f"--- EmbeddingManager: Kísérlet az embedding modell betöltésére: '{self.model_name}' ---")
            self.model = SentenceTransformer(self.model_name)
            print(f"--- EmbeddingManager: Embedding modell sikeresen betöltve: '{self.model_name}' ---")
        except Exception as e:
            detailed_error = str(e)
            print(f"--- EmbeddingManager: Hiba az embedding modell betöltése során. Használt modellnév: '{self.model_name}'. Részletes hiba: {detailed_error} ---")
            raise Exception(f"Hiba az embedding modell betöltése során: {detailed_error}")
    
    def create_embeddings(self, chunks: List[str], document_metadata: Dict) -> Optional[np.ndarray]:
        if not chunks or self.model is None:
            return None
        try:
            print(f"Embeddings létrehozása {len(chunks)} darab szövegrészletből...")
            embeddings = self.model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
            
            chunk_pages = document_metadata.get("chunk_pages", [])
            for i, chunk in enumerate(chunks):
                page_start = None
                page_end = None
                if i < len(chunk_pages):
                    page_start = chunk_pages[i].get("page_start")
                    page_end = chunk_pages[i].get("page_end")

                meta = {
                    "chunk_id": len(self.chunk_metadata),
                    "text": chunk,
                    "document_name": document_metadata["file_name"],
                    "document_hash": document_metadata["file_hash"],
                    "chunk_index": i
                }
                if page_start is not None:
                    meta["page_start"] = page_start
                if page_end is not None:
                    meta["page_end"] = page_end

                self.chunk_metadata.append(meta)
            return embeddings.astype('float32')
        except Exception as e:
            raise Exception(f"Hiba az embeddings létrehozása során: {str(e)}")
    
    def build_index(self, embeddings: Optional[np.ndarray]):
        if embeddings is None:
            return
        try:
            dimension = embeddings.shape[1]
            if self.index is None:
                self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            print(f"✅ Index építése kész. Összesen {self.index.ntotal} embedding.")
        except Exception as e:
            raise Exception(f"Hiba az index építése során: {str(e)}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None or self.model is None or len(self.chunk_metadata) == 0:
            return []
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result["similarity_score"] = float(score)
                    result["rank"] = i + 1
                    results.append(result)
            return results
        except Exception as e:
            print(f"Hiba a keresés során: {str(e)}")
            return []
    
    def save_index(self, filename: str = "legal_docs_index"):
        try:
            if self.index:
                index_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}.index")
                faiss.write_index(self.index, index_path)
                
                metadata_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
                
                print("✅ Index és metaadatok sikeresen mentve")
                return True
        except Exception as e:
            print(f"Hiba az index mentése során: {str(e)}")
        return False
    
    def load_index(self, filename: str = "legal_docs_index") -> bool:
        try:
            index_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}.index")
            metadata_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}_metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                
                print(f"✅ Index betöltve: {self.index.ntotal} embedding, {len(self.chunk_metadata)} metaadat")
                return True
            else:
                return False
        except Exception as e:
            print(f"Hiba az index betöltése során: {str(e)}")
            return False