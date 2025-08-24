import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from config import Config

# FAISS opcionális: ha nincs elérhető wheel (pl. Python 3.13), essünk vissza NumPy alapú keresésre
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False

class EmbeddingManager:
    def __init__(self):
        self.config = Config()
        self.model_name = str(self.config.EMBEDDING_MODEL).strip()
        self.model: Optional[SentenceTransformer] = None
        # FAISS index csak akkor, ha elérhető a könyvtár
        self._use_faiss: bool = bool(_HAS_FAISS)
        self.index: Optional[object] = None
        # NumPy alapú fallback mátrix (IP/koz-szim hasonlóság normalizált vektorokra)
        self.embeddings_matrix: Optional[np.ndarray] = None
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
            if self._use_faiss:
                dimension = embeddings.shape[1]
                if self.index is None:
                    self.index = faiss.IndexFlatIP(dimension)  # type: ignore
                # FAISS azonnal normalizált vektorokkal IP = cos sim
                self.index.add(embeddings)  # type: ignore
                total = int(self.index.ntotal)  # type: ignore
                print(f"✅ Index építése kész (FAISS). Összesen {total} embedding.")
            else:
                if self.embeddings_matrix is None:
                    self.embeddings_matrix = embeddings.astype("float32")
                else:
                    # vertikális összefűzés
                    self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings.astype("float32")])
                print(f"✅ Index építése kész (NumPy). Összesen {self.embeddings_matrix.shape[0]} embedding.")
        except Exception as e:
            raise Exception(f"Hiba az index építése során: {str(e)}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        if self.model is None or len(self.chunk_metadata) == 0:
            return []
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
            results: List[Dict] = []
            if self._use_faiss and self.index is not None:
                scores, indices = self.index.search(query_embedding, k)  # type: ignore
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx != -1 and idx < len(self.chunk_metadata):
                        result = self.chunk_metadata[idx].copy()
                        result["similarity_score"] = float(score)
                        result["rank"] = i + 1
                        results.append(result)
                return results
            # NumPy fallback
            if self.embeddings_matrix is None or self.embeddings_matrix.size == 0:
                return []
            # IP pontszám: mivel normalizált a kimenet, ez ~cosine sim
            scores_np = np.matmul(self.embeddings_matrix, query_embedding[0])
            if k <= 0:
                k = 5
            k = min(k, scores_np.shape[0])
            top_idx = np.argpartition(-scores_np, k - 1)[:k]
            # Rendezzük véglegesen
            top_idx = top_idx[np.argsort(-scores_np[top_idx])]
            for i, idx in enumerate(top_idx, 1):
                if 0 <= int(idx) < len(self.chunk_metadata):
                    result = self.chunk_metadata[int(idx)].copy()
                    result["similarity_score"] = float(scores_np[int(idx)])
                    result["rank"] = i
                    results.append(result)
            return results
        except Exception as e:
            print(f"Hiba a keresés során: {str(e)}")
            return []
    
    def save_index(self, filename: str = "legal_docs_index"):
        try:
            saved_any = False
            if self._use_faiss and self.index is not None:
                index_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}.index")
                faiss.write_index(self.index, index_path)  # type: ignore
                saved_any = True
            if (not self._use_faiss) and self.embeddings_matrix is not None:
                npy_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}.npy")
                np.save(npy_path, self.embeddings_matrix)
                saved_any = True

            if saved_any:
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
            npy_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}.npy")
            metadata_path = os.path.join(self.config.EMBEDDINGS_DIR, f"{filename}_metadata.json")

            if self._use_faiss and os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)  # type: ignore
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                print(f"✅ Index betöltve (FAISS): {int(self.index.ntotal)} embedding, {len(self.chunk_metadata)} metaadat")  # type: ignore
                return True
            if (not self._use_faiss) and os.path.exists(npy_path) and os.path.exists(metadata_path):
                self.embeddings_matrix = np.load(npy_path).astype("float32")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                print(f"✅ Index betöltve (NumPy): {self.embeddings_matrix.shape[0]} embedding, {len(self.chunk_metadata)} metaadat")
                return True
            return False
        except Exception as e:
            print(f"Hiba az index betöltése során: {str(e)}")
            return False