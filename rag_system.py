import os
import numpy as np
from typing import List, Dict
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from groq_client import GroqClient
from config import Config

class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.groq_client = GroqClient()
        self.documents_loaded = False
        self.initialize_system()

    def initialize_system(self):
        if self.embedding_manager.load_index():
            self.documents_loaded = True
            print("✅ Meglévő index sikeresen betöltve.")
        else:
            print("ℹ️ Nem található mentett index. A 'documents/uploaded' mappa feldolgozása következik...")
            self.process_documents_from_folder()

    def process_documents_with_progress(self):
        import glob
        pdf_files = glob.glob(os.path.join(self.config.DOCUMENTS_DIR, "*.pdf"))
        total = len(pdf_files)
        if not pdf_files:
            yield {"current": 0, "total": 0, "filename": None, "error": "⚠️ Nem található PDF fájl a 'documents/uploaded' mappában."}
            return

        print(f"📄 {total} PDF fájl feldolgozása indul...")
        any_success = False
        for i, file_path in enumerate(pdf_files, 1):
            try:
                print(f"Feldolgozás alatt: {os.path.basename(file_path)}")
                chunks, metadata = self.document_processor.process_pdf(file_path)
                embeddings = self.embedding_manager.create_embeddings(chunks, metadata)
                self.embedding_manager.build_index(embeddings)
                self.document_processor.save_processed_data(os.path.basename(file_path), chunks, metadata)
                yield {"current": i, "total": total, "filename": os.path.basename(file_path), "error": None}
                any_success = True
            except Exception as e:
                print(f"❌ Hiba a(z) {os.path.basename(file_path)} feldolgozása során: {e}")
                yield {"current": i, "total": total, "filename": os.path.basename(file_path), "error": str(e)}
        if any_success:
            self.embedding_manager.save_index()
            self.documents_loaded = True
            print(f"✅ A mappa feldolgozása befejeződött. Az index elmentve.")
        else:
            print("❌ Nem sikerült egyetlen dokumentumot sem feldolgozni.")

    def process_documents_from_folder(self):
        # Backward compatibility, non-progressive
        for _ in self.process_documents_with_progress():
            pass

    def add_documents(self, uploaded_files) -> Dict:
        results = {"success": [], "errors": [], "total_chunks": 0}
        new_embeddings_added = False
        for uploaded_file in uploaded_files:
            try:
                file_path = self._save_uploaded_file(uploaded_file)
                chunks, metadata = self.document_processor.process_pdf(file_path)
                embeddings = self.embedding_manager.create_embeddings(chunks, metadata)
                if embeddings is not None:
                    self.embedding_manager.build_index(embeddings)
                    self.document_processor.save_processed_data(uploaded_file.name, chunks, metadata)
                    results["success"].append({"filename": uploaded_file.name})
                    new_embeddings_added = True
            except Exception as e:
                results["errors"].append({"filename": uploaded_file.name, "error": str(e)})
        
        if new_embeddings_added:
            self.embedding_manager.save_index()
            self.documents_loaded = True
        return results

    def query(self, question: str, top_k: int = None) -> Dict:
        if not self.documents_loaded:
            return {"answer": "❌ Nincsenek betöltött dokumentumok. Kérlek, helyezz PDF fájlokat a 'documents/uploaded' mappába, majd indítsd újra az alkalmazást!", "sources": []}
        try:
            # Beállítások
            k = top_k if top_k is not None else self.config.TOP_K
            retrieve_n = max(k, self.config.RETRIEVE_N)

            # Multi-query (HU + RO fordítás, ha engedélyezett)
            queries = [question]
            if self.config.ENABLE_MULTIQUERY:
                ro = self.groq_client.translate_to_ro(question)
                if ro:
                    queries.append(ro)

            # Keresés több lekérdezéssel és egyesítés
            candidates: Dict[int, Dict] = {}
            for q in queries:
                results = self.embedding_manager.search_similar(q, retrieve_n)
                for r in results:
                    cid = int(r.get("chunk_id", -1))
                    if cid not in candidates:
                        candidates[cid] = r
                    else:
                        # tartsuk meg a magasabb hasonlóságot
                        if r.get("similarity_score", 0) > candidates[cid].get("similarity_score", 0):
                            candidates[cid] = r

            all_results = list(candidates.values())
            if not all_results:
                return {"answer": "❌ Nem találtam releváns információt a kérdésedre a dokumentumokban.", "sources": []}
            
            # Diverzifikáció (MMR) vagy sima top-k
            if self.config.ENABLE_DIVERSIFY and len(all_results) > k:
                selected = self._mmr_select(all_results, k, lambda_param=self.config.DIVERSIFY_LAMBDA)
            else:
                selected = sorted(all_results, key=lambda x: x.get("similarity_score", 0), reverse=True)[:k]

            # Rangsor frissítése
            for i, s in enumerate(selected, 1):
                s["rank"] = i

            answer = self.groq_client.generate_response(question, selected)
            sources = self._format_sources(selected)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"answer": f"❌ Hiba történt a lekérdezés során: {str(e)}", "sources": []}
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        file_path = os.path.join(self.config.DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        formatted = []
        for chunk in chunks:
            ps = chunk.get("page_start")
            pe = chunk.get("page_end")
            pages = None
            if ps and pe and ps != pe:
                pages = f"{ps}–{pe}"
            elif ps:
                pages = f"{ps}"
            formatted.append({
                "document": chunk.get("document_name", "Ismeretlen"),
                "relevance": f"{chunk.get('similarity_score', 0) * 100:.1f}%",
                "preview": chunk.get("text", "")[:250] + "...",
                "pages": pages
            })
        return formatted

    def _mmr_select(self, results: List[Dict], k: int, lambda_param: float = 0.6) -> List[Dict]:
        """Egyszerű MMR kiválasztás a redundancia csökkentésére a legjobb k elemre.
        A páronkénti hasonlóságot a SentenceTransformer beágyazásokkal számoljuk.
        """
        k = max(1, min(k, len(results)))
        # Query-hasonlóság már adott: similarity_score
        sims_to_query = np.array([r.get("similarity_score", 0.0) for r in results], dtype=np.float32)

        # Chunk szövegek beágyazása páronkénti hasonlósághoz
        texts = [r.get("text", "") for r in results]
        if not texts or self.embedding_manager.model is None:
            return sorted(results, key=lambda x: x.get("similarity_score", 0), reverse=True)[:k]

        emb = self.embedding_manager.model.encode(texts, normalize_embeddings=True)
        emb = emb.astype("float32")
        # Koszinusz ~ IP normalizált vektoroknál
        pairwise = np.matmul(emb, emb.T)

        selected_idx = []
        candidate_idx = list(range(len(results)))
        # Kezdjünk a legmagasabb query-sim-ű elemmel
        first = int(np.argmax(sims_to_query))
        selected_idx.append(first)
        candidate_idx.remove(first)

        while len(selected_idx) < k and candidate_idx:
            mmr_scores = []
            for idx in candidate_idx:
                diversity_penalty = max(pairwise[idx, j] for j in selected_idx) if selected_idx else 0.0
                score = lambda_param * sims_to_query[idx] - (1 - lambda_param) * diversity_penalty
                mmr_scores.append((score, idx))
            mmr_scores.sort(reverse=True)
            best_idx = mmr_scores[0][1]
            selected_idx.append(best_idx)
            candidate_idx.remove(best_idx)

        return [results[i] for i in selected_idx]
    
    def get_stats(self) -> Dict:
        if not self.documents_loaded or not self.embedding_manager.chunk_metadata:
            return {"documents": 0, "chunks": 0, "status": "Nincsenek betöltött dokumentumok"}
        
        unique_docs = sorted(list({meta.get("document_name", "ismeretlen") for meta in self.embedding_manager.chunk_metadata}))
        return {
            "documents": len(unique_docs),
            "chunks": len(self.embedding_manager.chunk_metadata),
            "status": "Rendszer kész",
            "document_list": unique_docs
        }