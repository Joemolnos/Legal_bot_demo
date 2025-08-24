import os
import json
import fitz  # PyMuPDF
import hashlib
from typing import List, Dict, Tuple
from langdetect import detect
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.config = Config()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Szükséges könyvtárak létrehozása"""
        for directory in [
            self.config.DOCUMENTS_DIR,
            self.config.CHUNKS_DIR,
            self.config.METADATA_DIR
        ]:
            os.makedirs(directory, exist_ok=True)
    
    def process_pdf(self, file_path: str) -> Tuple[List[str], Dict]:
        """PDF fájl feldolgozása szöveg kinyeréssel és oldal-információ megtartásával.
        A chunkolás oldalszinten történik, hogy a hivatkozás (page) visszaadható legyen.
        """
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            all_chunks: List[str] = []
            chunk_pages: List[Dict] = []

            aggregated_text = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                aggregated_text.append(page_text)
                # Oldalszintű chunkolás
                page_chunks = self._create_chunks(page_text)
                for ch in page_chunks:
                    all_chunks.append(ch)
                    chunk_pages.append({"page_start": page_num + 1, "page_end": page_num + 1})

            doc.close()

            full_text = "".join(aggregated_text)

            # Metaadatok
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "total_pages": total_pages,
                "file_size": os.path.getsize(file_path),
                "file_hash": self._get_file_hash(file_path),
                "language": self._detect_language(full_text),
                # Chunk -> oldal megfeleltetés a további pipeline-hoz
                "chunk_pages": chunk_pages,
            }

            return all_chunks, metadata

        except Exception as e:
            raise Exception(f"Hiba a PDF feldolgozás során ({os.path.basename(file_path)}): {str(e)}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Szöveg darabolása átfedéssel, jog-specifikus szeparátorok preferálásával."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Jog-specifikus szeparátorok (nem regex, de jól működő heurisztikák)
        separators = [
            "\nArt.", "\nART.", "\nArt ", "\nART ",
            "\nCapitolul", "\nCAPITOLUL", "\nCap.", "\nCAP.",
            "\nSecțiunea", "\nSECȚIUNEA",
            "\nLegea", "\nLEGEA",
            "\n\n", "\n", " "
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=separators,
        )

        chunks = text_splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

    def _detect_language(self, text: str) -> str:
        """Nyelv felismerése"""
        try:
            if not text.strip(): return "unknown"
            sample_text = text[:1500]
            return detect(sample_text)
        except:
            return "unknown"
    
    def _get_file_hash(self, file_path: str) -> str:
        """Fájl hash számítása"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_processed_data(self, file_name: str, chunks: List[str], metadata: Dict):
        """Feldolgozott adatok mentése"""
        base_name = os.path.splitext(file_name)[0]
        
        chunks_file = os.path.join(self.config.CHUNKS_DIR, f"{base_name}_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        metadata_file = os.path.join(self.config.METADATA_DIR, f"{base_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)