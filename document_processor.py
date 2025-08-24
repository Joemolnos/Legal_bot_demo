import os
import json
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
        """PDF fájl feldolgozása PyPDF2-vel (pure-Python), oldalszintű chunkolással.
        Cél: elkerülni a PyMuPDF (fitz) natív fordítását a Cloud környezetben.
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            all_chunks: List[str] = []
            chunk_pages: List[Dict] = []

            aggregated_text = []
            for page_num in range(total_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                aggregated_text.append(page_text)
                # Oldalszintű chunkolás
                page_chunks = self._create_chunks(page_text)
                for ch in page_chunks:
                    all_chunks.append(ch)
                    chunk_pages.append({"page_start": page_num + 1, "page_end": page_num + 1})

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
        """Szöveg darabolása átfedéssel, jog-specifikus sor-kezdések preferálásával.
        LangChain helyett könnyű, beépített megoldás a gyorsabb és stabilabb deploy érdekében.
        """
        CHUNK = max(200, int(self.config.CHUNK_SIZE or 1000))
        OVERLAP = max(0, int(self.config.CHUNK_OVERLAP or 200))

        # Sor eleji jogi markerek – ha egy sor ezek valamelyikével kezdődik, jó töréspont lehet
        headings = (
            "Art.", "ART.", "Art ", "ART ",
            "Capitolul", "CAPITOLUL", "Cap.", "CAP.",
            "Secțiunea", "SECȚIUNEA",
            "Legea", "LEGEA"
        )

        lines = text.split("\n")
        chunks: List[str] = []
        buf: List[str] = []
        cur_len = 0

        def flush_buffer():
            nonlocal buf, cur_len
            if not buf:
                return
            chunk_text = "\n".join(buf).strip()
            if len(chunk_text) > 50:
                chunks.append(chunk_text)
            buf = []
            cur_len = 0

        for line in lines:
            line = line.rstrip()
            line_len = len(line) + 1  # +1 az esetleges újsorért

            # Ha a jelenlegi sor jogi fejezet/jelölő és már elég nagy a puffer, zárjuk le
            if buf and line.startswith(headings) and cur_len >= int(CHUNK * 0.7):
                flush_buffer()

            # Ha túlcsordulna, zárjuk le és kezdjük újra
            if cur_len + line_len > CHUNK and buf:
                prev = "\n".join(buf)
                flush_buffer()
                # Átfedés karakterszinten (egyszerű): vágjuk ki az előző chunk utolsó OVERLAP részét
                if OVERLAP > 0:
                    overlap_text = prev[-OVERLAP:]
                    if overlap_text.strip():
                        buf.append(overlap_text)
                        cur_len = len(overlap_text)

            buf.append(line)
            cur_len += line_len

        flush_buffer()
        return chunks

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