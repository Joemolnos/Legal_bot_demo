import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # Fallback: nincs tomllib
    _toml = None

# .env fájl betöltése (lokális fejlesztéshez)
load_dotenv()

class Config:
    def __init__(self):
        # Lokális secrets előtöltése (ha létezik .streamlit/secrets.toml)
        self._secrets = {}
        self._load_local_secrets()
        # Könyvtárak létrehozása
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Szükséges könyvtárak létrehozása"""
        for directory in [self.DOCUMENTS_DIR, self.DATA_DIR, 
                         self.EMBEDDINGS_DIR, self.CHUNKS_DIR, self.METADATA_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def _load_local_secrets(self):
        """Projekt- és felhasználói szintű secrets.toml beolvasása anélkül, hogy st.secrets-t használnánk.
        Ez elkerüli a Streamlit automatikus hibaüzenetét, ha nincs secrets.toml.
        Elérési sorrend prioritás szerint: projekt/.streamlit > home/.streamlit
        """
        if _toml is None:
            return
        candidates = [
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path.home() / ".streamlit" / "secrets.toml",
        ]
        merged = {}
        for p in candidates:
            try:
                if p.exists():
                    with open(p, "rb") as f:
                        data = _toml.load(f) or {}
                    # Projekt szintű előnyt élvez, ezért először adjuk hozzá a projektet
                    merged = {**data, **merged} if p == candidates[0] else {**merged, **data}
            except Exception:
                # Ha bármi gond van a fájl beolvasásával, hagyjuk figyelmen kívül
                continue
        self._secrets = merged

    def _get_setting(self, key, default=None, value_type=str):
        """Beállítás lekérése környezeti változóból vagy Streamlit secrets-ből"""
        value = None
        # 1) Lokális secrets.toml (ha létezik és be tudtuk olvasni)
        if isinstance(self._secrets, dict) and self._secrets:
            if 'general' in self._secrets and isinstance(self._secrets['general'], dict) and key in self._secrets['general']:
                value = self._secrets['general'][key]
            elif key in self._secrets:
                value = self._secrets[key]

        # 2) Környezeti változó / .env
        if value is None:
            value = os.getenv(key, default)
        
        # Típuskonverzió
        if value_type == int and value is not None:
            return int(value)
        elif value_type == float and value is not None:
            return float(value)
        return value
    
    # API konfiguráció
    @property
    def GROQ_API_KEY(self):
        return self._get_setting("GROQ_API_KEY")
    
    # Embedding beállítások
    @property
    def EMBEDDING_MODEL(self):
        return self._get_setting("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Szövegfeldolgozás
    @property
    def CHUNK_SIZE(self):
        return self._get_setting("CHUNK_SIZE", 1000, int)
    
    @property
    def CHUNK_OVERLAP(self):
        return self._get_setting("CHUNK_OVERLAP", 200, int)
    
    # LLM beállítások
    @property
    def MAX_TOKENS(self):
        return self._get_setting("MAX_TOKENS", 2048, int)
    
    @property
    def TEMPERATURE(self):
        return self._get_setting("TEMPERATURE", 0.3, float)
    
    # LLM beállítások (modell és kontextus keret)
    @property
    def LLM_MODEL(self):
        return self._get_setting("LLM_MODEL", "openai/gpt-oss-120b")

    @property
    def CONTEXT_TOKEN_BUDGET(self):
        return self._get_setting("CONTEXT_TOKEN_BUDGET", 1800, int)

    @property
    def TOP_K(self):
        return self._get_setting("TOP_K", 6, int)

    @property
    def RETRIEVE_N(self):
        return self._get_setting("RETRIEVE_N", 40, int)

    @property
    def ENABLE_MULTIQUERY(self):
        # bool parse: accept "true"/"1"/True
        val = str(self._get_setting("ENABLE_MULTIQUERY", "false")).lower()
        return val in ("1", "true", "yes", "on")

    @property
    def ENABLE_DIVERSIFY(self):
        val = str(self._get_setting("ENABLE_DIVERSIFY", "true")).lower()
        return val in ("1", "true", "yes", "on")

    @property
    def DIVERSIFY_LAMBDA(self):
        return self._get_setting("DIVERSIFY_LAMBDA", 0.6, float)

    # Fájl útvonalak
    DOCUMENTS_DIR = "documents/uploaded"
    DATA_DIR = "data"
    EMBEDDINGS_DIR = "data/embeddings"
    CHUNKS_DIR = "data/chunks"
    METADATA_DIR = "data/metadata"
    
    # Támogatott fájlformátumok
    SUPPORTED_FORMATS = ['.pdf']
    
    # Nyelvek (információs jellegű – a jelenlegi pipeline főként HU-ra optimalizált)
    INPUT_LANGUAGE = "hu"  # Magyar
    OUTPUT_LANGUAGE = "hu"  # Magyar