from groq import Groq
from typing import List, Dict
from config import Config

class GroqClient:
    def __init__(self):
        self.config = Config()
        self.client = Groq(api_key=self.config.GROQ_API_KEY)
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        try:
            # Építsük fel és vágjuk a kontextust a token kerethez igazítva
            context = self._build_context(context_chunks, budget=self.config.CONTEXT_TOKEN_BUDGET)
            prompt = self._build_prompt(query, context)
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                model=self.config.LLM_MODEL,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Hiba történt a válasz generálása során: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        return """Te egy magyar jogi asszisztens vagy, aki magyar és román nyelvű jogi dokumentumrészleteket elemez, és mindig magyarul válaszol.
FONTOS SZABÁLYOK:
1. Mindig MAGYARUL válaszolj.
2. Kizárólag a megadott kontextusra támaszkodj. Ha a kontextus nem elegendő, jelezd, hogy nincs elég információ.
3. Idézz közvetlenül a kontextusból annak EREDETI nyelvén. Ha az idézet román nyelvű, mellékelj rövid magyar értelmezést/fordítást. Ha az idézet magyar, ne erőltesd román idézetet.
4. Hivatkozz a forrásokra (pl. dokumentumnév; ha lehet: cikk/oldal).
5. Legyél tömör, pontos és közérthető; a jogi szakkifejezéseket röviden magyarázd el.
Feladat: Elemezd a rendelkezésre álló szövegrészleteket, és adj pontos, forrásolt választ magyarul a felhasználó kérdésére a kontextus alapján."""
    
    def _build_context(self, context_chunks: List[Dict], budget: int) -> str:
        """Kontextus építése és vágása hozzávetőleges token kerethez.
        Egyszerű karakter/token ~4:1 aránnyal becsülünk a költséghatékonyságért.
        """
        if not context_chunks:
            return "Nincs releváns kontextus találva a dokumentumokban."

        approx_chars_budget = budget * 4  # durva becslés
        used_chars = 0
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            doc_name = chunk.get('document_name', 'Ismeretlen dokumentum')
            text = chunk.get('text', '')
            ps = chunk.get('page_start')
            pe = chunk.get('page_end')
            page_str = ""
            if ps and pe and ps != pe:
                page_str = f"; oldalak: {ps}–{pe}"
            elif ps:
                page_str = f"; oldal: {ps}"
            piece = f"[Forrás: {doc_name}{page_str}]\n{text}\n"
            if used_chars + len(piece) > approx_chars_budget:
                remaining = max(0, approx_chars_budget - used_chars)
                if remaining > 0:
                    context_parts.append(piece[:remaining])
                break
            context_parts.append(piece)
            used_chars += len(piece)
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
FELHASZNÁLÓ KÉRDÉSE:
"{query}"

RELEVÁNS DOKUMENTUMRÉSZLETEK (eredeti nyelven):
---
{context}
---

UTASÍTÁSOK:
- Mindig magyarul válaszolj.
- Az idézeteket a kontextus EREDETI nyelvén add meg. Ha román idézet szerepel, tegyél mellé rövid magyar értelmezést/fordítást. Ha magyar az idézet, ne erőltesd román idézetet.
- Ha a kontextus nem elegendő, jelezd egyértelműen.
"""

    def translate_to_ro(self, text: str) -> str:
        """Egyszerű HU→RO fordítás a Groq LLM-mel, csak a fordítást adja vissza."""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Egy fordító vagy. Fordítsd le a felhasználó magyar üzenetét román nyelvre. Csak a román fordítást add vissza."},
                    {"role": "user", "content": text},
                ],
                model=self.config.LLM_MODEL,
                max_tokens=512,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def test_connection(self) -> bool:
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model=self.config.LLM_MODEL,
                max_tokens=5,
                temperature=0.0,
            )
            return True
        except Exception as e:
            print(f"Groq API kapcsolat hiba: {str(e)}")
            return False