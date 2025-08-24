# Magyar Jogi Dokumentum RAG Rendszer (Verzió 0.1)

Egy AI-alapú rendszer magyar jogi dokumentumok elemzésére magyar nyelvű válaszadással, automatikus inicializálással.

## 🚀 Gyors Indítás

1.  **Csomagold ki a `legal_rag_project.zip` fájlt.**

2.  **Helyezd el a dokumentumokat**: Másold a magyar nyelvű PDF dokumentumaidat a `legal-rag-demo/documents/uploaded/` mappába.

3.  **Virtuális környezet (ajánlott)**:
    ```bash
    cd legal-rag-demo
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # vagy
    venv\Scripts\activate     # Windows
    ```

4.  **Függőségek telepítése**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **API Kulcs beállítása**:
    *   Nevezd át a `.env.example` fájlt `.env`-re.
    *   Szerkeszd a `.env` fájlt és add meg a Groq API kulcsodat:
        ```env
        GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ```

6.  **Alkalmazás indítása**:
    ```bash
    streamlit run app.py
    ```
    Az első indításkor a rendszer automatikusan feldolgozza a `documents/uploaded` mappában lévő PDF-eket. Ez eltarthat egy ideig. A későbbi indítások már gyorsak lesznek.

## 🎯 Funkciók

- **Automatikus inicializálás**: A `documents/uploaded` mappában lévő PDF-ek feldolgozása az első indításkor.
- **Intelligens keresés**: SOTA Qwen/Qwen3-Embedding-0.6B multilingual embedding modellel történik a hasonlóságkeresés (100+ nyelv támogatás, state-of-the-art pontosság). EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B **Magyar válaszadás**: magyar dokumentumok elemzése magyar nyelvű, kontextusfüggő válaszokkal.
- **Modern UI**: Könnyen kezelhető, reszponzív Streamlit felület.
- **Gyorsaság**: Groq API integráció a gyors LLM válaszokért.

## 🔧 Használat

1.  **Indítás után**: Várd meg, amíg a rendszer befejezi a kezdeti feldolgozást.
2.  **Kérdésfeltevés**: Írd be a kérdésedet magyarul a szövegdobozba, majd kattints a "Kérdés Feltevése" gombra.
3.  **Eredmények**: Olvasd el a választ és tekintsd át a forrásokat, amelyek alapján a válasz született.
4.  **Bővítés**: Ha új dokumentumokat szeretnél hozzáadni, használd az oldalsávban található fájlfeltöltőt.
