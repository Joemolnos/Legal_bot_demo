# Streamlit Cloud Telepítési Útmutató

Ez az útmutató segít a Román Jogi Dokumentum Elemző alkalmazás Streamlit Cloud-ra való telepítésében.

## 1. Előkészületek

### GitHub Repository létrehozása

1. Hozz létre egy privát GitHub repository-t
2. Töltsd fel a teljes projektet a repository-ba

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/felhasznaloneved/repo-nev.git
git push -u origin main
```

## 2. Streamlit Cloud beállítása

1. Jelentkezz be a [Streamlit Cloud](https://streamlit.io/cloud)-ba GitHub fiókkal
2. Kattints a "New app" gombra
3. Válaszd ki a repository-t, branch-et és add meg az app.py elérési útját
4. Állítsd be a Python verziót (3.9 vagy 3.10 ajánlott)
5. Kattints a "Deploy!" gombra

## 3. Környezeti változók beállítása

A Streamlit Cloud-on a környezeti változókat a következőképpen kell beállítani:

1. Nyisd meg az alkalmazás beállításait
2. Kattints a "Secrets" fülre
3. Add meg a következő titkos adatokat:

```toml
[general]
GROQ_API_KEY = "your_groq_api_key_here"
LLM_MODEL = "llama3-8b-8192"  # Groq modell neve
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2048
TEMPERATURE = 0.3
```

## 4. Dokumentumok kezelése

A Streamlit Cloud-on a feltöltött dokumentumok nem maradnak meg az alkalmazás újraindítása után. Két lehetőséged van:

### A. Dokumentumok tárolása a repository-ban

Ha a dokumentumok nem bizalmasak vagy már nyilvánosak, tárold őket a repository-ban a `documents/uploaded` mappában.

### B. S3 vagy más felhőtárhely használata

Bizalmas dokumentumok esetén használj S3-at vagy más felhőtárhelyet, és módosítsd a kódot ennek megfelelően.

## 5. Memóriahasználat optimalizálása

A Streamlit Cloud ingyenes verziója korlátozott memóriával rendelkezik. Optimalizálási tippek:

- Használj kisebb embedding modellt, ha szükséges
- Korlátozd a feldolgozott dokumentumok számát
- Állítsd be a `CHUNK_SIZE` értékét alacsonyabbra
- Használj kisebb LLM modellt, ha a válaszok minősége elfogadható marad

## 6. Hibaelhárítás

Ha az alkalmazás nem indul el vagy hibát jelez:

1. Ellenőrizd a Streamlit Cloud naplókat
2. Győződj meg róla, hogy minden szükséges függőség szerepel a requirements.txt fájlban
3. Ellenőrizd, hogy a GROQ_API_KEY helyesen van beállítva
4. Ellenőrizd, hogy a könyvtárstruktúra megfelelő-e

## 7. Tesztelés

Telepítés után teszteld az alkalmazást:
- Próbálj meg új dokumentumot feltölteni
- Tegyél fel kérdéseket magyarul
- Ellenőrizd, hogy a válaszok helyesek és magyarul érkeznek-e
