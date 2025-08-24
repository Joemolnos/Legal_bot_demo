import streamlit as st
import os
import time
from rag_system import RAGSystem
from groq_client import GroqClient
from config import Config

# Oldal konfiguráció
st.set_page_config(
    page_title="Magyar Jogi dokumentum lekérdező rendszer – Demó",
    page_icon="⚖️",
    layout="wide"
)

# CSS stílus
st.markdown("""
<style>
/* Webfont és alap tipográfia */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif;
}

/* Színváltozók */
:root {
  --primary: #1f4e79;
  --primary-600: #173e60;
  --primary-700: #12344f;
  --muted: #667085;
  --stroke: #e5e7eb;
  --bg: #f7f9fc;
}

/* Háttér és konténer */
.stApp {
  background: radial-gradient(60% 60% at 0% 0%, rgba(31,78,121,.08), transparent 60%),
              radial-gradient(50% 50% at 100% 0%, rgba(15,23,42,.06), transparent 50%),
              linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
}
.block-container { padding-top: 1rem; }

/* Fejléc */
.main-header {
    font-size: 2.6rem;
    color: var(--primary);
    text-align: center;
    margin-bottom: 0.75rem;
    letter-spacing: .2px;
}

/* Hero szekció */
.hero {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 1.1rem 1.25rem;
  border-radius: 16px;
  color: #fff;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-600) 55%, #0f172a 100%);
  box-shadow: 0 10px 30px rgba(31,78,121,.25);
  border: 1px solid rgba(255,255,255,.2);
  margin-bottom: 1.1rem;
}
.hero-title { font-size: 1.4rem; font-weight: 700; }
.hero-subtitle { font-size: .95rem; opacity: .95; }
.hero-badges { display: flex; gap: .4rem; align-items: center; }
.pill { padding: .25rem .6rem; border-radius: 999px; font-size: .75rem; font-weight: 600; border: 1px solid rgba(255,255,255,.35); background: rgba(255,255,255,.15); }

/* Központi konténerek */
.card {
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* Spinner középre */
.stSpinner > div > div { text-align: center; }

/* Állapot dobozok */
.success-box { padding: 0.9rem 1rem; border-radius: 10px; background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #1b5e20; }
.error-box   { padding: 0.9rem 1rem; border-radius: 10px; background-color: #ffebee; border: 1px solid #ffcdd2; color: #b71c1c; }

/* Forrás doboz és badge-ek */
.source-box {
  padding: 0.9rem 1rem; border-radius: 12px; background: linear-gradient(180deg,#f6faff 0%,#eff6ff 100%);
  border: 1px solid #d6e7ff; margin: 0.5rem 0; box-shadow: 0 1px 2px rgba(16,24,40,.06);
}
.badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-right: .4rem; }
.badge-blue { background: #e6f0ff; color: var(--primary); }
.badge-gray { background: #eef1f5; color: #334155; }

/* Primary gomb stílus – gradiensekkel */
.stButton>button {
  border-radius: 12px; padding: 0.7rem 1rem; font-weight: 700;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-700) 100%);
  color: #fff; border: 0; box-shadow: 0 6px 16px rgba(31,78,121,.28);
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 20px rgba(31,78,121,.32); }
.stButton>button:active { transform: translateY(0); }

/* Textarea, input */
textarea, input, .stTextInput>div>div>input {
  border-radius: 12px !important; border: 1px solid #dbe2ea !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04) !important; background: #fff !important;
}

/* Tabs – Pill stílus */
.stTabs [role="tablist"] button[role="tab"] {
  border: 1px solid var(--stroke); border-radius: 999px; margin-right: .5rem; padding: .4rem .8rem;
  background: #fff; color: #0f172a; font-weight: 600;
}
.stTabs [role="tablist"] button[aria-selected="true"] {
  background: var(--primary); color: #fff; border-color: var(--primary-600);
}

/* Expander kinézet */
details {
  border: 1px solid var(--stroke); border-radius: 12px; background: #fff; box-shadow: 0 1px 2px rgba(16,24,40,.06);
}
details>summary { list-style: none; padding: .9rem 1rem; font-weight: 600; color: var(--primary); border-bottom: 1px solid transparent; }
details[open]>summary { border-bottom-color: var(--stroke); }

/* Sidebar háttér */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 60%, #f7f9fc 100%);
  border-right: 1px solid var(--stroke);
}

/* Segéd margók */
.mt-1 { margin-top: .5rem; }
.mb-1 { margin-bottom: .5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system(engine_key: str = "v1"):
    """A RAG rendszert inicializálja és cache-eli."""
    with st.spinner('🔄 Rendszer inicializálása... Ez az első indításkor több percig is eltarthat, ha dokumentumokat kell feldolgozni.'):
        rag_system = RAGSystem()
        # Groq API melegítése (cold start elkerülésére)
        try:
            _ = rag_system.groq_client.test_connection()
        except Exception:
            pass
    return rag_system

def initialize_app_state():
    """Az alkalmazás session state-jét inicializálja."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    # Fejléc
    st.markdown('<h1 class="main-header">⚖️ Magyar jogi dokumentum lekérdező – Demó</h1>', unsafe_allow_html=True)
    # Hero szekció
    st.markdown(
        '''
        <div class="hero">
          <div>
            <div class="hero-title">Gyors keresés, hivatkozott forrásokkal</div>
            <div class="hero-subtitle">RAG alapú válaszok, oldal- és relevancia-jelzőkkel</div>
          </div>
          <div class="hero-badges">
            <span class="pill">Demo</span>
            <span class="pill">RAG-alapú</span>
            <span class="pill">HU</span>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    # Jogi és adatvédelmi nyilatkozatok (összecsukható)
    with st.expander("⚠️ Jogi nyilatkozat (nem minősül jogi tanácsadásnak)"):
        st.markdown(
            """
            - Az alkalmazás kizárólag bemutató jellegű, tájékoztató eszköz. Nem minősül jogi tanácsadásnak vagy szakvéleménynek.
            - A használat nem hoz létre ügyvéd–ügyfél jogviszonyt.
            - A válaszok generatív nyelvi modell által készülnek, tartalmazhatnak pontatlanságokat, hiányosságokat vagy elavult információt.
            - A kapott eredményeket minden esetben ellenőrizd hivatalos és hiteles forrásokban (pl. Nemzeti Jogszabálytár).
            - A szolgáltató a használatból eredő esetleges károkért felelősséget nem vállal; a használat saját kockázatra történik.
            """
        )

    with st.expander("🔐 Adatvédelem (GDPR) – rövid tájékoztató"):
        st.markdown(
            """
            - Az alkalmazás nem gyűjt és nem tárol a felhasználóról személyes adatokat; nincs analitika vagy külön naplózás a bemenetekről a mi oldalunkon.
            - A beírt kérdés és a kiválasztott kontextus az LLM szolgáltató (Groq) felé továbbításra kerülhet az API-hívás részeként feldolgozás céljából.
            - Kérjük, NE adj meg személyes adatot, különleges adatot vagy üzleti titkot. Ha mégis megadsz, az adatkezelés jogalapja a te kifejezett hozzájárulásod.
            - Az LLM-szolgáltató önálló adatkezelő/feldolgozó lehet; adattovábbítás harmadik országba (pl. USA) előfordulhat az infrastruktúra miatt.
            - Mi nem tároljuk a kérdéseket; a szolgáltató esetleges naplózására és megőrzésére az ő feltételei érvényesek.
            - Érintetti jogaid (hozzáférés, helyesbítés, törlés, korlátozás, tiltakozás, adathordozhatóság) gyakorlásához fordulj az üzemeltetőhöz és/vagy a szolgáltatóhoz.
            """
        )

    with st.expander("📄 Felhasználási feltételek (röviden)"):
        st.markdown(
            """
            - A szolgáltatás "ahogy van" (AS IS) biztosított, jótállás és szavatosság nélkül.
            - Tilos a szolgáltatás jogellenes célú használata, személyes adatok bevitele, illetve a tartalmak jogsértő felhasználása.
            - Engedélyezett: kizárólag demo/bemutató célú, belső teszteléshez kötött használat.
            - Szerzői jog: a feltöltött/használt dokumentumok jogai a jogosultakat illetik meg; az itt megjelenített részletek idézésként/összefoglalóként szolgálnak.
            """
        )

    with st.expander("ℹ️ Pontosság és források"):
        st.markdown(
            """
            - A rendszer PDF-ekből készít kivonatokat és hivatkozásokat (oldalszámokkal), de ez nem minősül hiteles kiadványnak.
            - A jogszabályok gyakran változnak; mindig ellenőrizd a legfrissebb hivatalos szövegeket és publikációkat.
            """
        )
    
    # Alkalmazás inicializálása – cache kulcs a session_state-ből, hogy lehessen újraindítani
    engine_key = st.session_state.get("engine_key", "v1")
    rag_system = initialize_rag_system(engine_key)
    initialize_app_state()

    # Dokumentumfeldolgozás progresszív visszajelzéssel, ha nincs index
    if not rag_system.documents_loaded:
        if 'doc_progress_done' not in st.session_state:
            st.session_state.doc_progress_done = False
        if not st.session_state.doc_progress_done:
            st.warning('A dokumentumok feldolgozása folyamatban van. Ez eltarthat néhány percig...')
            progress_bar = st.progress(0, text="Feldolgozás indítása...")
            status_placeholder = st.empty()
            errors = []
            for progress in rag_system.process_documents_with_progress():
                if progress['total'] > 0:
                    percent = progress['current'] / progress['total']
                    progress_bar.progress(percent, text=f"{progress['current']}/{progress['total']} dokumentum feldolgozva")
                if progress['filename']:
                    msg = f"Feldolgozás alatt: {progress['filename']}"
                    if progress['error']:
                        msg += f" – Hiba: {progress['error']}"
                        errors.append(msg)
                    status_placeholder.info(msg)
                if progress['error'] and not progress['filename']:
                    status_placeholder.error(progress['error'])
            st.session_state.doc_progress_done = True
            progress_bar.empty()
            if errors:
                st.error("\n".join(errors))
            st.success("✅ A dokumentumok feldolgozása befejeződött. Az alkalmazás most már használható!")
            st.experimental_rerun()
        else:
            st.info('A dokumentumok feldolgozása sikeresen megtörtént. Frissítsd az oldalt, ha nem látnád az új funkciókat.')
            st.stop()

    # Sidebar - Dokumentum kezelés
    with st.sidebar:
        st.header("📄 Dokumentum Kezelés")
        
        # API kulcs ellenőrzése
        config = Config()
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            st.error("❌ Groq API kulcs nincs beállítva! Állítsd be a .env fájlban.")
            st.stop()
        
        # Megjelenés – színséma választó
        with st.expander("🎨 Megjelenés(inaktív funkció)", expanded=False):
            scheme = st.selectbox("Színséma", ["Kék", "Zöld", "Bíbor", "Narancs"], index=0)
            shades = {
                "Kék": ("#1f4e79", "#173e60", "#12344f"),
                "Zöld": ("#047857", "#065f46", "#064e3b"),
                "Bíbor": ("#6b21a8", "#581c87", "#4c1d95"),
                "Narancs": ("#b45309", "#92400e", "#7c2d12"),
            }
            p, p600, p700 = shades.get(scheme, shades["Kék"])
            st.markdown(
                f"""
                <style>
                :root {{ --primary: {p}; --primary-600: {p600}; --primary-700: {p700}; }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Groq kapcsolat tesztelése
        with st.expander("🔧 API és kapcsolat"):
            if st.button("Kapcsolat Tesztelése"):
                with st.spinner("Kapcsolódás..."):
                    groq_client = GroqClient()
                    if groq_client.test_connection():
                        st.success("✅ Groq API kapcsolat OK!")
                    else:
                        st.error("❌ Groq API kapcsolat hiba!")

        # Fájl feltöltés (további dokumentumokhoz)
        with st.expander("📤 Új PDF-ek Hozzáadása"):
            uploaded_files = st.file_uploader(
                "Válassz PDF fájlokat a bővítéshez",
                type=['pdf'],
                accept_multiple_files=True,
                help="Ezek a dokumentumok a már meglévőkhöz adódnak hozzá."
            )
            if uploaded_files and st.button("🚀 Új Dokumentumok Feldolgozása"):
                with st.spinner('📖 Dokumentumok feldolgozása...'):
                    results = rag_system.add_documents(uploaded_files)
                    if results["success"]:
                        st.markdown(f'<div class="success-box">✅ Sikeresen hozzáadva {len(results["success"]) } dokumentum</div>', unsafe_allow_html=True)
                    if results["errors"]:
                        for error in results["errors"]:
                            st.markdown(f'<div class="error-box">❌ {error["filename"]}: {error["error"]}</div>', unsafe_allow_html=True)
                    st.rerun()

        # Keresési beállítások
        with st.expander("🔎 Keresési Beállítások", expanded=False):
            cfg = Config()
            top_k = st.slider(
                "Top-K (visszaadott kontextus darabok száma)",
                min_value=1, max_value=20, value=cfg.TOP_K, step=1,
                help="A legrelevánsabb szövegrészletek száma, amelyet a modell kontextusként megkap."
            )

        # Rendszer statisztikák
        with st.expander("📊 Rendszer Állapot", expanded=False):
            stats = rag_system.get_stats()
            st.metric("Dokumentumok száma", stats.get("documents", 0))
            st.metric("Feldolgozott szövegrészletek", stats.get("chunks", 0))
            st.info(f"**Állapot:** {stats.get('status', 'Ismeretlen')}")
            if stats.get("document_list"):
                st.markdown("**Betöltött dokumentumok listája**")
                for doc in sorted(stats["document_list"]):
                    st.write(f"• {doc}")
            # RAG motor újraindítása (cache törlés)
            if st.button("♻️ RAG motor újraindítása"):
                st.session_state["engine_key"] = str(time.time())
                st.rerun()
    
    # Fő tartalom Tabs – Kérdezés és Előzmények
    tab_query, tab_history = st.tabs(["💬 Kérdezés", "💭 Előzmények"])

    with tab_query:
        st.subheader("Kérdezz a dokumentumokról!")
        # Kérdés beviteli mező
        question = st.text_area(
            "Tedd fel a kérdésedet magyarul:",
            height=100,
            placeholder="Például: Hogyan rendelkezik az alaptörvény a polgárok választási jogairól?"
        )
        st.info("Tipp a prompthoz: légy konkrét (jogszabály/cikk/oldal), adj kontextust, és kerüld a személyes adatokat.")

        ask_button = st.button("🔍 Kérdés feltevése", type="primary", use_container_width=True)

        # Kérdés feldolgozása és azonnali megjelenítése
        latest_response = None
        if ask_button and question.strip():
            if not rag_system.documents_loaded:
                st.error("Először helyezz PDF fájlokat a 'documents/uploaded' mappába, majd indítsd újra az alkalmazást oldalfrissítéssel!")
            else:
                with st.spinner('🤔 Gondolkodom és a választ fordítom...'):
                    latest_response = rag_system.query(question, top_k=top_k)
                    st.session_state.chat_history.insert(0, {"question": question, "response": latest_response})

        # Ha most nincs friss válasz, mutassuk a legutóbbit
        if latest_response is None and st.session_state.chat_history:
            latest_response = st.session_state.chat_history[0]["response"]

        if latest_response:
            st.markdown("---")
            st.markdown("#### Válasz")
            st.markdown(latest_response['answer'])
            # Letöltés gomb
            dl_text = latest_response['answer']
            if latest_response.get('sources'):
                dl_text += "\n\nForrások:\n" + "\n".join([f"- {s['document']} (oldal: {s.get('pages','-')}, relevancia: {s['relevance']})" for s in latest_response['sources']])
            st.download_button("⬇️ Válasz letöltése (TXT)", data=dl_text, file_name="valasz.txt")

            if latest_response.get('sources'):
                st.markdown("##### Források:")
                # Két hasábos megjelenítés nagy kijelzőn
                cols = st.columns(2)
                for j, source in enumerate(latest_response['sources']):
                    col = cols[j % 2]
                    page_str = source.get('pages') or "-"
                    with col:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<div class="mb-1"><span class="badge badge-blue">Forrás {j+1}</span>'
                            f'<span class="badge badge-gray">Oldal: {page_str}</span>'
                            f'<span class="badge badge-gray">Relevancia: {source["relevance"]}</span></div>'
                            f'<strong>{source["document"]}</strong><br>'
                            f'<i>"{source["preview"]}"</i>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    with tab_history:
        if st.session_state.chat_history:
            if st.button("🗑️ Előzmények törlése"):
                st.session_state.chat_history = []
                st.rerun()

            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"**Kérdés #{len(st.session_state.chat_history)-i}:** {chat['question'][:80]}...", expanded=(i==0)):
                    st.markdown("##### Válasz:")
                    st.markdown(chat['response']['answer'])
                    if chat['response'].get('sources'):
                        st.markdown("##### Források:")
                        for j, source in enumerate(chat['response']['sources'], 1):
                            page_str = source.get('pages') or "-"
                            st.markdown(
                                f'<div class="source-box">'
                                f'<div class="mb-1"><span class="badge badge-blue">Forrás {j}</span>'
                                f'<span class="badge badge-gray">Oldal: {page_str}</span>'
                                f'<span class="badge badge-gray">Relevancia: {source["relevance"]}</span></div>'
                                f'<strong>{source["document"]}</strong><br>'
                                f'<i>"{source["preview"]}"</i>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

if __name__ == "__main__":
    main()