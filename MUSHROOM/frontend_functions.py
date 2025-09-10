import streamlit as st
import requests

def background():
    # IMPORTANT : appelle st.set_page_config avant d'appeler background()
    st.markdown("""
    <style>
    /* ================== Fonts & palette ================== */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@400;600&display=swap');
    :root{
      --bg:#f3f2ec;        /* beige clair */
      --ink:#171513;       /* texte */
      --accent:#6a4c93;    /* violet */
      --accent-2:#ff7a59;  /* orange corail */
      --muted:#8b8a85;
    }

    /* ============ Débloque le fond (enlève couches blanches) ============ */
    [data-testid="stHeader"]{ background: transparent; }
    .block-container{ background: transparent; }
    .stApp { background: transparent; } /* on laisse le conteneur root transparent */

    /* ================== Fond à pois ================== */
    /* Appliqué sur la vue principale pour être visible partout */
    [data-testid="stAppViewContainer"]{
      background:
        radial-gradient(rgba(0,0,0,0.10) 1.2px, transparent 1.2px) 0 0/24px 24px,
        radial-gradient(rgba(0,0,0,0.07) 1.2px, transparent 1.2px) 12px 12px/24px 24px,
        var(--bg);
    }
    /* Sidebar en beige uni (optionnel) */
    [data-testid="stSidebar"]{ background: var(--bg); }

    /* ================== Typo & titres ================== */
    h1, h2, h3, .stMarkdown h1 {
      font-family: "Cormorant Garamond", Georgia, serif !important;
      letter-spacing:.5px; color:var(--ink);
    }
    .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stRadio, .stCheckbox, .stFileUploader, .stCameraInput {
      font-family:"Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }

    /* ================== Séparateur fin ================== */
    .hr { height:1px; background:#00000010; margin:12px 0 24px; }

    /* ================== Badge cartouche ================== */
    .badge{
      display:inline-block; padding:8px 14px; border:1px solid #00000030;
      border-radius:8px; font-size:14px; color:var(--ink); background:#ffffffa6; backdrop-filter: blur(2px);
    }

    /* ================== Cards ================== */
    .card{
      border:1px solid #00000015; border-radius:16px; padding:16px;
      background:#fff; box-shadow:0 4px 12px #0000000d;
    }

    /* ================== Boutons ================== */
    .stButton>button{
      border-radius:12px; padding:10px 16px; font-weight:600;
      border:1px solid #00000020;
    }
    .stButton>button[kind="primary"]{ background:var(--accent); color:white; }
    .stButton>button:hover{ transform:translateY(-1px); }

    /* ================== Pills ================== */
    .pill{ padding:6px 10px; border-radius:999px; font-weight:600; font-size:13px; }
    .pill.ok  { background:#e8fff1; color:#207a43; border:1px solid #b8e7c7; }
    .pill.no  { background:#fff0f0; color:#b00020; border:1px solid #f3c1c1; }
    .pill.mid { background:#fff9e8; color:#8a5a00; border:1px solid #f3ddae; }
    </style>
    """, unsafe_allow_html=True)
    return 1


def title():
    st.markdown("""
    <h1 style="text-align:center; font-size:58px; margin-bottom:4px;">
      🍄 What is this Mushroom?
    </h1>
    <p style="text-align:center; margin-top:-6px;">
      <span class="badge">Mushroom Species Predictor</span>
    </p>
    """, unsafe_allow_html=True)
    return 1


def url_exists(url: str, timeout: float = 4.0) -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/1.0)"}
    try:
        # HEAD d'abord (léger)
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        if r.status_code < 400:
            return True
        # certains sites bloquent HEAD -> on tente un petit GET
        if r.status_code in (400, 401, 403, 405):
            r = requests.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=headers)
            return r.status_code < 400
        return False
    except requests.RequestException:
        return False
