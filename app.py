import streamlit as st
from streamlit_option_menu import option_menu

# Configuration de la page
st.set_page_config(
    page_title="Clustering App",
    layout="wide",
    page_icon="📊"
)

# Style CSS commun
COMMON_STYLE = """
    <style>
        .header { color: #2c3e50; margin-bottom: 1.5rem; }
        .feature-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upload-box { border: 2px dashed #ccc; border-radius: 5px; padding: 2rem; text-align: center; }
        .footer { text-align: center; margin-top: 3rem; color: #7f8c8d; }
        .plot-container { background-color: white; padding: 1rem; border-radius: 5px; }
    </style>
"""
st.markdown(COMMON_STYLE, unsafe_allow_html=True)

# Navigation
with st.sidebar:
    selected = option_menu(
        "Menu Principal",
        ["🏠 Accueil", "🔶 K-Means", "🌳 CAH"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav-link": {"font-size": "16px", "text-align": "left"}
        }
    )

if selected == "🏠 Accueil":
    st.markdown('<h1 class="header">👋 Bienvenue sur l\'application de Clustering</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔶 K-Means Manuel</h3>
            <p>Implémentation personnalisée avec choix de métrique de distance et visualisation de la convergence.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Visualisations</h3>
            <p>Graphiques interactifs des clusters et courbes d'inertie pour analyser vos résultats.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌳 Classification Hiérarchique</h3>
            <p>CAH avec différents critères de liaison et génération de dendrogrammes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Export des Résultats</h3>
            <p>Export au format PDF/ZIP avec rapports automatiques.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">© 2025 Reda Erraoui - FID1 | Application développée avec Streamlit</div>', unsafe_allow_html=True)

elif selected == "🔶 K-Means":
    st.switch_page("pages/1_KMeans.py")
elif selected == "🌳 CAH":
    st.switch_page("pages/2_CAH.py")
