import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os

# Configuration pour éviter les erreurs SSL
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Sentiments IMDb",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Téléchargement des ressources NLTK
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return set(stopwords.words('english'))
    except Exception as e:
        st.error(f"Erreur NLTK: {e}")
        return set()


class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = None

    def load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            model_path = 'models/saved_models/imdb_model.h5'
            vectorizer_path = 'models/saved_models/tfidf_vectorizer.joblib'

            if not os.path.exists(model_path):
                st.error("❌ Modèle non trouvé. Veuillez d'abord exécuter model_training.py")
                return False

            self.model = load_model(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.stop_words = load_nltk_resources()
            return True

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des modèles : {e}")
            return False

    def preprocess_text(self, text):
        """Prétraitement du texte identique à l'entraînement"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        return ' '.join(tokens)

    def predict_sentiment(self, text):
        """Prédit le sentiment d'un texte"""
        try:
            # Prétraitement
            cleaned_text = self.preprocess_text(text)

            # Vectorisation
            text_vectorized = self.vectorizer.transform([cleaned_text])

            # Prédiction
            prediction = self.model.predict(text_vectorized.toarray(), verbose=0)[0][0]

            # Interprétation
            sentiment = "POSITIF" if prediction > 0.5 else "NÉGATIF"
            confidence = prediction if sentiment == "POSITIF" else 1 - prediction

            return sentiment, confidence, prediction, cleaned_text

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return "ERREUR", 0.0, 0.5, ""


def main():
    # Initialisation
    analyzer = SentimentAnalyzer()

    # Sidebar
    st.sidebar.title("🎬 IMDb Sentiment Analysis")
    st.sidebar.markdown("---")

    # Chargement des modèles
    with st.sidebar:
        st.info("Chargement des modèles...")
        if analyzer.load_models():
            st.success("✅ Modèles chargés avec succès!")
        else:
            st.error("❌ Erreur de chargement des modèles")
            st.info("💡 Astuce : Exécutez d'abord 'python model_training.py' pour entraîner le modèle")
            return

    # Page principale
    st.title("🎬 Analyse de Sentiments des Critiques IMDb")
    st.markdown("""
    Cette application utilise un réseau neuronal pour classer les critiques de films en **positives** ou **négatives**.
    Entrez votre critique ci-dessous et découvrez l'analyse de sentiment !
    """)

    # Layout en colonnes
    col1, col2 = st.columns([2, 1])

    with col1:
        # Zone de texte pour la critique
        st.subheader("📝 Entrez votre critique de film")

        # Exemples prédéfinis
        example_reviews = {
            "🌟 Critique positive": "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged from beginning to end. The cinematography was breathtaking and the character development was exceptional. Highly recommended!",
            "👎 Critique négative": "Terrible movie. Poor acting, boring plot, and awful direction. I want my money back. The dialogue was cringe-worthy and the special effects looked cheap. One of the worst films I've ever seen.",
            "😐 Critique mixte": "The cinematography was beautiful and the actors did a decent job, but the plot was confusing and the pacing was too slow. Some scenes were brilliant while others fell completely flat."
        }

        selected_example = st.selectbox(
            "Choisir un exemple :",
            list(example_reviews.keys()),
            index=0
        )

        user_review = st.text_area(
            "Votre critique :",
            value=example_reviews[selected_example],
            height=200,
            placeholder="Écrivez votre critique de film ici..."
        )

        # Bouton d'analyse
        if st.button("🎯 Analyser le sentiment", type="primary", use_container_width=True):
            if user_review.strip():
                with st.spinner("Analyse en cours..."):
                    sentiment, confidence, raw_score, cleaned_text = analyzer.predict_sentiment(user_review)

                    # Affichage des résultats
                    st.subheader("📊 Résultats de l'analyse")

                    # Métriques
                    col_metric1, col_metric2, col_metric3 = st.columns(3)

                    with col_metric1:
                        if sentiment == "POSITIF":
                            st.metric("Sentiment", "POSITIF 🎉", delta="Positive", delta_color="normal")
                        else:
                            st.metric("Sentiment", "NÉGATIF 😞", delta="Negative", delta_color="off")

                    with col_metric2:
                        st.metric("Confiance", f"{confidence:.2%}")

                    with col_metric3:
                        st.metric("Score brut", f"{raw_score:.4f}")

                    # Jauge de sentiment
                    st.subheader("📈 Score de sentiment")
                    fig = go.Figure()

                    fig.add_trace(go.Indicator(
                        mode="gauge+number+delta",
                        value=raw_score * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Score de sentiment"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightcoral"},
                                {'range': [40, 60], 'color': "lightyellow"},
                                {'range': [60, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Détails techniques
                    with st.expander("🔍 Détails techniques"):
                        st.write(f"**Texte prétraité :**")
                        st.code(cleaned_text)
                        st.write(f"**Score de prédiction :** {raw_score:.6f}")
                        st.write(f"**Seuil de classification :** 0.5")

            else:
                st.warning("⚠️ Veuillez entrer une critique à analyser.")

    with col2:
        st.subheader("ℹ️ Comment ça marche ?")
        st.markdown("""
        **Technologies utilisées :**

        🔤 **Prétraitement** : NLTK  
        📊 **Vectorisation** : TF-IDF  
        🧠 **Modèle** : Réseau neuronal  
        🎨 **Interface** : Streamlit

        **Processus :**
        1. Nettoyage du texte
        2. Suppression mots vides
        3. Vectorisation TF-IDF
        4. Prédiction réseau neuronal
        5. Analyse résultat
        """)

        st.markdown("---")
        st.subheader("📁 Structure du projet")
        st.code("""
        imdb_sentiment_analysis/
        ├── app.py
        ├── model_training.py
        ├── requirements.txt
        ├── models/
        │   └── saved_models/
        └── IMDB Dataset.csv
        """)


if __name__ == "__main__":
    main()