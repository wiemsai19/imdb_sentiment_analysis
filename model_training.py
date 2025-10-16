import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Configuration pour éviter les erreurs SSL
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Téléchargement des ressources NLTK
print("📥 Téléchargement des ressources NLTK...")
nltk.download('stopwords')
nltk.download('punkt')


class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        self.model = None
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Prétraitement du texte"""
        if not isinstance(text, str):
            return ""

        # Conversion en minuscules
        text = text.lower()

        # Suppression des balises HTML
        text = re.sub(r'<.*?>', '', text)

        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Suppression de la ponctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenisation et suppression des mots vides
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        return ' '.join(tokens)

    def load_and_preprocess_data(self, file_path):
        """Charge et prétraite les données"""
        print("📊 Chargement des données...")
        data = pd.read_csv(file_path)

        # Échantillonnage pour tests rapides (à enlever en production)
        data = data.sample(2000, random_state=42)
        print(f"✅ {len(data)} critiques chargées")

        # Prétraitement
        print("🔄 Prétraitement des textes...")
        data['cleaned_review'] = data['review'].apply(self.preprocess_text)
        data['sentiment_encoded'] = data['sentiment'].map({'positive': 1, 'negative': 0})

        print(f"📈 Distribution des sentiments :")
        print(data['sentiment'].value_counts())

        return data

    def train_model(self, data):
        """Entraîne le modèle"""
        print("🔤 Vectorisation TF-IDF...")
        X = self.vectorizer.fit_transform(data['cleaned_review'])
        y = data['sentiment_encoded'].values

        # Conversion en format dense pour TensorFlow
        X_dense = X.toarray()

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X_dense, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📚 Données d'entraînement : {X_train.shape}")
        print(f"📊 Données de test : {X_test.shape}")

        # Construction du modèle simplifié
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Compilation
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("🧠 Architecture du modèle :")
        self.model.summary()

        # Entraînement
        early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

        print("🎯 Début de l'entraînement...")
        history = self.model.fit(
            X_train, y_train,
            epochs=5,  # Réduit pour tests rapides
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Évaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Précision sur le test : {test_accuracy:.4f}")

        return history, test_accuracy

    def save_models(self, model_dir='models/saved_models'):
        """Sauvegarde le modèle et le vectorizer"""
        os.makedirs(model_dir, exist_ok=True)

        # Sauvegarde du modèle Keras
        self.model.save(f'{model_dir}/imdb_model.h5')

        # Sauvegarde du vectorizer
        joblib.dump(self.vectorizer, f'{model_dir}/tfidf_vectorizer.joblib')

        print(f"💾 Modèle sauvegardé : {model_dir}/imdb_model.h5")
        print(f"💾 Vectorizer sauvegardé : {model_dir}/tfidf_vectorizer.joblib")


def main():
    """Fonction principale pour l'entraînement"""
    trainer = ModelTrainer()

    try:
        # Chargement des données
        data = trainer.load_and_preprocess_data('IMDB Dataset.csv')

        # Entraînement du modèle
        history, accuracy = trainer.train_model(data)

        # Sauvegarde des modèles
        trainer.save_models()

        print(f"🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📊 Précision finale : {accuracy:.4f}")

    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement : {e}")
        print("💡 Vérifiez que le fichier 'IMDB Dataset.csv' est dans le bon dossier")


if __name__ == "__main__":
    main()