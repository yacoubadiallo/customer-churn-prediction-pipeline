import joblib
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
import os

# Il est important d'importer CustomOneHotEncoder même si non directement utilisé ici,
# car joblib a besoin de connaître la classe pour désérialiser le pipeline.
from encoder import CustomOneHotEncoder

def load_pipeline(filepath="pipeline_model.pkl"):
    """
    Charge le pipeline entraîné depuis le chemin spécifié.
    """
    try:
        pipeline = joblib.load(filepath)
        print(f"Pipeline chargé depuis '{filepath}'.")
        return pipeline
    except FileNotFoundError:
        print(f"Erreur : Le fichier du pipeline '{filepath}' n'a pas été trouvé. Assurez-vous que le chemin est correct.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du pipeline : {e}")
        return None

def load_test_data():
    """
    Charge les ensembles de test sauvegardés.
    """
    try:
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')
        print("X_test et y_test chargés.")
        return X_test, y_test
    except FileNotFoundError:
        print("Erreur : Les fichiers X_test.pkl ou y_test.pkl n'ont pas été trouvés.")
        print("Veuillez d'abord exécuter pipeline_training.py pour générer ces fichiers.")
        return None, None
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement des données de test : {e}")
        return None, None

def make_predictions(pipeline, X_data):
    """
    Réalise des prédictions à l'aide du pipeline.
    """
    print("\nRéalisation des prédictions sur les données de test...")
    predictions = pipeline.predict(X_data)
    print("Prédictions effectuées.")
    return predictions

def evaluate_model(y_true, y_pred):
    """
    Évalue le modèle à l'aide du Recall, Precision et F1-score.
    """
    print("\n--- Évaluation du Modèle ---")
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("--- Fin de l'Évaluation ---")

if __name__ == "__main__":
    model_filepath = "pipeline_model.pkl"

    loaded_pipeline = load_pipeline(model_filepath)
    if loaded_pipeline is None:
        exit()

    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        exit()

    y_pred = make_predictions(loaded_pipeline, X_test)

    evaluate_model(y_test, y_pred)

    print("\n--- Processus de prédiction et d'évaluation terminé ---")