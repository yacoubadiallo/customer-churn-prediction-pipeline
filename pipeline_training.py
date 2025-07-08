# Pipeline/pipeline_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Importation de la classe CustomOneHotEncoder
from encoder import CustomOneHotEncoder

def load_data(filepath):
    """
    Charge le dataset depuis le chemin spécifié.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Chargement des données depuis '{filepath}' réussi. {len(df)} lignes, {len(df.columns)} colonnes.")
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filepath}' n'a pas été trouvé. Assurez-vous que le chemin est correct.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du fichier : {e}")
        return None

def preprocess_data(df):
    """
    Prétraite le DataFrame :
    - Sépare la cible des features.
    - Binarise la colonne cible.
    - Sélectionne explicitement les features à utiliser, en excluant les colonnes inutiles ou sources de fuite.
    """
    df_processed = df.copy()

    # CRITIQUE : Identifier la colonne cible en premier
    target_column_name = 'Attrition_Flag' # Selon votre description de fonctionnalités
    if target_column_name not in df_processed.columns:
        print(f"Erreur : La colonne cible '{target_column_name}' n'a pas été trouvée dans les données.")
        return None, None # Retourne None pour df_processed et y

    df_processed['target'] = df_processed[target_column_name].apply(lambda x: 1 if x == "Attrited Customer" else 0)
    print(f"Colonne cible '{target_column_name}' renommée en 'target' et binarisée.")

    features_to_keep = [
        'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
        'Marital_Status', 'Income_Category', 'Card_Category',
        'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]

    # Valider que toutes les features_to_keep sont réellement dans le DataFrame
    missing_features = [col for col in features_to_keep if col not in df_processed.columns]
    if missing_features:
        print(f"Avertissement : Les caractéristiques suivantes spécifiées pour le modèle sont manquantes dans le dataset : {missing_features}")
        # Optionnellement, supprimer les features manquantes de features_to_keep si vous voulez continuer
        features_to_keep = [col for col in features_to_keep if col in df_processed.columns]

    # Séparer les caractéristiques (X) et la cible (y)
    X = df_processed[features_to_keep].copy()
    y = df_processed['target']

    print(f"Caractéristiques sélectionnées pour le modèle ({len(X.columns)}): {X.columns.tolist()}")
    print(f"Taille finale de X après prétraitement: {X.shape}")
    print(f"Taille finale de y après prétraitement: {y.shape}")

    return X, y

def build_and_train_pipeline(X_train, y_train):
    """
    Construit et entraîne le pipeline de machine learning.
    """
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()
    print(f"\nCaractéristiques catégorielles identifiées pour l'encodage par CustomOneHotEncoder: {categorical_features}")

    pipeline = Pipeline([
        ('encoder', CustomOneHotEncoder(columns=categorical_features)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    print("\nDébut de l'entraînement du pipeline...")
    pipeline.fit(X_train, y_train)
    print("Entraînement du pipeline terminé.")

    return pipeline

def save_pipeline(pipeline, filename="pipeline_model.pkl"):
    """
    Sauvegarde le pipeline entraîné.
    """
    try:
        joblib.dump(pipeline, filename)
        print(f"\nPipeline sauvegardé sous '{filename}'.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du pipeline : {e}")

if __name__ == "__main__":
    data_filepath = "BankChurners.csv"
    output_model_path = "pipeline_model.pkl"

    df = load_data(data_filepath)
    if df is None:
        exit()

    # Maintenant preprocess_data retourne X et y directement
    X, y = preprocess_data(df)
    if X is None or y is None:
        exit()

    # Diviser les données en ensembles d'entraînement et de test
    print(f"\nDivision des données en ensembles d'entraînement et de test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} lignes")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]} lignes")

    # Construire et entraîner le pipeline
    trained_pipeline = build_and_train_pipeline(X_train, y_train)

    # Sauvegarder le pipeline
    save_pipeline(trained_pipeline, output_model_path)

    # Sauvegarder X_test et y_test pour le script de prédiction
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    print("X_test et y_test sauvegardés pour la prédiction.")

    print("\n--- Processus d'entraînement terminé ---")