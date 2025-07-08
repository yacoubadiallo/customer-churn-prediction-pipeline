# Pipeline/encoder.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encodeur personnalisé pour les variables catégorielles.
    Effectue un One-Hot Encoding sur les colonnes spécifiées.
    """
    def __init__(self, columns=None):
        """
        Initialise l'encodeur.

        Args:
            columns (list, optional): Liste des noms de colonnes à encoder.
                                     Si None, toutes les colonnes de type 'object' seront encodées.
        """
        self.columns = columns
        self.categories_ = {} # Pour stocker les catégories uniques de chaque colonne

    def fit(self, X, y=None):
        """
        Apprend les catégories uniques pour chaque colonne spécifiée.

        Args:
            X (pd.DataFrame): Les données d'entraînement.
            y (pd.Series, optional): La variable cible (non utilisée dans fit pour un encodeur).

        Returns:
            self: L'instance de l'encodeur ajustée.
        """
        if self.columns is None:
            # Si aucune colonne n'est spécifiée, encoder toutes les colonnes de type 'object'
            self.columns = X.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if col in X.columns:
                self.categories_[col] = X[col].astype(str).unique()
            else:
                print(f"Avertissement : La colonne '{col}' spécifiée pour l'encodage n'existe pas dans les données.")
        return self

    def transform(self, X):
        """
        Transforme les données en appliquant le One-Hot Encoding.

        Args:
            X (pd.DataFrame): Les données à transformer.

        Returns:
            pd.DataFrame: Les données transformées avec One-Hot Encoding.
        """
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            if col in X_transformed.columns:
                for cat in categories:
                    # Créer une nouvelle colonne binaire pour chaque catégorie
                    X_transformed[f"{col}_{cat}"] = (X_transformed[col].astype(str) == cat).astype(int)
                X_transformed = X_transformed.drop(columns=[col])
            else:
                # Si la colonne n'est pas présente dans les données à transformer,
                # ajouter des colonnes de zéros pour les catégories apprises.
                for cat in categories:
                    X_transformed[f"{col}_{cat}"] = 0 # Colonne de zéros
        return X_transformed