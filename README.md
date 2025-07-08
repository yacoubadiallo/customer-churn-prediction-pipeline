Projet de Prédiction de l'Attrition Client
Ce projet développe un pipeline de Machine Learning complet pour prédire l'attrition des clients d'une banque. L'objectif est d'identifier les clients à risque de départ afin de mettre en place des stratégies de rétention proactives.

Fonctionnalités Clés
Prétraitement de Données Robuste : Gestion de la variable cible Attrition_Flag et sélection explicite des caractéristiques pertinentes du dataset BankChurners.csv pour prévenir la fuite d'informations (data leakage).

Pipeline ML Intégré : Utilisation de scikit-learn pour construire un pipeline incluant un encodeur personnalisé (CustomOneHotEncoder), un StandardScaler et un RandomForestClassifier.

Évaluation Rigoureuse : Le modèle est entraîné sur un ensemble de données et évalué sur un ensemble de test distinct pour garantir la fiabilité des prédictions.

Scores de Performance du Modèle
Après entraînement et évaluation sur l'ensemble de test, le modèle a obtenu les métriques de performance suivantes, démontrant sa capacité à identifier les clients à risque :

Recall (Rappel) : 0.7600

Precision (Précision) : 0.9321

F1-score : 0.8373

Ces scores sont le résultat d'un pipeline optimisé, où la prévention de la fuite de données a permis d'obtenir des performances réalistes et significatives pour le problème de l'attrition client.
