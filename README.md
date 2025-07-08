Projet de Prédiction de l'Attrition Client
Ce projet développe un pipeline de Machine Learning pour prédire le départ des clients (attrition) d'une banque. L'objectif est d'identifier les clients à risque pour permettre des actions de rétention proactives.

Structure du Projet
Pipeline/
├── encoder.py              # Encodeur One-Hot Custom
├── pipeline_training.py    # Script d'entraînement du modèle
├── pipeline_prediction.py  # Script de prédiction et d'évaluation
├── pipeline_model.pkl      # Modèle entraîné (généré)
├── X_test.pkl              # Données de test (généré)
├── y_test.pkl              # Cible de test (généré)
├── BankChurners.csv        # Données source
├── captured_training.png   # Preuve de l'entraînement
├── captured_prediction.png # Preuve de la prédiction et des résultats
└── README.md
Fonctionnalités Clés
Nettoyage et Prétraitement des Données : La variable cible Attrition_Flag est binarisée. Les caractéristiques sont explicitement sélectionnées à partir des données BankChurners.csv pour assurer la qualité des données et prévenir la fuite d'informations (data leakage).

Pipeline ML Complet : Utilisation de scikit-learn pour un pipeline intégrant un encodeur personnalisé (CustomOneHotEncoder), un StandardScaler et un RandomForestClassifier.

Séparation Entraînement/Test : Division des données pour une évaluation robuste.

Évaluation des Performances : Mesure du Rappel (Recall), de la Précision (Precision) et du Score F1 (F1-score) sur l'ensemble de test, reflétant la capacité réelle du modèle à prédire l'attrition.

Exécution du Projet
Prérequis
Python 3.8+

Bibliothèques : pandas, scikit-learn, joblib (installez-les via pip install pandas scikit-learn joblib)

Données : BankChurners.csv doit être dans le dossier Pipeline/.

Naviguer vers le dossier Pipeline/ dans votre terminal.

Entraîner le modèle :

python pipeline_training.py
![training](https://github.com/user-attachments/assets/16384c9b-d822-439f-b761-c5574729a7d4)


Prédire et Évaluer :

python pipeline_prediction.py
![prediction](https://github.com/user-attachments/assets/c1a41509-f815-4d04-8085-11ecee66642b)

Résultat et Leçons Apprises
Le pipeline produit des prédictions robustes. La résolution de la fuite de données a été un point d'apprentissage crucial, transformant des scores artificiels (1.00) en des métriques réalistes (par exemple, Recall: 0.76, Precision: 0.93, F1-score: 0.84). Ce projet démontre ma capacité à construire des solutions ML complètes et fiables.
