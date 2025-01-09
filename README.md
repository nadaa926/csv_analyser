CSV Analyzer
Une application web Django permettant de lire, analyser et visualiser des fichiers CSV. Idéal pour explorer des données de manière interactive grâce à des fonctionnalités comme la recherche, les statistiques descriptives et la visualisation graphique.
________________________________________
Prérequis
Assurez-vous d'avoir les éléments suivants installés avant de commencer :
•	Python 3.8 ou supérieur
•	Django 4.x
•	pip
________________________________________
Installation
Étapes
1.	Clonez le dépôt :
bash
Copier le code
git clone https://github.com/votre-utilisateur/csv-analyzer.git
2.	Accédez au répertoire du projet :
bash
Copier le code
cd csv-analyzer
3.	Installez les dépendances :
bash
Copier le code
pip install -r requirements.txt
4.	Appliquez les migrations pour configurer la base de données :
bash
Copier le code
python manage.py migrate
5.	Lancez le serveur de développement :
bash
Copier le code
python manage.py runserver
6.	Accédez à l'application via http://localhost:8000.
________________________________________
Utilisation
Fonctionnalités principales
•	Téléversement de fichier CSV : Téléchargez un fichier via l'interface de téléversement et accédez à une page de menu pour choisir les différentes fonctionnalités.
•	Consultation des données :
o	Affichez toutes les données du fichier CSV sur la page data.
o	Recherchez des lignes ou colonnes spécifiques à l'aide d'une chaîne de caractères ou d'un index via la page de recherche. Les résultats s'affichent en temps réel.
•	Statistiques descriptives : Accédez à des métriques clés comme la moyenne, la médiane, l'écart-type, etc., via la page statistics.
•	Visualisation interactive : Créez des graphiques interactifs pour explorer les données sur la page visualization.
Navigation entre les pages
•	Menu principal (menu.html) : Accès à toutes les fonctionnalités principales de l'application.
•	Données (data.html) : Affichage complet des données.
•	Recherche (Recherche.html) : Recherche par colonnes, lignes ou chaîne de caractères.
•	Statistiques (statistics.html) : Résultats des analyses statistiques.
•	Visualisation (visualization.html) : Graphiques interactifs basés sur les données.
________________________________________
Fonctionnalités principales des fichiers
views.py
•	Gère la logique métier, y compris :
o	Le téléversement de fichiers CSV.
o	L'analyse statistique.
o	La gestion et la création de visualisations.
models.py
•	Définit les modèles pour stocker des informations sur :
o	Les fichiers CSV téléversés.
o	Les résultats d'analyses (si nécessaire).
urls.py
•	Configure les routes pour accéder aux différentes pages de l'application.
Répertoire templates/csv_analyzer/
•	Contient les fichiers HTML pour le rendu de l'interface utilisateur :
o	menu.html : Menu principal.
o	data.html : Affichage des données CSV.
o	Recherche.html : Interface de recherche.
o	statistics.html : Résultats statistiques.
o	visualization.html : Visualisation graphique.
o	upload_form.html : Formulaire de téléversement.
o	upload_error.html : Message d'erreur pour les fichiers incorrects.
Répertoire migrations/
•	Contient les fichiers de migration générés automatiquement pour gérer la base de données.

