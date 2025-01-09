import csv
import logging
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponseBadRequest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px
import plotly.io as pio

logger = logging.getLogger(__name__)

def save_plot_to_base64(fig):
    """Sauvegarde un graphique Matplotlib en Base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return image_base64

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import csv
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

def convert_cell(cell):
    """Fonction de conversion d'une cellule (par exemple convertir en entier ou flottant)."""
    try:
        # Essayer de convertir en nombre entier ou flottant
        if cell.isdigit():
            return int(cell)
        return float(cell)
    except ValueError:
        return cell  # Si non convertible, renvoyer la cellule telle quelle

def analyze_data(data, headers, errors):
    """Fonction d'analyse des données."""
    # Cette fonction devrait effectuer l'analyse des données et renvoyer un résultat
    # Vous pouvez ajouter ici votre logique d'analyse statistique
    return {"result": "statistical analysis placeholder"}  # Placeholder pour l'exemple

def upload_csv(request):
    """Gestion de l'upload d'un fichier CSV."""
    errors = []
    headers = []
    data = []
    columns = []
    selected_column = None
    stats_result = None
    error = None

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        filepath = fs.path(filename)

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as file:  # Utilisation de utf-8-sig pour gérer le BOM
                reader = csv.reader(file)
                
                # Lire les entêtes (première ligne du fichier CSV)
                try:
                    headers = next(reader)
                    columns = headers  # Utilisation des headers comme colonnes
                except StopIteration:
                    errors.append("Le fichier CSV est vide.")
                    return render(request, 'csv_analyzer/upload_form.html', {'errors': errors})

                # Lire les données (lignes suivantes)
                for row_num, row in enumerate(reader, start=2):  # Début à la ligne 2 (1 pour les entêtes)
                    data_row = []
                    for cell_num, cell in enumerate(row):
                        try:
                            converted_cell = convert_cell(cell)
                            data_row.append(converted_cell)
                        except ValueError:
                            errors.append(f"Erreur à la ligne {row_num}, colonne {cell_num + 1}: '{cell}'.")
                            logger.error(f"Erreur de conversion dans le fichier {filename}, ligne {row_num}, cellule {cell_num + 1}.")
                    data.append(data_row)
        except Exception as e:
            errors.append(f"Erreur inattendue: {str(e)}")
            logger.exception(f"Erreur inattendue lors du traitement du fichier {filename}: {e}")
        finally:
            fs.delete(filename)

        # Si des erreurs ont été collectées, les afficher
        if errors:
            return render(request, 'csv_analyzer/upload_form.html', {'errors': errors})

        # Sauvegarder les données dans la session pour réutilisation
        request.session['uploaded_data'] = {'data': data, 'headers': headers}

        # Effectuer l'analyse des données
        stats_result = analyze_data(data, headers, errors)

        # Vérifier si une colonne a été sélectionnée via le GET
        if request.GET.get("column"):
            selected_column = request.GET["column"]

        # Retourner les résultats au template
        return render(request, "csv_analyzer/upload_form.html", {
            "columns": columns,
            "selected_column": selected_column,
            "stats_result": stats_result,
            "error": error,
        })


    def convert_cell(cell):
        """Convertit une cellule CSV en nombre ou chaîne."""
    try:
        return float(cell.replace(",", "."))  # Conversion stricte pour les nombres
    except ValueError:
        return cell.strip()  # Retourne une chaîne si ce n'est pas un nombre

import pandas as pd

def analyze_data(data, headers, errors):
    """Analyse améliorée des données CSV avec distinction stricte entre numérique et catégorique."""
    analysis = {}

    if not data:
        errors.append("Le fichier CSV est vide.")
        return analysis

    df = pd.DataFrame(data, columns=headers)

    for header in headers:
        column_data = df[header]

        if pd.api.types.is_numeric_dtype(column_data):
            # Analyse pour les colonnes numériques
            numeric_data = column_data.dropna().astype(float)
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)].tolist()

            analysis[header] = {
                "type": "numeric",
                "mean": numeric_data.mean(),
                "median": numeric_data.median(),
                "min": numeric_data.min(),
                "max": numeric_data.max(),
                "stddev": numeric_data.std(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outliers": outliers,
            }

        else:
            # Analyse pour les colonnes catégoriques
            categorical_data = column_data.dropna()
            value_counts = categorical_data.value_counts()
            mode = value_counts.idxmax() if not value_counts.empty else None

            analysis[header] = {
                "type": "categorical",
                "unique_count": value_counts.count(),
                "mode": mode,
                "mode_count": value_counts.max() if mode else 0,
            }

    return analysis

# Variable globale pour stocker les données entre les requêtes
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from django.shortcuts import render

import plotly.express as px
import pandas as pd
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder  # Correction ici
import json

import plotly.express as px
import pandas as pd
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
import json

import plotly.express as px
import pandas as pd
import json
from plotly.utils import PlotlyJSONEncoder

def visualization(request):
    # Récupérer les données téléchargées de la session
    uploaded_data = request.session.get('uploaded_data', None)
    selected_column = request.POST.get('column', None)  # La colonne choisie par l'utilisateur
    plot_type = request.POST.get('plot_type', None)  # Le type de graphique choisi par l'utilisateur
    error = None
    graph = None
    columns_with_types = []
    columns = []

    if uploaded_data is not None:
        try:
            # Convertir les données en DataFrame si elles sont fournies sous forme de dictionnaire
            if isinstance(uploaded_data, dict):
                uploaded_data = pd.DataFrame(uploaded_data['data'], columns=uploaded_data['headers'])

            # Si uploaded_data est un DataFrame valide
            if isinstance(uploaded_data, pd.DataFrame) and not uploaded_data.empty:
                columns = uploaded_data.columns.tolist()

                # Générer la liste des colonnes et leurs types
                columns_with_types = [
                    {'name': col, 'type': 'numeric' if pd.api.types.is_numeric_dtype(uploaded_data[col]) else 'categorical'}
                    for col in uploaded_data.columns
                ]

                # Vérifier si une colonne et un type de graphique sont sélectionnés
                if selected_column and selected_column in columns and plot_type:
                    column_data = uploaded_data[selected_column]
                    column_type = 'numeric' if pd.api.types.is_numeric_dtype(column_data) else 'categorical'

                    # Vérifier si le type de graphique est compatible avec le type de colonne
                    if plot_type == 'histplot' and column_type == 'numeric':
                        fig = px.histogram(column_data.dropna(), nbins=30, title=f"Histogramme de {selected_column}")
                    elif plot_type == 'boxplot' and column_type == 'numeric':
                        fig = px.box(uploaded_data, y=selected_column, title=f"Boxplot de {selected_column}")
                    elif plot_type == 'kdeplot' and column_type == 'numeric':
                        fig = px.density_contour(column_data.dropna(), title=f"KDE Plot de {selected_column}")
                    elif plot_type == 'scatterplot' and column_type == 'numeric' and len(column_data) > 1:
                        fig = px.scatter(uploaded_data, x=uploaded_data.index, y=selected_column, title=f"Scatter Plot de {selected_column}")
                    elif plot_type == 'barplot' and column_type == 'categorical':
                        fig = px.bar(uploaded_data, x=selected_column, title=f"Bar Plot de {selected_column}")
                    elif plot_type == 'countplot' and column_type == 'categorical':
                        fig = px.histogram(uploaded_data, x=selected_column, title=f"Count Plot de {selected_column}")
                    elif plot_type == 'heatmap' and all(pd.api.types.is_numeric_dtype(uploaded_data[col]) for col in uploaded_data.columns):
                        # Créer une Heatmap pour les données numériques
                        fig = px.imshow(uploaded_data.corr(), title="Heatmap de corrélation des colonnes numériques")
                    elif plot_type == 'lineplot' and column_type == 'numeric' and len(column_data) > 1:
                        # Créer un Line Plot (tracer la colonne choisie en fonction de l'index)
                        fig = px.line(uploaded_data, x=uploaded_data.index, y=selected_column, title=f"Line Plot de {selected_column}")
                    else:
                        error = f"Type de graphique choisis:'{plot_type}' n'est pas compatible avec colonne {column_type}.  Veuillez choisir autre type de graphique"

                    if not error:
                        # Convertir le graphique Plotly en format JSON avec PlotlyJSONEncoder
                        graph = json.dumps(fig, cls=PlotlyJSONEncoder)

        except Exception as e:
            error = f"Erreur lors du traitement des données : {str(e)}"

    return render(request, 'csv_analyzer/visualization.html', {
        'columns': columns,  # Afficher la liste des colonnes
        'columns_with_types': columns_with_types,  # Colonnes avec leurs types
        'graph': graph,  # Graph en JSON si disponible
        'error': error,  # Message d'erreur si disponible
    })





def upload_csv(request):
    """Gestion de l'upload d'un fichier CSV."""
    errors = []
    headers = []
    data = []
    columns = []
    selected_column = None
    stats_result = None
    error = None

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        filepath = fs.path(filename)

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as file:  # Utilisation de utf-8-sig pour gérer le BOM
                reader = csv.reader(file)
                
                # Lire les entêtes (première ligne du fichier CSV)
                try:
                    headers = next(reader)
                    columns = headers  # Utilisation des headers comme colonnes
                except StopIteration:
                    errors.append("Le fichier CSV est vide.")
                    return render(request, 'csv_analyzer/upload_form.html', {'errors': errors})

                # Lire les données (lignes suivantes)
                for row_num, row in enumerate(reader, start=2):  # Début à la ligne 2 (1 pour les entêtes)
                    data_row = []
                    for cell_num, cell in enumerate(row):
                        try:
                            converted_cell = convert_cell(cell)
                            data_row.append(converted_cell)
                        except ValueError:
                            errors.append(f"Erreur à la ligne {row_num}, colonne {cell_num + 1}: '{cell}'.")
                            logger.error(f"Erreur de conversion dans le fichier {filename}, ligne {row_num}, cellule {cell_num + 1}.")
                    data.append(data_row)
        except Exception as e:
            errors.append(f"Erreur inattendue: {str(e)}")
            logger.exception(f"Erreur inattendue lors du traitement du fichier {filename}: {e}")
        finally:
            fs.delete(filename)

        # Si des erreurs ont été collectées, les afficher
        if errors:
            return render(request, 'csv_analyzer/upload_form.html', {'errors': errors})

        # Sauvegarder les données dans la session pour réutilisation
        request.session['uploaded_data'] = {'data': data, 'headers': headers}

        # Effectuer l'analyse des données
        stats_result = analyze_data(data, headers, errors)

        # Vérifier si une colonne a été sélectionnée via le GET
        if request.GET.get("column"):
            selected_column = request.GET["column"]

        # Rediriger vers la page de menu après téléchargement réussi
        return redirect('menu')

    else:
        # Si la méthode n'est pas POST, simplement afficher le formulaire d'upload
        return render(request, 'csv_analyzer/upload_form.html')
def data_preview(request):
    # Récupérer les données et en-têtes depuis la session
    uploaded_data = request.session.get('uploaded_data', None)

    if uploaded_data is not None:
        data = uploaded_data['data']
        headers = uploaded_data['headers']
        
        # Convertir les données en HTML pour les afficher dans un tableau
        data_html = "<table class='data-table'><thead><tr>"

        # Ajouter les en-têtes du tableau
        for header in headers:
            data_html += f"<th>{header}</th>"
        
        data_html += "</tr></thead><tbody>"

        # Ajouter les lignes de données
        for row in data:
            data_html += "<tr>"
            for cell in row:
                data_html += f"<td>{cell}</td>"
            data_html += "</tr>"
        
        data_html += "</tbody></table>"
    else:
        data_html = "No data available. Please upload a CSV file first."

    return render(request, 'csv_analyzer/data.html', {'data_html': data_html})

import csv
# ... autres imports

from django.shortcuts import render, redirect
import pandas as pd
def recherche(request):
    """Page de recherche dans les données CSV."""
    if request.method == "POST":
        search_type = request.POST.get("type")
        data_list = request.session.get("uploaded_data", {}).get("data", [])
        headers = request.session.get("uploaded_data", {}).get("headers", [])
        data = pd.DataFrame(data_list, columns=headers)

        context = {"data": data_list, "headers": headers}

        if search_type == "index":
            try:
                index = int(request.POST.get("index", -1))
                if 0 <= index < len(data):
                    context["row"] = data.iloc[index].tolist()
                    context["row_headers"] = headers
                else:
                    context["error"] = "Index hors limites."
            except ValueError:
                context["error"] = "Index invalide."

        elif search_type == "column":
            column = request.POST.get("column")
            if column in data.columns:
                context["column"] = data[column].tolist()
                context["column_name"] = column
            else:
                context["error"] = "Colonne non trouvée."

        elif search_type == "search":
            query = request.POST.get("query", "").strip()
            if query:
                matching_rows = data[data.applymap(lambda x: query.lower() in str(x).lower()).any(axis=1)]
                context["search_results"] = matching_rows.values.tolist()
                context["search_headers"] = headers
            else:
                context["error"] = "Veuillez entrer un mot ou un chiffre pour rechercher."

        return render(request, "csv_analyzer/recherche.html", context)

    elif request.method == "GET":
        if "uploaded_data" in request.session:
            data_list = request.session["uploaded_data"]["data"]
            headers = request.session["uploaded_data"]["headers"]
            return render(request, "csv_analyzer/recherche.html", {"data": data_list, "headers": headers})
        else:
            return redirect("upload_csv")

    return render(request, "csv_analyzer/recherche.html")

def menu(request):
    if 'uploaded_data' in request.session:
        try:
            data_list = request.session['uploaded_data']['data']
            headers = request.session['uploaded_data']['headers']

            # Convertir les données en DataFrame
            data = pd.DataFrame(data_list, columns=headers)

            # Analyse des données
            analysis_results, graphs = analyze_data(data_list, headers, [])

            return render(request, 'csv_analyzer/menu.html', {
                'data': data.to_html(classes='data-table', index=False),  # Convertir DataFrame en HTML pour affichage
                'headers': headers,
                'analysis': analysis_results,
                'graphs': graphs,
            })
        except Exception as e:
            logger.error(f"Erreur lors du rendu du menu : {str(e)}")
            return render(request, 'csv_analyzer/menu.html', {
                'error': "Une erreur est survenue lors de l'analyse des données. Veuillez réessayer."
            })
    else:
        # Rediriger vers la page d'upload si aucune donnée n'est disponible
        return redirect('upload_csv')



import pandas as pd

def statistical_analysis(request):
    # Récupérer les données téléchargées de la session
    uploaded_data = request.session.get('uploaded_data', None)
    selected_column = request.GET.get('column', None)  # La colonne choisie par l'utilisateur
    stats_result = {}  # Résultat de l'analyse
    error = None

    if uploaded_data is not None:
        # Convertir les données en DataFrame si nécessaire
        if isinstance(uploaded_data, dict):
            try:
                # Convertir les données en DataFrame à partir du dictionnaire
                uploaded_data = pd.DataFrame(uploaded_data['data'], columns=uploaded_data['headers'])
            except Exception as e:
                error = f"Erreur lors de la conversion des données : {str(e)}"
                uploaded_data = None

        # Si nous avons un DataFrame, procéder à l'analyse
        if isinstance(uploaded_data, pd.DataFrame):
            if not uploaded_data.empty:
                # Si une colonne est sélectionnée
                if selected_column and selected_column in uploaded_data.columns:
                    column_data = uploaded_data[selected_column]

                    # Vérifier si la colonne est numérique
                    if pd.api.types.is_numeric_dtype(column_data):
                        try:
                            numeric_data = pd.to_numeric(column_data.dropna(), errors='coerce')  # Ignorer les valeurs NaN
                            q1 = numeric_data.quantile(0.25)
                            q3 = numeric_data.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)].tolist()

                            # Retourner les résultats de l'analyse pour les données numériques
                            stats_result = {
                                'type': 'numeric',
                                'mean': numeric_data.mean(),
                                'median': numeric_data.median(),
                                'stddev': numeric_data.std(),
                                'min': numeric_data.min(),
                                'max': numeric_data.max(),
                                'outliers': outliers,
                            }
                        except Exception as e:
                            error = f"Erreur lors de l'analyse de la colonne numérique : {str(e)}"
                    else:
                        # Si la colonne est catégorielle
                        try:
                            mode = column_data.mode()[0] if not column_data.mode().empty else None
                            mode_count = column_data[column_data == mode].count() if mode is not None else 0
                            
                            # Retourner les résultats pour les données catégorielles
                            stats_result = {
                                'type': 'categorical',
                                'mode': mode,
                                'mode_count': mode_count,
                            }
                        except Exception as e:
                            error = f"Erreur lors de l'analyse de la colonne catégorielle : {str(e)}"
                
                else:
                    error = "La colonne sélectionnée n'existe pas ou n'a pas été choisie."
            else:
                error = "Les données sont vides."

        else:
            error = "Les données téléchargées ne sont pas dans un format valide (doivent être sous forme de DataFrame)."
    else:
        error = "Aucune donnée disponible. Veuillez d'abord télécharger un fichier CSV."

    # Passer les résultats à la vue pour l'affichage
    return render(request, 'csv_analyzer/statistics.html', {
        'columns': list(uploaded_data.columns) if isinstance(uploaded_data, pd.DataFrame) else [],
        'stats_result': stats_result,
        'error': error,
        'selected_column': selected_column,
    })

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_plot(data, plot_type, column):
    # Générer le graphique en fonction du type
    if plot_type == 'histplot':
        fig = sns.histplot(data[column], kde=True)
    elif plot_type == 'boxplot':
        fig = sns.boxplot(x=data[column])
    elif plot_type == 'scatterplot':
        fig = sns.scatterplot(data=data, x=column, y=data.columns[1])  # Exemple de scatter avec une autre colonne
    elif plot_type == 'kdeplot':
        fig = sns.kdeplot(data[column])
    elif plot_type == 'barplot':
        fig = sns.barplot(x=data[column].value_counts().index, y=data[column].value_counts())
    elif plot_type == 'countplot':
        fig = sns.countplot(x=data[column])
    
    # Convertir le graphique Seaborn en image PNG
    buf = BytesIO()
    fig.figure.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Retourner le HTML pour l'intégration
    return f'<img src="data:image/png;base64,{img_str}" />'
