import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import fisher_exact
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import missingno as msno
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Analyse de la Santé des Animaux Domestiques",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour télécharger un graphique
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
        return data
    except:
        st.error("Le fichier 'data.xlsx' n'a pas été trouvé. Utilisez l'option de téléchargement ci-dessous.")
        return None

# Fonction pour prétraiter les données
@st.cache_data
def preprocess_data(data):
    if data is None:
        return None
    
    # Copie des données
    data_imputed = data.copy()
    
    # Imputation des valeurs manquantes avec KNN
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        imputer_knn = KNNImputer(n_neighbors=5)
        data_knn = pd.DataFrame(imputer_knn.fit_transform(data[num_cols]),
                             columns=num_cols)
        
        for col in num_cols:
            data_imputed[col] = data_knn[col]
    
    # Correction des poids négatifs
    if 'Poids' in data_imputed.columns:
        negatif_poids_index = data_imputed[data_imputed['Poids'] < 0].index
        median_positive_weight = data_imputed[data_imputed['Poids'] > 0]['Poids'].median()
        data_imputed.loc[negatif_poids_index, 'Poids'] = median_positive_weight
    # Création de classes d'âge
    if 'Âge' in data_imputed.columns:
        data_imputed['Classe_age'] = pd.cut(data_imputed['Âge'],
            bins=[0, 1, 3, 7, 12, 20],
            labels=['<1 an', '1-3 ans', '3-7 ans', '7-12 ans', '12+ ans'])            
    
    return data_imputed

# Fonction pour l'ACP
@st.cache_data
def perform_pca(data, n_components=3):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclusion de certaines colonnes si nécessaire
    if 'Classe_age' in numeric_cols:
        numeric_cols.remove('Classe_age')
    
    # Standardisation
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numeric_cols])
    
    # ACP
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    
    # DataFrame des composantes
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    if 'Maladie' in data.columns:
        pca_df['Maladie'] = data['Maladie']
    
    return pca_df, pca, numeric_cols

# Navigation sidebar
st.sidebar.title("Navigation")
pages = [
    "Accueil",
    "Exploration des données",
    "Analyse descriptive",
    "Visualisations interactives",
    "Analyse multivariée",
    "Profils types",
    "Prédiction (Bonus)"
]
page = st.sidebar.radio("Aller à", pages)

# Option d'upload de fichier
st.sidebar.header("Données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel", type=["xlsx"])

# Chargement des données
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
else:
    data = load_data()

# Prétraitement des données
if data is not None:
    data_processed = preprocess_data(data)
else:
    data_processed = None

# ====================== PAGE D'ACCUEIL ======================
if page == "Accueil":
    st.title("🐾 Projet d'Analyse de Données sur la Santé des Animaux Domestiques")
    
    st.markdown("""
    ## Introduction
    
    ### Contexte et objectifs
    Ce projet vise à analyser les données de santé d'animaux domestiques pour identifier les facteurs de risque 
    associés à diverses maladies. L'objectif principal est de comprendre comment les caractéristiques physiologiques 
    et démographiques des animaux influencent leur état de santé.
    
    ### Description du dataset
    Le jeu de données contient des informations sur des animaux domestiques (chiens et chats) incluant leur espèce, 
    race, âge, poids, signes vitaux (température, respiration, pulse), niveau d'activité, qualité de sommeil 
    et diagnostic de maladie.
    
    ### Problématique
    Comment les caractéristiques physiologiques et démographiques des animaux domestiques influencent-elles leur 
    état de santé, et peut-on identifier des profils à risque pour certaines maladies?
    
    ### Objectifs
    - Identifier les facteurs les plus corrélés avec les différentes maladies.
    - Déterminer si certaines races ou espèces sont plus prédisposées à certaines maladies.
    - Analyser l'impact de l'âge et du poids sur la santé des animaux.
    - Visualiser les relations complexes entre les variables.
    - Construire des profils types d'animaux sains et malades.
    - Générer des recommandations personnalisées.
    """)
    
    
   
    
    
    
# ====================== EXPLORATION DES DONNÉES ======================
elif page == "Exploration des données":
    st.title("📊 Exploration des données")
    
    if data_processed is not None:
        st.header("Aperçu du dataset")
        
        # Option pour voir les données brutes ou prétraitées
        data_view = st.radio("Afficher", ["Données brutes", "Données prétraitées"])

        if data_view == "Données brutes":
            st.dataframe(data)
            
            
        else:
            st.dataframe(data_processed)
            
            # Section pour les méthodes de prétraitement
            st.subheader("Méthodes de prétraitement appliquées")
            
            with st.expander("1. Traitement des valeurs manquantes"):
                st.markdown("""
                **Trois méthodes testées :**
                - Imputation par moyenne
                - Imputation par médiane
                - Imputation par KNN (k-plus proches voisins)
                """)
                
                # Méthodes d'imputation
                imputer_mean = SimpleImputer(strategy='mean')
                data_mean = pd.DataFrame(imputer_mean.fit_transform(data.select_dtypes(include=['float64', 'int64'])),
                                        columns=data.select_dtypes(include=['float64', 'int64']).columns)

                imputer_median = SimpleImputer(strategy='median')
                data_median = pd.DataFrame(imputer_median.fit_transform(data.select_dtypes(include=['float64', 'int64'])),
                                          columns=data.select_dtypes(include=['float64', 'int64']).columns)

                imputer_knn = KNNImputer(n_neighbors=5)
                data_knn = pd.DataFrame(imputer_knn.fit_transform(data.select_dtypes(include=['float64', 'int64'])),
                                       columns=data.select_dtypes(include=['float64', 'int64']).columns)
                
                # Comparaison des méthodes
                st.markdown("**Comparaison des méthodes d'imputation (exemple pour la Température)**")
                plt.figure(figsize=(12, 6))
                sns.kdeplot(data['Température'].dropna(), label='Original', color='black')
                sns.kdeplot(data_mean['Température'], label='Moyenne', linestyle='--')
                sns.kdeplot(data_median['Température'], label='Médiane', linestyle=':')
                sns.kdeplot(data_knn['Température'], label='KNN', linestyle='-.')
                plt.title('Comparaison des méthodes d\'imputation')
                plt.legend()
                st.pyplot(plt)
                
                st.markdown("""
                **Choix final :** Méthode KNN  
                *Justification : Préserve mieux la distribution originale des données*
                """)
            
            with st.expander("2. Correction des anomalies (Poids négatifs)"):
                st.markdown("""
                **Procédure appliquée :**
                - Identification des valeurs négatives ({} cas détectés)
                - Remplacement par la médiane des valeurs positives
                """.format(len(data[data['Poids'] < 0])))
                
                # Visualisation avant/après correction
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(data['Poids'], ax=ax1)
                ax1.set_title('Distribution avant correction')
                sns.boxplot(data_processed['Poids'], ax=ax2)
                ax2.set_title('Distribution après correction')
                st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            if data_view == "Données brutes":
                st.header("Structure du dataset")
                buffer = io.StringIO()
                data.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)
                
                # Ajout de l'interprétation de la structure
                st.markdown("""
                **Interprétation - Colonnes avec valeurs manquantes :**
                
                Un **nettoyage des données manquantes** est nécessaire avant toute analyse.
                * Certaines colonnes ont des **valeurs manquantes** (ex : `Âge`, `Poids`, `Température`, etc.).
                """)
                
                # Ajout de la description des variables
                st.header("Description des Variables")
                var_description = pd.DataFrame({
                    'Variable': ['Espèce', 'Âge', 'Poids', 'Race', 'Température', 'Respiration', 'Pulse', 
                                'Intensité_activité', 'Score_sommeil', 'Maladie'],
                    'Type': ['Catégorielle', 'Numérique', 'Numérique', 'Catégorielle', 'Numérique', 
                            'Numérique', 'Numérique', 'Numérique', 'Numérique', 'Catégorielle'],
                    'Description': [
                        "L'espèce de l'animal (ex : chien, chat, etc.)",
                        "Âge de l'animal (en années)",
                        "Poids de l'animal (en kilogrammes)",
                        "La race spécifique de l'animal au sein de son espèce",
                        "Température corporelle de l'animal (en degrés Celsius)",
                        "Fréquence respiratoire de l'animal (nombre de respirations par minute)",
                        "Fréquence cardiaque de l'animal (battements par minute)",
                        "Niveau d'activité physique de l'animal",
                        "Évaluation de la qualité ou de la durée du sommeil de l'animal",
                        "Type de maladie diagnostiquée ou état de santé de l'animal"
                    ]
                })
                st.table(var_description)
            else:
                st.header("Structure du dataset")
                buffer = io.StringIO()
                data_processed.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)

        with col2:
            if data_view == "Données brutes":
                st.header("Statistiques descriptives")
                st.dataframe(data.describe())
                # Ajout de l'interprétation
                st.markdown("""
                **Interprétation:**
                - Variables comme Température, Score_sommeil ➔ stables et peu dispersées.
                - Variables comme Poids, Pulse ➔ grande variabilité et possibles valeurs extrêmes.
                - Intensité_activité ➔ distribution très asymétrique (beaucoup d'inactifs).
                - Présence d'anomalies ➔ surtout sur Poids (valeurs négatives).
                """)
            else:
                st.header("Statistiques descriptives")
                st.dataframe(data_processed.describe())

        st.header("Valeurs manquantes")
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(data, ax=ax)
        st.pyplot(fig)

        st.download_button(
            label="Télécharger les données prétraitées",
            data=data_processed.to_csv(index=False).encode('utf-8'),
            file_name='data_processed.csv',
            mime='text/csv',
        )
    else:
        st.error("Impossible d'afficher l'exploration des données. Aucun dataset n'est chargé.")

# ====================== ANALYSE DESCRIPTIVE ======================
elif page == "Analyse descriptive":
    st.title("📈 Analyse descriptive")
    
    if data_processed is not None:
        tab1, tab2 = st.tabs(["Variables quantitatives", "Variables qualitatives"])
        
        with tab1:
            st.header("Distribution des variables quantitatives")
            
            num_vars = data_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'Classe_age' in num_vars:
                num_vars.remove('Classe_age')
            
            selected_var = st.selectbox("Choisir une variable", num_vars)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data_processed[selected_var].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution de {selected_var}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Boxplot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=data_processed[selected_var], ax=ax)
                ax.set_title(f"Boxplot de {selected_var}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Statistiques détaillées
            st.subheader(f"Statistiques détaillées - {selected_var}")
            stats = data_processed[selected_var].describe().to_frame().T
            stats['skew'] = data_processed[selected_var].skew()
            stats['kurtosis'] = data_processed[selected_var].kurtosis()
            st.dataframe(stats)
            
        with tab2:
            st.header("Distribution des variables qualitatives")
            
            cat_vars = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'Classe_age' in data_processed.columns:
                cat_vars.append('Classe_age')
            
            if cat_vars:
                selected_cat_var = st.selectbox("Choisir une variable", cat_vars)
                
                # Diagramme en barres
                fig, ax = plt.subplots(figsize=(12, 6))
                counts = data_processed[selected_cat_var].value_counts()
                sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
                ax.set_title(f"Distribution de {selected_cat_var}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Tableau de fréquences
                freq_table = pd.DataFrame({
                    'Fréquence': counts,
                    'Pourcentage (%)': counts / counts.sum() * 100
                })
                st.dataframe(freq_table)
                
                # Interprétations spécifiques par variable
                st.subheader("Interprétation")
                
                if selected_cat_var == 'Espèce':
                    st.markdown("""
                    **Distribution par espèce :**
                    - Chiens (environ 550)
                    - Chats (environ 520)
                    
                    **Analyse :**
                    - Répartition équilibrée entre chiens et chats
                    - Permet des comparaisons fiables entre les deux espèces
                    """)
                
                elif selected_cat_var == 'Race':
                    st.markdown("""
                    **Distribution par race :**
                    - Golden Retriever (environ 170 individus)
                    - Husky, Bengal et Dalmatien (90-100 individus)
                    - Siamois, Persan et Sphynx (20-40 individus)
                    
                    **Analyse :**
                    - Variabilité des races représentées
                    - Permet d'identifier des prédispositions raciales spécifiques
                    - Certaines races moins représentées (à prendre en compte dans les analyses)
                    """)
                
                elif selected_cat_var == 'Maladie':
                    st.markdown("""
                    **Distribution par maladie :**
                    - Maladies infectieuses (environ 250 cas)
                    - Maladies respiratoires et cardiaques (230 et 210 cas)
                    - Maladies endocriniennes (environ 100 cas)
                    - Animaux sains (environ 100 individus)
                    
                    **Analyse :**
                    - Échantillon représentatif avec suffisamment de cas par catégorie
                    - Permet des conclusions statistiquement valides
                    - Groupe témoin (animaux sains) présent pour comparaison
                    """)
                
                else:
                    st.markdown("""
                    **Analyse générale :**
                    - Distribution représentative des différentes catégories
                    - Permet une analyse statistique fiable
                    - Vérifier les effectifs minimums pour les catégories moins représentées
                    """)
                
            else:
                st.info("Aucune variable catégorielle détectée dans le dataset.")
            
            if 'Maladie' in data_processed.columns and len(cat_vars) > 1:
                st.subheader("Analyse d'association avec la maladie")
                
                cross_var = st.selectbox("Croiser avec Maladie", [v for v in cat_vars if v != 'Maladie'])
                
                # Tableau croisé avec mise en forme
                st.markdown("### 1. Tableau de contingence")
                cross_tab = pd.crosstab(data_processed[cross_var], data_processed['Maladie'])
                st.dataframe(cross_tab.style.background_gradient(cmap='Blues'))
                st.markdown("""**Interpretation:**

Les chats sont plus touchés par le cancer, tandis que les chiens souffrent davantage de problèmes cardiaques/respiratoires.

Les maladies endocriniennes (diabète, hyperthyroïdie) sont légèrement plus fréquentes chez les chats.""")
                
                # Test du Chi2
                st.markdown("### 2. Test d'indépendance du Chi2")
                chi2, p, dof, expected = chi2_contingency(cross_tab)
                st.write(f"- χ² = {chi2:.2f}")
                st.write(f"- p-value = {p:.4f}")
                st.write(f"- Degrés de liberté = {dof}")
                
                if p < 0.01:
                    st.success("Association très significative (p < 0.01)")
                elif p < 0.05:
                    st.success("Association significative (p < 0.05)")
                elif p < 0.1:
                    st.info("Association marginalement significative (p < 0.10)")
                else:
                    st.info("Pas d'association significative (p ≥ 0.10)")
                
                # Analyse des rapports de cotes (OR) pour les principales maladies
                if cross_var == 'Espèce':
                    st.markdown("### 3. Analyse des rapports de cotes (OR)")
                    
                    # Fonction pour calculer l'OR et son intervalle de confiance
                    def calculate_or(row, maladie):
                        # Tableau 2x2 : [[a, b], [c, d]]
                        a = row['chat']  # chats avec la maladie
                        b = cross_tab.loc['chat', cross_tab.columns != maladie].sum()  # chats sans la maladie
                        c = row['chien']  # chiens avec la maladie
                        d = cross_tab.loc['chien', cross_tab.columns != maladie].sum()  # chiens sans la maladie

                        # Calcul OR et p-value (test exact de Fisher)
                        oddsratio, pvalue = fisher_exact([[a, b], [c, d]])

                        # Intervalle de confiance à 95%
                        log_or = np.log(oddsratio)
                        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
                        ci_low = np.exp(log_or - 1.96*se)
                        ci_high = np.exp(log_or + 1.96*se)

                        return oddsratio, pvalue, (ci_low, ci_high)
                    
                    # Calcul pour chaque maladie significative
                    for maladie in cross_tab.columns:
                        try:
                            or_value, p_value, ci = calculate_or(cross_tab.loc[:, maladie], maladie)
                            st.write(f"**{maladie}**:")
                            st.write(f"- OR = {or_value:.2f} [IC95%: {ci[0]:.2f}-{ci[1]:.2f}]")
                            st.write(f"- p-value = {p_value:.4f}")
                            
                            if p_value < 0.05:
                                if or_value > 1:
                                    st.success(f"Les chats ont {or_value:.1f} fois plus de risque de {maladie} que les chiens (significatif)")
                                else:
                                    st.success(f"Les chiens ont {1/or_value:.1f} fois plus de risque de {maladie} que les chats (significatif)")
                            else:
                                st.info("Pas d'association significative")
                                
                            st.write("---")
                        except:
                            continue
                    
                    st.markdown("""
                    **Interprétation des rapports de cotes :**
                    - OR > 1 : Risque accru pour les chats
                    - OR < 1 : Risque accru pour les chiens
                    - IC95% qui ne contient pas 1 : Association significative
                    """)
                
                elif cross_var == 'Race':
                    st.markdown("""
                    **Analyse par race :**
                    - Aucune association significative entre race et maladie (p = 0.2227)
                    - L'espèce semble être un facteur plus déterminant que la race
                    
                    **Conclusion :**
                    Les prédispositions pathologiques semblent liées principalement à l'espèce plutôt qu'à la race spécifique.
                    """)
                
                else:
                    st.markdown("""
                    **Analyse générale :**
                    - Le test du Chi2 évalue l'indépendance entre deux variables qualitatives
                    - Une p-value faible (<0.05) suggère que les variables sont associées
                    - Pour les tableaux 2x2, le rapport de cotes (OR) quantifie l'intensité de cette association
                    """)
    else:
        st.error("Impossible d'afficher l'analyse descriptive. Aucun dataset n'est chargé.")
# ====================== VISUALISATIONS INTERACTIVES ======================
elif page == "Visualisations interactives":
    st.title("🔍 Visualisations interactives")
    
    if data_processed is not None:
        viz_type = st.radio("Type de visualisation", 
                            ["Distribution par âge",
                             "Relation Variables vs Maladie", 
                             "Analyse bivariée des variables qualitatives",
                             "Matrice de corrélation"])
        
        
        
        if viz_type == "Distribution par âge":
            st.header("Répartition par classe d'âge")
            
            if 'Âge' in data_processed.columns:
                # Création de classes d'âge adaptées aux animaux si elles n'existent pas déjà
                if 'Classe_age' not in data_processed.columns:
                    data_processed['Classe_age'] = pd.cut(data_processed['Âge'],
                                        bins=[0, 1, 3, 7, 12, 20],  # Plages adaptées aux animaux
                                        labels=['<1 an', '1-3 ans', '3-7 ans', '7-12 ans', '12+ ans'])
                
                # Distribution par classe d'âge
                fig = px.histogram(data_processed, x='Classe_age', 
                                   color='Maladie' if 'Maladie' in data_processed.columns else None,
                                   barmode='group',
                                   title="Répartition des pets par classe d'âge",
                                   height=500)
                
                fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['<1 an', '1-3 ans', '3-7 ans', '7-12 ans', '12+ ans']})
                st.plotly_chart(fig, use_container_width=True)
                
                
                
                # Ajout de l'interprétation
                st.subheader("Interprétation:")
                
                st.markdown("""
                **Répartition par âge :**
                - La majorité des animaux ont entre 7 et 12 ans.
                - Les animaux de 3 à 7 ans sont aussi nombreux.
                - Peu d'animaux sont âgés de moins de 3 ans.
                
                **État de santé par âge :**
                - Les maladies augmentent avec l'âge, surtout entre 7-12 ans et 12+ ans.
                - Cancers et maladies respiratoires dominent chez les 7-12 ans.
                - Maladies cardiaques fréquentes chez les 12+ ans.
                - Les jeunes animaux (< 3 ans) présentent principalement des maladies infectieuses.
                
                **Animaux sains :**
                - La proportion d'animaux sains est faible dans toutes les tranches d'âge.
                - Très peu d'animaux sains chez les plus âgés (12+ ans).
                """)
            else:
                st.warning("La variable 'Âge' n'est pas disponible dans le dataset.")

        elif viz_type == "Relation Variables vs Maladie":
            if 'Maladie' in data_processed.columns:
                st.header("Relation entre variables et maladies")
                
                # Choix de la variable
                num_vars = data_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if 'Classe_age' in num_vars:
                    num_vars.remove('Classe_age')
                
                var_to_plot = st.selectbox("Choisir une variable", num_vars)
                
                # Boxplot interactif
                fig = px.box(data_processed, x='Maladie', y=var_to_plot, 
                             color='Maladie', notched=True,
                             title=f"Distribution de {var_to_plot} par type de maladie")
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Ajout de l'interprétation selon la variable choisie
                st.subheader("Interprétation:")
                
                # Interprétation spécifique selon la variable sélectionnée
                if var_to_plot == "Age" or var_to_plot == "Âge":
                    st.markdown("**Âge** :Répartition générale :Tous les groupes de maladies ont une distribution assez large des âges (entre ~1 à ~15 ans).Cela montre que les maladies touchent aussi bien les jeunes que les plus vieux.Cancer & maladies cardiaques :Médiane un peu plus élevée (~9 ans), ce qui suggère que ces maladies sont plus fréquentes chez les animaux plus âgés.Animaux sains :Médiane autour de 8 ans, un peu plus basse que certains types de maladies.Cela peut indiquer qu’il y a aussi beaucoup de jeunes animaux en bonne santé.")
                elif var_to_plot == "Poids":
                    st.markdown("**Poids** : → Les animaux malades ont tendance à avoir un poids plus faible que les animaux sains.")
                elif var_to_plot == "Temperature" or var_to_plot == "Température":
                    st.markdown("""
                    **Température** : 
                    - Température élevée → associée aux maladies infectieuses et respiratoires.
                    - Température basse → associée aux maladies endocriniennes.
                    """)
                elif var_to_plot == "Respiration":
                    st.markdown("""
                    **Respiration** : 
                    - Fréquence respiratoire élevée → signe de maladies respiratoires ou cardiaques.
                    - Respiration plus faible → observée dans les maladies endocriniennes.
                    """)
                elif var_to_plot == "Pulse" or var_to_plot == "Pulse":
                    st.markdown("""
                    **Pulse (Fréquence cardiaque)** : 
                    - Pulse bas → lié aux cancers et maladies endocriniennes.
                    - Pulse élevé → observé chez les animaux sains et ceux atteints de maladies infectieuses.
                    """)
                else:
                    st.info(f"Aucune interprétation spécifique disponible pour la variable {var_to_plot}.")
                
                
                
            else:
                st.warning("La variable 'Maladie' n'est pas disponible dans le dataset.")
        
        elif viz_type == "Analyse bivariée des variables qualitatives":
            st.header("Analyse bivariée des variables qualitatives par rapport à la maladie")
            
            st.markdown("""
            Analysons la relation entre les variables catégorielles et la présence de maladie.
            """)
            
            if 'Maladie' in data_processed.columns:
                # Variables qualitatives (hors Maladie)
                qualitative_vars = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
                if 'Maladie' in qualitative_vars:
                    qualitative_vars.remove('Maladie')
                
                if qualitative_vars:
                    # Sélection de la variable à visualiser
                    var_to_plot = st.selectbox("Choisir une variable qualitative à croiser avec 'Maladie'", qualitative_vars)
                    
                    # Création du graphique avec Plotly
                    fig = px.histogram(data_processed, 
                                      x=var_to_plot, 
                                      color='Maladie',
                                      barmode='group',
                                      title=f'Répartition de {var_to_plot} par maladie',
                                      height=600)
                    
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ajout de l'interprétation selon la variable choisie
                    st.subheader("Interprétation:")
                    
                    # Interprétation spécifique selon la variable sélectionnée
                    if var_to_plot.lower() == "espèce" or var_to_plot.lower() == "espece":
                        st.markdown("""
                        **Relation entre Espèce et Maladie**
                        - Les chats sont plus susceptibles aux cancers et maladies endocriniennes
                        - Les chiens présentent davantage de maladies cardiaques
                        - Les maladies infectieuses et respiratoires affectent les deux espèces avec une légère prévalence chez les chiens
                        """)
                    elif var_to_plot.lower() == "race":
                        st.markdown("""
                        **Relation entre Race et Maladie**
                        - Certaines races présentent des prédispositions spécifiques à certaines maladies
                        - Les données montrent des variations importantes dans la distribution des maladies selon les races
                        """)
                    else:
                        st.info(f"Aucune interprétation spécifique disponible pour la relation entre {var_to_plot} et Maladie.")
                    
                    # Tableau croisé pour analyse statistique
                    st.subheader("Tableau croisé")
                    
                    # Création du tableau croisé
                    crosstab = pd.crosstab(data_processed[var_to_plot], data_processed['Maladie'])
                    st.dataframe(crosstab)
                    
                    # Tableau en pourcentage
                    st.subheader("Tableau en pourcentage par ligne")
                    crosstab_pct = pd.crosstab(data_processed[var_to_plot], data_processed['Maladie'], normalize='index') * 100
                    crosstab_pct = crosstab_pct.round(2)
                    st.dataframe(crosstab_pct)
                    
                    # Option pour afficher plusieurs variables
                    if len(qualitative_vars) > 1:
                        st.subheader("Visualisation multiple")
                        show_multiple = st.checkbox("Afficher plusieurs variables croisées avec 'Maladie'")
                        
                        if show_multiple:
                            # Limiter à 4 variables pour des raisons de lisibilité
                            vars_to_display = st.multiselect("Sélectionner jusqu'à 2 variables", 
                                                           qualitative_vars, 
                                                           default=qualitative_vars[:min(2, len(qualitative_vars))])
                            
                            if vars_to_display:
                                # Créer un subplot pour chaque variable
                                from plotly.subplots import make_subplots
                                import plotly.graph_objects as go
                                
                                fig = make_subplots(rows=len(vars_to_display), 
                                                   cols=1,
                                                   subplot_titles=[f'Répartition de {var} par maladie' for var in vars_to_display])
                                
                                for i, var in enumerate(vars_to_display):
                                    # Créer un dataframe de comptage
                                    count_df = pd.crosstab(data_processed[var], data_processed['Maladie'])
                                    
                                    # Pour chaque catégorie de maladie
                                    for j, disease in enumerate(count_df.columns):
                                        fig.add_trace(
                                            go.Bar(
                                                x=count_df.index,
                                                y=count_df[disease],
                                                name=disease,
                                                legendgroup=disease,
                                                showlegend=True if i==0 else False
                                            ),
                                            row=i+1, col=1
                                        )
                                
                                height_per_subplot = 400
                                fig.update_layout(
                                    height=height_per_subplot * len(vars_to_display),
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucune variable qualitative (hors 'Maladie') n'a été trouvée dans le dataset.")
            else:
                st.error("La variable 'Maladie' n'est pas disponible dans le dataset. Cette analyse n'est pas possible.")
        
        elif viz_type == "Matrice de corrélation":
            st.header("Matrice de corrélation")
            
            # Calcul de la matrice de corrélation
            num_vars = data_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'Classe_age' in num_vars:
                num_vars.remove('Classe_age')
            
            corr_matrix = data_processed[num_vars].corr()
            
            # Visualisation avec heatmap
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           color_continuous_scale='RdBu_r',
                           aspect="auto",
                           title="Matrice de corrélation des variables quantitatives")
            
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation des corrélations fortes
            st.subheader("Corrélations notables")
            
            # Extraire les paires avec une corrélation absolue > 0.4 (hors diagonale)
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.4:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Corrélation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                strong_corr_df = strong_corr_df.sort_values('Corrélation', key=abs, ascending=False)
                st.dataframe(strong_corr_df)
            else:
                st.info("Aucune corrélation forte (|r| > 0.4) n'a été détectée.")
                
            # Ajout de l'interprétation détaillée
            st.subheader("Interprétation:")
            st.markdown("""
            **Corrélations fortes (positives et négatives)**
            - **Respiration et Pulse** : Forte corrélation positive (0.73) - quand la fréquence respiratoire augmente, le pulse tend également à augmenter, ce qui est physiologiquement cohérent.
            - **Intensité d'activité et Score de sommeil** : Forte corrélation négative (-0.65) - plus l'activité physique est intense, plus la qualité du sommeil diminue.
            - **Poids et Respiration** : Corrélation négative modérée (-0.42) - les animaux de poids plus élevé tendent à avoir une fréquence respiratoire plus basse.
            
            **Corrélations modérées**
            - **Poids et Pulse** : Corrélation négative (-0.33) - le pulse tend à être plus bas chez les animaux de poids plus élevé.
            - **Température et Respiration** : Corrélation positive (0.34) - une température corporelle plus élevée s'accompagne d'une augmentation de la fréquence respiratoire.
            - **Température et Pulse** : Corrélation positive (0.27) - la fièvre tend à accélérer le rythme cardiaque.
            """)
    else:
        st.error("Impossible d'afficher les visualisations. Aucun dataset n'est chargé.")
# ====================== ANALYSE MULTIVARIÉE ======================
elif page == "Analyse multivariée":
    st.title("🔄 Analyse multivariée")
    
    if data_processed is not None:
        analysis_type = st.radio("Type d'analyse", ["ACP (Analyse en Composantes Principales)", 
                                                   "AC (Analyse des Correspondances)"])
        
        if analysis_type == "ACP (Analyse en Composantes Principales)":
            st.header("Analyse en Composantes Principales (ACP)")
            
            # Calcul de l'ACP
            n_components = st.slider("Nombre de composantes", 2, 5, 3)
            pca_results, pca_model, features = perform_pca(data_processed, n_components)
            
            # Affichage des variances expliquées
            explained_var = pca_model.explained_variance_ratio_ * 100
            cumulative_var = np.cumsum(explained_var)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Variance expliquée")
                fig = px.bar(x=[f"PC{i+1}" for i in range(len(explained_var))],
                            y=explained_var,
                            labels={'x': 'Composante', 'y': 'Variance expliquée (%)'},
                            title="Variance expliquée par composante")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Variance cumulée")
                fig = px.line(x=[f"PC{i+1}" for i in range(len(cumulative_var))],
                             y=cumulative_var,
                             markers=True,
                             labels={'x': 'Composante', 'y': 'Variance cumulée (%)'},
                             title="Variance cumulée")
                
                fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
                st.plotly_chart(fig, use_container_width=True)
            
            # Ajout de l'interprétation de la variance expliquée
            st.subheader("Interprétation de la variance")
            st.write("""
            Le graphique des valeurs propres montre que les trois premières composantes principales (PC1, PC2 et PC3) résument à elles seules 70 % de l'information contenue dans les données. Cela signifie qu'on peut réduire le nombre de dimensions de l'analyse à 3 sans perdre beaucoup d'informations. La courbe rouge (variance cumulée) atteint environ 96,5 % avec les cinq premières composantes, mais on remarque un "coude" clair après la 3ᵉ, ce qui indique que les composantes suivantes (PC4 et PC5) n'apportent presque rien de plus. En résumé, 3 dimensions suffisent pour bien représenter les données tout en simplifiant l'analyse.
            """)

            # Cercle des corrélations
            st.subheader("Cercle des corrélations")
            
            # Sélection des axes pour le cercle des corrélations
            dim1, dim2 = st.columns(2)
            with dim1:
                pc_x = st.selectbox("Axe X", [f"PC{i+1}" for i in range(n_components)], index=0)
            with dim2:
                pc_y = st.selectbox("Axe Y", [f"PC{i+1}" for i in range(n_components)], index=1)
            
            # Extraction des coordonnées des variables
            loadings = pca_model.components_.T
            pc_idx_x = int(pc_x[2]) - 1
            pc_idx_y = int(pc_y[2]) - 1
            
            # Création du cercle des corrélations
            fig = go.Figure()
            
            # Cercle unité
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = np.cos(theta)
            y_circle = np.sin(theta)
            fig.add_trace(go.Scatter(x=x_circle, y=y_circle, mode='lines', 
                                    line=dict(color='grey', width=1, dash='dash'),
                                    name='Cercle unité'))
            
            # Axes
            fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, 
                         line=dict(color="grey", width=1))
            fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, 
                         line=dict(color="grey", width=1))
            
            # Flèches pour les variables
            for i, feature in enumerate(features):
                fig.add_trace(go.Scatter(x=[0, loadings[i, pc_idx_x]], 
                                        y=[0, loadings[i, pc_idx_y]],
                                        mode='lines+markers+text',
                                        name=feature,
                                        line=dict(color='red', width=2),
                                        marker=dict(size=4),
                                        text=[None, feature],
                                        textposition="top center"))
            
            fig.update_layout(
                title=f"Cercle des corrélations ({pc_x} vs {pc_y})",
                xaxis_title=f"{pc_x} ({explained_var[pc_idx_x]:.1f}%)",
                yaxis_title=f"{pc_y} ({explained_var[pc_idx_y]:.1f}%)",
                xaxis=dict(range=[-1.1, 1.1]),
                yaxis=dict(range=[-1.1, 1.1]),
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Ajout de l'interprétation du cercle des corrélations
            st.subheader("Interprétation du cercle des corrélations")
            st.write("""
            Ce cercle des corrélations illustre les relations entre les variables physiologiques et les deux premières 
            composantes principales (PC1 et PC2), expliquant ensemble 55,6% de la variance totale. La disposition des 
            variables révèle que :
            
            - **PC1 (31,4% de variance)** est fortement corrélée avec :
              - Intensité_activité et Poids (à droite, corrélations positives)
              - Score_sommeil (à gauche, corrélations négatives), suggérant un axe opposant l'activité physique au repos.
            
            - **PC2 (24,2% de variance)** est liée à :
              - Température et Respiration (en haut)
              - Âge et Pulse (en bas), reflétant probablement un gradient de métabolisme ou d'état physiologique.
            
            Les variables proches du cercle unité (comme Intensité_activité) ont une contribution majeure à la structure des 
            données, tandis que leur regroupement par secteurs indique des corrélations internes (ex : Poids et Âge semblent 
            associés). Cette analyse permet d'identifier les dimensions clés structurant la variabilité des données animales.
            """)
            
            # Interprétation des axes
            st.subheader("Contribution des variables aux axes")
            
            # Calcul des contributions
            loadings_squared = loadings**2
            contributions = loadings_squared / np.sum(loadings_squared, axis=0)
            contrib_df = pd.DataFrame(contributions, index=features, 
                                     columns=[f"PC{i+1}" for i in range(n_components)])
            
            fig = px.bar(contrib_df.reset_index().melt(id_vars='index', var_name='Composante', value_name='Contribution'),
                        x='index', y='Contribution', color='Composante', barmode='group',
                        labels={'index': 'Variable'},
                        title="Contribution des variables aux composantes principales")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des contributions
            st.dataframe(contrib_df.style.highlight_max(axis=0))
            
            
        elif analysis_type == "AC (Analyse des Correspondances)":
            st.header("Analyse des Correspondances (AC)")
            
            # Message d'information sur l'AC
            st.info("""
            L'Analyse des Correspondances est particulièrement adaptée pour explorer les relations entre variables 
            catégorielles. Elle nous permet de visualiser les associations entre les modalités de ces variables.
            """)
            
            # Vérification des variables catégorielles
            cat_vars = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'Classe_age' in data_processed.columns:
                cat_vars.append('Classe_age')
            
            if len(cat_vars) >= 2:
                # Sélection des variables pour l'AC
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Variable en lignes", cat_vars, index=0, key="ac_var1")
                with col2:
                    var2 = st.selectbox("Variable en colonnes", [c for c in cat_vars if c != var1], 
                                       index=0 if cat_vars[0] != var1 else 1, key="ac_var2")
                
                # Paramètres supplémentaires
                with st.expander("Paramètres avancés"):
                    color_palette = st.text_input("Palette de couleurs (format JSON)", 
                                                 value='{"chat":"#1f77b4", "chien":"#ff7f0e"}', 
                                                 help="Dictionnaire JSON de paires 'valeur: couleur'")
                    try:
                        palette = json.loads(color_palette)
                    except:
                        st.warning("Format de palette invalide. Utilisation de la palette par défaut.")
                        palette = {"chat":"#1f77b4", "chien":"#ff7f0e"}
                
                # Création du tableau de contingence
                if var1 not in data_processed.columns or var2 not in data_processed.columns:
                    missing_vars = [v for v in [var1, var2] if v not in data_processed.columns]
                    st.error(f"Erreur : Variable(s) manquante(s) : {missing_vars}")
                else:
                    cont_table = pd.crosstab(data_processed[var1], data_processed[var2])
                    
                    # Vérification de la taille du tableau
                    if cont_table.shape[0] < 2 or cont_table.shape[1] < 2:
                        st.error(f"Erreur : Tableau trop petit ({cont_table.shape[0]}x{cont_table.shape[1]}). Minimum 2x2 requis.")
                    else:
                        # Affichage du tableau de contingence
                        st.subheader("Tableau de contingence")
                        st.dataframe(cont_table)
                        
                        # Calcul de l'AC avec prince
                        n_components = min(2, cont_table.shape[0]-1, cont_table.shape[1]-1)
                        
                        # On utilise try/except pour gérer les éventuelles erreurs
                        try:
                            from prince import CA
                            ca = CA(n_components=n_components)
                            ca.fit(cont_table)
                            
                            # Métriques
                            total_inertia = sum(ca.eigenvalues_)
                            perc_inertia = [eig/total_inertia*100 for eig in ca.eigenvalues_]
                            
                            # Affichage des résultats
                            st.subheader("Résultats de l'AC")
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.write(f"Dimensions calculées : {n_components}")
                                for i in range(n_components):
                                    st.write(f"Axe F{i+1} - Valeur propre : {ca.eigenvalues_[i]:.4f} ({perc_inertia[i]:.1f}%)")
                            
                            # Test Chi2
                            with metrics_col2:
                                from scipy.stats import chi2_contingency
                                chi2, p, dof, expected = chi2_contingency(cont_table)
                                st.write(f"Test d'indépendance Chi2 : χ² = {chi2:.3f}, p-value = {p:.4f}")
                                
                                if p < 0.05:
                                    st.success("Association significative (p < 0.05)")
                                else:
                                    st.info("Pas d'association significative (p ≥ 0.05)")
                            
                            # Récupération des coordonnées
                            rows = ca.row_coordinates(cont_table)
                            cols = ca.column_coordinates(cont_table)
                            
                            # Visualisations
                            if n_components == 1:
                                # Graphique unidimensionnel
                                st.subheader("Représentation unidimensionnelle")
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Tracé des modalités de la variable en lignes
                                ax.scatter(rows[0], rows[0], c='blue', label=var1, s=100)
                                for i, txt in enumerate(rows.index):
                                    ax.annotate(txt, (rows[0][i], rows[0][i]),
                                              color='blue', ha='center', va='center',
                                              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                                
                                # Tracé des modalités de la variable en colonnes
                                ax.scatter(cols[0], cols[0], c='red', label=var2, s=100, marker='s')
                                for i, txt in enumerate(cols.index):
                                    ax.annotate(txt, (cols[0][i], cols[0][i]),
                                              color='red', ha='center', va='center',
                                              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                                
                                ax.set_xlabel('Dimension 1')
                                ax.set_ylabel('Dimension 1')  # Même dimension sur les deux axes
                                ax.set_title(f'AC - {var1} vs {var2}\n(F1 explique {perc_inertia[0]:.1f}% de l\'inertie)')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                # Profils relatifs
                                st.subheader(f"Profils relatifs par {var1}")
                                st.dataframe(cont_table.div(cont_table.sum(axis=1), axis=0).style.format('{:.1%}'))
                                
                            elif n_components >= 2:
                                # Graphique bidimensionnel
                                st.subheader("Représentation bidimensionnelle")
                                
                                fig, ax = plt.subplots(figsize=(10, 10))
                                
                                # Tracé des modalités de la variable en lignes
                                for modality in rows.index:
                                    ax.scatter(rows.loc[modality, 0], rows.loc[modality, 1],
                                             color=palette.get(modality, 'gray'), s=200, label=modality)
                                    ax.text(rows.loc[modality, 0], rows.loc[modality, 1], modality,
                                          ha='center', va='bottom', fontsize=12)
                                
                                # Tracé des modalités de la variable en colonnes
                                for modality in cols.index:
                                    ax.scatter(cols.loc[modality, 0], cols.loc[modality, 1],
                                             color='red', marker='s', s=200)
                                    ax.text(cols.loc[modality, 0], cols.loc[modality, 1], modality,
                                          ha='center', va='bottom', fontsize=10, color='red')
                                
                                # Cercle et axes
                                ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
                                ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
                                ax.set_title(f"AC 2D - {var1} vs {var2}\nF1: {perc_inertia[0]:.1f}% | F2: {perc_inertia[1]:.1f}%", pad=20)
                                ax.set_xlabel("Axe F1")
                                ax.set_ylabel("Axe F2")
                                ax.grid(True)
                                ax.legend()
                                
                                st.pyplot(fig)
                            
                            # Contributions aux axes
                            st.subheader("Contributions aux axes")
                            
                            # Contributions des modalités de la variable en lignes
                            row_contrib = (ca.row_coordinates(cont_table)**2).div(ca.eigenvalues_, axis=1)
                            row_contrib = row_contrib.div(row_contrib.sum(axis=0), axis=1)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"{var1} - Contributions aux axes :")
                                st.dataframe(row_contrib.style.background_gradient(cmap='Blues', axis=0).format('{:.1%}'))
                            
                            # Contributions des modalités de la variable en colonnes
                            col_contrib = (ca.column_coordinates(cont_table)**2).div(ca.eigenvalues_, axis=1)
                            col_contrib = col_contrib.div(col_contrib.sum(axis=0), axis=1)
                            
                            with col2:
                                st.write(f"{var2} - Contributions aux axes :")
                                st.dataframe(col_contrib.style.background_gradient(cmap='Reds', axis=0).format('{:.1%}'))
                            
                            # Interprétation automatique
                            st.subheader("Interprétation automatique")
                            
                            if p < 0.05:
                                # Identification des contributions majeures
                                row_major_contrib = row_contrib.idxmax(axis=0).iloc[0]
                                col_major_contrib = col_contrib.idxmax(axis=0).iloc[0]
                                
                                # Calcul des profils relatifs
                                row_profiles = cont_table.div(cont_table.sum(axis=1), axis=0)
                                
                                st.markdown(f"""
                                ### Association significative :
                                
                                Le test du chi² (p = {p:.4f}) confirme une relation statistiquement significative entre {var1} et {var2}.
                                
                                ### Analyse des associations :
                                
                                - La modalité **{row_major_contrib}** de {var1} contribue le plus à l'axe F1 ({row_contrib.loc[row_major_contrib, 0]:.1%})
                                - La modalité **{col_major_contrib}** de {var2} contribue le plus à l'axe F1 ({col_contrib.loc[col_major_contrib, 0]:.1%})
                                
                                ### Caractérisation des profils :
                                """)
                                
                                # Affichage des profils caractéristiques pour chaque modalité de var1
                                for modality in row_profiles.index:
                                    top_assoc = row_profiles.loc[modality].sort_values(ascending=False).head(2)
                                    st.markdown(f"""Ce graphique est une analyse des correspondances (AC) montrant la relation entre des espèces (en bleu) et des maladies (en rouge) selon une dimension principale (F1), qui explique 100 % de l'inertie des données. Chaque point représente une espèce ou une catégorie de maladie, leur proximité indiquant une association. Par exemple, le "chien" est associé aux "maladies cardiaques" et "infectieuses," tandis que le "chat" est lié aux "maladies endocriniennes." Le terme "sain" est positionné près du centre, indiquant une absence marquée de liens avec des maladies spécifiques.""")
                            else:
                                st.write("Pas d'association significative détectée. Les variables semblent indépendantes.")
                            
                        except Exception as e:
                            st.error(f"Erreur lors de l'analyse des correspondances : {e}")
                            st.info("Assurez-vous que la bibliothèque prince est installée et que les données sont appropriées pour l'analyse.")
            else:
                st.warning("Au moins deux variables catégorielles sont nécessaires pour l'Analyse des Correspondances.")
    else:
        st.error("Impossible d'effectuer l'analyse multivariée. Aucun dataset n'est chargé.")

# ====================== PROFILS TYPES ======================
elif page == "Profils types":
    st.title("👥 Profils types")
    
    if data_processed is not None:
        if 'Maladie' in data_processed.columns:
            # Sélection de la maladie
            diseases = sorted(data_processed['Maladie'].unique())
            selected_disease = st.selectbox("Choisir une maladie", diseases)
            
            # Filtrage des données
            disease_data = data_processed[data_processed['Maladie'] == selected_disease]
            non_disease_data = data_processed[data_processed['Maladie'] != selected_disease]
            
            st.header(f"Profil type pour la maladie : {selected_disease}")
            
            # Variables numériques
            num_vars = data_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'Classe_age' in num_vars:
                num_vars.remove('Classe_age')
            
            # Sélection des variables pour l'analyse
            selected_vars = st.multiselect("Choisir les variables à analyser", 
                                         num_vars, 
                                         default=num_vars[:5])
            
            if selected_vars:
                # Tableau comparatif
                st.subheader("Tableau comparatif")
                
                # Calcul des moyennes
                disease_means = disease_data[selected_vars].mean()
                overall_means = data_processed[selected_vars].mean()
                
                # Préparation des données pour le tableau
                comparison_data = pd.DataFrame({
                    'Variable': selected_vars,
                    f'Moyenne {selected_disease}': disease_means.values,
                    'Moyenne globale': overall_means.values,
                    'Différence (%)': ((disease_means / overall_means) - 1) * 100
                })
                
                # Affichage du tableau
                st.dataframe(comparison_data.style.highlight_max(axis=0, subset=['Différence (%)']))
                
                # Ajout de l'interprétation après le tableau comparatif
                if selected_disease == 'cancer':
                    st.subheader("Interprétation du profil cancer")
                    st.markdown("""
                    **Âge avancé (+4.8%)** : Confirme que le cancer est plus fréquent chez les animaux âgés.
                    
                    **Perte de poids marquée (-10.1%)** : Signature de la cachexie cancéreuse, un critère diagnostique important.
                    
                    **Température élevée (+1.35%)** : Suggère une inflammation systémique ou une infection secondaire.
                    
                    **Respiration accélérée (+7.3%)** : Peut indiquer des métastases pulmonaires ou une anémie.
                    
                    **Pulse légèrement plus rapide (+4%)** : Conséquence possible de la fièvre, de la douleur ou de l'anémie.
                    
                    **Conclusion** : Ces paramètres forment un tableau clinique cohérent avec un état cancéreux, notamment en phase active ou avancée.
                    """)
                
                # Variables catégorielles
                st.subheader("Distribution des variables catégorielles")
                
                cat_vars = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
                if 'Classe_age' in data_processed.columns:
                    cat_vars.append('Classe_age')
                
                if 'Maladie' in cat_vars:
                    cat_vars.remove('Maladie')
                
                if cat_vars:
                    selected_cat_var = st.selectbox("Choisir une variable catégorielle", cat_vars)
                    
                    # Calcul des proportions
                    disease_counts = disease_data[selected_cat_var].value_counts(normalize=True) * 100
                    overall_counts = data_processed[selected_cat_var].value_counts(normalize=True) * 100
                    
                    # Fusion des données
                    cat_comp = pd.DataFrame({
                        f"{selected_disease} (%)": disease_counts,
                        "Global (%)": overall_counts
                    }).fillna(0).reset_index().rename(columns={'index': selected_cat_var})
                    
                    # Graphique en barres groupées
                    fig = px.bar(cat_comp.melt(id_vars=selected_cat_var, var_name='Groupe', value_name='Pourcentage'),
                                x=selected_cat_var, y='Pourcentage', color='Groupe', barmode='group',
                                title=f"Distribution de {selected_cat_var} : {selected_disease} vs Global")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune variable catégorielle disponible.")
                
                # Caractéristiques distinctives
                st.subheader("Caractéristiques distinctives")
                
                # Variables numériques
                st.write("**Variables numériques distinctives :**")
                
                # Calcul des écarts relatifs
                distinctive_num = []
                for var in num_vars:
                    disease_mean = disease_data[var].mean()
                    overall_mean = data_processed[var].mean()
                    relative_diff = ((disease_mean / overall_mean) - 1) * 100
                    
                    distinctive_num.append({
                        'Variable': var,
                        'Moyenne maladie': disease_mean,
                        'Moyenne globale': overall_mean,
                        'Écart (%)': relative_diff
                    })
                
                distinctive_num_df = pd.DataFrame(distinctive_num)
                distinctive_num_df = distinctive_num_df.sort_values('Écart (%)', key=abs, ascending=False)
                
                st.dataframe(distinctive_num_df.head(5).style.format({
                    'Moyenne maladie': '{:.2f}',
                    'Moyenne globale': '{:.2f}',
                    'Écart (%)': '{:.1f}%'
                }).background_gradient(cmap='RdYlGn', subset=['Écart (%)']))
            else:
                st.warning("Veuillez sélectionner au moins une variable pour générer le profil.")
        else:
            st.warning("La variable 'Maladie' est nécessaire pour cette analyse.")
    else:
        st.error("Impossible d'afficher les profils types. Aucun dataset n'est chargé.")
# ====================== PROFILS TYPES ======================
elif page == "Prédiction (Bonus)":
    st.title("👥 Prediction et Recommendation")
    import streamlit as st
    import pandas as pd
    import joblib
    import json
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    from langchain_google_genai import ChatGoogleGenerativeAI
    import os

    # MUST BE FIRST - Fix for Windows compatibility
    import asyncio
    import sys
    if sys.platform == "win32":
        if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Initialize APIs
    os.environ["PINECONE_API_KEY"] = "8a12b614-e097-456f-8a7c-0b0d74aaa0e9"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCACvEqrxa5fDe7c09KLN2pUKt_BJorLME"

    pc = Pinecone(api_key="8a12b614-e097-456f-8a7c-0b0d74aaa0e9")
    index = pc.Index("projet")
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)



    # Charger les modèles
    @st.cache_resource
    def load_models():
        try:
            gbm_model = joblib.load('assets/gbm_model.pkl')
            scaler = joblib.load('assets/scaler.pkl')
            return gbm_model, scaler
        except Exception as e:
            st.error(f"Erreur de chargement des modèles: {str(e)}")
            return None, None

    gbm_model, scaler = load_models()

    # Mappings
    espece_mapping = {1: 'chien', 0: 'chat'}
    race_mapping = {
        0: "Abyssin", 1: "Bengal", 2: "Berger Allemand", 
        3: "Bouledogue Français", 4: "British Shorthair",
        5: "Dalmatien", 6: "Golden Retriever", 7: "Husky",
        8: "Maine Coon", 9: "Persan", 10: "Pinscher nain",
        11: "Siamois", 12: "Sphynx", 13: "Teckel",
        14: "Westie", 15: "Yorkshire"
    }
    intensite_activite_mapping = {0: 'Faible', 1: 'Moyenne', 2: 'Élevée'}
    maladie_mapping = {
        0: "cancer", 1: "maladie cardiaque", 
        2: "maladie endocriniennes", 3: "maladies infectieuses",
        4: "maladies respiratoires", 5: "sain"
    }

    def generate_advice(result):
        """Generate veterinary advice using Pinecone and Gemini"""
        question = (
            f"En tant que vétérinaire expert, donnez des conseils pour un {result['Espèce']} "
            f"de race {result['Race']} ({result['Âge']} ans) diagnostiqué avec: {result['Diagnostic']}. "
            f"Paramètres: Température={result['Température']}°C, Pouls={result['Pulse']} BPM, "
            f"Respiration={result['Respiration']}/min, Activité={result['Activité']}, "
            f"Sommeil={result['Sommeil']}/10. Fournissez: 1) Traitement recommandé 2) Conseils alimentaires "
            f"3) Plan de suivi 4) Prévention. Soyez précis et professionnel."
        )
        
        try:
            results = index.query(
                vector=embedding_model.encode(question).tolist(),
                top_k=5,
                include_metadata=True
            )
            texts = [result['metadata']['text'] for result in results['matches']]
            answer = llm.invoke(f"Répondez en français à cette question de conseil vétérinaire en vous basant sur ce contexte: {texts}\n\nQuestion: {question}")
            return answer.content
        except Exception as e:
            st.error(f"Erreur de génération de conseils: {str(e)}")
            return None

    def predict_disease(input_data):
        try:
            expected_features = [
                "Espèce", "Âge", "Poids", "Race", "Température", 
                "Respiration", "Pulse", "Intensité_activité", "Score_sommeil"
            ]
            sample = pd.DataFrame([input_data], columns=expected_features)
            
            if hasattr(scaler, 'feature_names_in_'):
                if not all(f in scaler.feature_names_in_ for f in expected_features):
                    st.error("Certaines caractéristiques ne correspondent pas au modèle!")
                    return None
                    
            sample_scaled = scaler.transform(sample)
            return gbm_model.predict(sample_scaled)[0]
        except Exception as e:
            st.error(f"Erreur de prédiction: {str(e)}")
            return None

    # Interface
    st.title("🐾 Diagnostic Vétérinaire IA")

    with st.form("animal_form"):
        st.header("Informations sur l'animal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            espece = st.radio("Espèce", options=["Chien", "Chat"])
            age = st.number_input("Âge (années)", min_value=0, max_value=30)
            poids = st.number_input("Poids (kg)", min_value=0.1, max_value=100.0)
            activite = st.select_slider("Intensité d'activité", options=["Faible", "Moyenne", "Élevée"])
            
        with col2:
            race = st.selectbox("Race", options=list(race_mapping.values()))
            temperature = st.number_input("Température (°C)", min_value=35.0, max_value=42.0)
            respiration = st.number_input("Respiration (resp/min)", min_value=5, max_value=100)
            sommeil = st.slider("Score de sommeil (1-10)", min_value=1, max_value=10, value=5)
        
        pulse = st.slider("Pulse (BPM)", min_value=40, max_value=200)
        submitted = st.form_submit_button("Diagnostiquer")

    if submitted and gbm_model is not None and scaler is not None:
        input_data = {
            "Espèce": 1 if espece == "Chien" else 0,
            "Âge": age,
            "Poids": poids,
            "Race": [k for k, v in race_mapping.items() if v == race][0],
            "Température": temperature,
            "Respiration": respiration,
            "Pulse": pulse,
            "Intensité_activité": [k for k, v in intensite_activite_mapping.items() if v == activite][0],
            "Score_sommeil": sommeil
        }
        
        with st.spinner("Analyse en cours..."):
            prediction = predict_disease(input_data)
            
            if prediction is not None:
                result = {
                    "Espèce": espece,
                    "Race": race,
                    "Âge": age,
                    "Diagnostic": maladie_mapping[prediction],
                    "Température": temperature,
                    "Pulse": pulse,
                    "Respiration": respiration,
                    "Activité": activite,
                    "Sommeil": sommeil
                }
                
                st.success("Analyse terminée !")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Résultats")
                    st.json(result)
                    
                with col2:
                    st.subheader("Diagnostic")
                    if result["Diagnostic"] == "sain":
                        st.success("✅ Animal en bonne santé")
                    else:
                        st.error(f"⚠️ Diagnostic: {result['Diagnostic'].upper()}")
                
                with st.spinner("Génération des conseils..."):
                    advice = generate_advice(result)
                    if advice:
                        st.subheader("💡 Conseils vétérinaires")
                        st.markdown(advice)
                    else:
                        st.warning("Impossible de générer des conseils personnalisés")

    st.sidebar.info("""
    Ce système fournit un diagnostic préliminaire.
    Consultez toujours un vétérinaire pour un examen complet.
    """)
# Pied de page
st.markdown("""
---
Développé avec ❤️ en Python et Streamlit | © 2025
""") 