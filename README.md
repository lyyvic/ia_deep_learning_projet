# Detection et classification d'objets spatiaux

## Structure du rendu e-campus

Le dossier `groupe_Parriault_Zory_Mangin_Lapointe/` est organisé pour la remise :

- `README.md` : rapport du projet ;
- `CONTRIBUTORS.md` : membres et responsabilités ;
- `CHOIX_MODELES_ET_HYPERPARAMETRES.md` : justification des expérimentations ;
- `ANALYSE_CONFORMITE.md` : vérification des critères du sujet ;
- `src/` : notebooks, données, modèles, utilitaires et dépendances ;
- `sujet_projet/` : documents fournis avec le sujet.

Les fichiers du dossier racine du dépôt sont conservés afin de ne pas perdre l’historique de travail.

Projet de deep learning realise avec TensorFlow dans le cadre du cours de 5e annee. Le domaine retenu est celui des **images** : le projet distingue les satellites des debris spatiaux, puis estime une boite englobante autour de l'objet.

## 1. Sujet, problematique et objectifs

### Problematique

Les images astronomiques ou spatiales peuvent contenir des satellites operationnels, des vehicules spatiaux identifies ou des debris. Une surveillance automatisee doit pouvoir signaler rapidement un debris tout en conservant une faible probabilite de fausse alerte.

L'objectif du projet est de contribuer au nettoyage de l'environnement spatial : distinguer un objet encore identifiable comme un satellite d'un debris susceptible de présenter un risque, puis fournir une localisation approximative de l'objet. Dans ce contexte, une alerte doit être lisible et rapide.

Le jeu source contient initialement 11 classes detaillees, mais la mission est volontairement simplifiée en deux classes :

- `satellite` : les 10 classes autres que `debris` ;
- `debris` : les objets etiquetes `debris`.

Ce regroupement transforme une classification multi-classe initiale en un **probleme binaire** adapte au tri satellite/debris. Nous ne cherchons pas à reconnaitre le nom precis d'une mission : ce travail suppose que nous ne sommes pas specialistes des différents types de satellites et qu'il n'est pas necessaire pour la mission de verifier individuellement l'identite technique de chaque modèle. Identifier automatiquement un debris est donc deja utile pour orienter l'alerte et une verification humaine.

### Objectifs

1. Construire un pipeline TensorFlow/Keras reproductible a partir d'annotations YOLO.
2. Classifier une image en `satellite` ou `debris`.
3. Localiser l'objet avec une boite englobante normalisee.
4. Comparer plusieurs strategies de taux d'apprentissage.
5. Evaluer les modeles sur les jeux de validation et de test, sans modifier les parametres apres mesure finale.
6. Fournir une analyse qualitative des erreurs, les limites du systeme et les pistes d'amelioration.

## 2. Donnees

### Source et licence

- Dataset : **Space debris - v2**, projet Roboflow **space-debris-mugw2**, version 2.
- URL : <https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2>
- Licence : **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- Export : 2467 images au format JPEG.
- Format d'annotation : YOLOv8, avec une ligne par objet :

```text
class_id x_center y_center width height
```

Les 11 classes originales sont : `cheops`, `debris`, `double_start`, `earth_observation_sat_1`, `lisa_pathfinder`, `proba_2`, `proba_3_csc`, `proba_3_ocs`, `smart_1`, `soho` et `xmm_newton`.

### Obtention

Telecharger l'export depuis la page Roboflow ci-dessus, ou utiliser le lien donne dans `data/data.yaml`, puis placer les repertoires `train`, `valid` et `test` dans `data/`. Les fichiers `data/README.dataset.txt` et `data/README.roboflow.txt` donnent les informations d'export fournies par Roboflow.

### Taille des ensembles

| Ensemble | Images | Satellites | Debris |
|---|---:|---:|---:|
| Train | 2105 | 1919 | 186 |
| Validation | 239 | 218 | 21 |
| Test | 123 | 112 | 11 |

Le jeu d'entrainement est fortement desequilibre : environ 91 % d'images satellites et 9 % de debris. La proportion n'est pas modifiee sur validation et test afin de mesurer les performances sur une distribution proche des donnees originales.

## 3. Pretraitement

### Annotations et decoupage

Les ensembles `train`, `valid` et `test` sont deja separes dans les donnees. Pour chaque image, le premier fichier d'annotation YOLO fournit son identifiant de classe et sa boite. Les classes originales sont converties en identifiant binaire :

- `0` pour `satellite` ;
- `1` pour `debris`.

Les fichiers sont associes a leur image via `create_dataset_dataframe()`. Les annotations fournies par Roboflow sont normalisees par rapport a la largeur et a la hauteur.

### Conversion des boites

Les coordonnees YOLO sont converties dans l'ordre `[ymin, xmin, ymax, xmax]`, utilise par TensorFlow/Keras :

```text
xmin = x_center - width / 2
ymin = y_center - height / 2
xmax = x_center + width / 2
ymax = y_center + height / 2
```

Chaque exemple contient une seule boite, aussi aucune logique de padding multi-boites n'est necessaire.

### Pipeline `tf.data`

Les deux notebooks utilisent :

- `tf.io.read_file` puis `tf.image.decode_jpeg` ;
- redimensionnement des images en 224 x 224 ;
- `keras.applications.mobilenet_v2.preprocess_input`, c'est-a-dire une conversion adaptee au MobileNetV2 pre-entraine ;
- `shuffle` seulement pour l'entraînement ;
- lots de taille 16 ;
- `prefetch(tf.data.AUTOTUNE)` ;
- lecture/traitement parallele avec `num_parallel_calls=tf.data.AUTOTUNE`.

### Augmentation

Pendant l'entrainement de classification, une augmentation aleatoire est appliquee :

- retournement horizontal et vertical ;
- rotation aleatoire, avec remplissage par reflexion pour eviter les coins noirs.

Roboflow applique egalement lors de l'export une orientation EXIF automatique, un redimensionnement a 640 x 640 avec deformation, ainsi que trois versions de chaque source avec 50 % de retournement horizontal et vertical.

### Compensation du desequilibre

Le notebook utilise des poids de classe calcules sur l'ensemble d'entrainement :

```text
poids(classe) = nombre total / (nombre de classes x nombre d'exemples de la classe)
```

Cette methode conserve toutes les donnees originales tout en donnant un poids comparable aux deux classes dans la perte.

## 4. Architectures des modeles

### Classification binaire

Le modele principal utilise le transfert d'apprentissage avec MobileNetV2 pre-entraine sur ImageNet :

1. MobileNetV2 sans tete de classification (`include_top=False`) ;
2. extraction globale par `GlobalAveragePooling2D` ;
3. `Dropout(0.3)` ;
4. couche dense finale a une sortie ;
5. activation `sigmoid` donnant la probabilite du debris.

MobileNetV2 est adaptée aux images modestes et permet une extraction convolutionnelle déjà validée sur un grand corpus. Le backbone est fine-tuné : ses poids sont rendus entraînables dans les phases d'entraînement documentées.

Perte : `BinaryCrossentropy`.

Metriques :

- accuracy ;
- precision sur les debris ;
- rappel sur les debris ;
- PR-AUC ;
- matrice de confusion.

### Detection de la boite

Le second notebook part d'un poids MobileNetV2 de classification puis ajoute une sortie multi-taches :

- `class_output` : classification satellite/debris, avec sigmoide ;
- `bbox_output` : quatre coordonnees `[ymin, xmin, ymax, xmax]`, avec sigmoide.

Le modele est compile avec :

- `BinaryCrossentropy` pour la classe ;
- `Huber` pour les boites ;
- poids de perte classe/boite : 1/5 ;
- Adam ;
- IoU personnalisee ;
- accuracy de boite : proportion d'IoU >= 0.5 ;
- MAE des quatre coordonnees.

## 5. Entrainement

### Configuration commune

- Taille d'image : 224 x 224 ;
- taille de lot : 16 ;
- taux d'apprentissage initial : `1e-3` ;
- optimiseur : Adam ;
- EarlyStopping sur PR-AUC pour la classification ;
- sauvegarde des meilleurs poids selon la PR-AUC ou l'IoU ;
- GPU utilise automatiquement lorsqu'il est disponible, sinon CPU ;
- suivi experimentale avec Weights & Biases.

### Classification

Le notebook realise jusqu'a 70 epoques sur le jeu d'entrainement original, avec poids de classe. Le `ModelCheckpoint` conserve les meilleurs poids selon `val_pr_auc`. Le `EarlyStopping` surveille la meme metrique avec une patience de 65 epoques. Les poids sont restaures a la meilleure epoch.

Trois strategies ont ete implementees :

| Strategie | Comportement |
|---|---|
| `hardcoded` | taux d'apprentissage fixe a `1e-3` |
| `cosine_decay` | decroissance cosinus jusqu'a 1 % du taux initial |
| `plateau` | division du taux par deux lorsque la validation stagne, avec un minimum de `1e-6` |

Ces variantes sont selectionnees avec la variable `LEARNING_RATE_MODE` ou la variable d'environnement du notebook.

### Boites englobantes

Le notebook reprend le modele de classification `src/models/Class_Detection/satellite_debris_hard_70_retrain_v4.keras`, degel le backbone, puis ajoute une sortie boite. L'entrainement dure jusqu'a 60 epoques. Le modele est selectionne selon `val_bbox_output_iou`, avec EarlyStopping sur la meme metrique.

Le modele final de localisation est `src/models/BBOX/satellite_debris_bbox_lronplateau.keras`.

## 6. Resultats sauvegardes

### Classification sur test

L'evaluation finale sauvegardee dans `results/final/Classification/classification_final.json`, avec un seuil de decision fixe a 0.5, donne :

| Classe | Precision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| Satellite | 0.9825 | 1.0000 | 0.9912 | 112 |
| Debris | 1.0000 | 0.8182 | 0.9000 | 11 |

Autres mesures :

- accuracy globale : 0.9837 ;
- 9 debris sur 11 sont detectes au seuil 0.5 ;
- aucun satellite n'est confondu avec un debris dans cette sortie sauvegardee ;
- PR-AUC debris : 0.9202 ;
- 2 debris sur 11 ne sont pas detectes.

Ces chiffres sont a lire avec prudence : le support debris du test ne represente que 11 images.

### Localisation sur test

L'evaluation finale sauvegardee dans `results/final/BBOX/bbox_final.json` donne :

- accuracy de classe : 97.56 % ;
- IoU moyenne : 0.6604 (mediane : 0.6945) ;
- accuracy des boites : 87.80 % pour IoU >= 0.5 ;
- detection complete (classe correcte et IoU >= 0.5) : 87.80 %.

L'IoU moyenne est de 0.3566 pour les debris et de 0.6693 pour les satellites. Les resultats des comparaisons exploratoires de variantes sur le test sont descriptifs ; la selection des modeles repose sur les metriques de validation.

## 7. Seuil de decision

Le modele produit une probabilite de debris. Le seuil par defaut est 0.5. Le notebook mesure plusieurs seuils sur validation seulement : 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 et 0.9.

Cette separation est importante :

- la validation permet d'etudier les compromis precision/rappel ;
- le test ne sert qu'a l'evaluation finale ;
- un seuil ne doit pas etre optimise sur un test contenant seulement 11 debris.

Selon l'application, un seuil plus bas peut etre prefere pour maximiser le rappel de debris, au prix d'un nombre plus important de fausses alertes.

## 8. Analyse des erreurs

### Classification

Les erreurs principales a attendre sont :

- faux negatifs sur les debris peu contrastes, petits ou partiellement caches ;
- petites images ou recadrages qui empechent MobileNetV2 d'identifier les details ;
- similarites visuelles entre certaines classes satellites et les debris ;
- biais induit par les retransmissions, le redimensionnement et les augmentations de Roboflow ;
- instabilite statistique due au faible nombre de debris sur validation et test.

### Boites

Les observations disponibles dans le notebook montrent que l'IoU peut rester moyenne meme lorsque la classe est correcte. Les cas difficiles concernent notamment les objets proches des bords, les petites boites, les objets allonges et les boites dont les limites sont ambiguës.

## 9. Points d'attention du système

- Le jeu test contient 11 debris : ces observations permettent d'obtenir une première mesure, mais les pourcentages doivent être interpretes avec prudence.
- Les annotations privilégient un objet principal par image. Le modèle apprend donc une détection simplifiée de cet objet et ne vise pas encore plusieurs debris présents simultanément.
- Le redimensionnement en 224 x 224 facilite l'apprentissage avec MobileNetV2, mais peut atténuer les details des objets tres petits.
- Les boites proches des bords, les objets allongés et les limites visuellement ambiguës sont les cas où la localisation est la plus delicate.
- Les résultats peuvent être sensibles aux retransmissions, aux recadrages et aux augmentations appliquées aux images.
- Les poids MobileNetV2 peuvent nécessiter un premier téléchargement depuis ImageNet.
- Comme pour toute alerte automatique, une probabilité proche du seuil 0.5 ou une boite de petite surface doit pouvoir déclencher une revue humaine avant toute decision opérationnelle.

## 10. Ameliorations possibles

1. Ajouter une validation croisee afin de reduire l'incertitude sur les predictions de debris.
2. Equilibrer ou sur-echantillonner le debris avec une strategie documentee.
3. Ajouter des augmentations de luminosite, contraste, bruit et defaut de compression.
4. Tester EfficientNet ou un transfert d'apprentissage deux phases.
5. Degeler progressivement le backbone et comparer plusieurs taux d'apprentissage.
6. Consolider les resultats des variantes dans un tableau `CSV/JSON` produit automatiquement.
7. Approfondir la comparaison des courbes, matrices de confusion et exemples d'erreurs entre les configurations.
8. Evaluer au niveau du pixel ou de l'objet avec un vrai detecteur multi-boites.
9. Ajouter des tests unitaires sur la conversion YOLO et les dimensions des boites.
10. Ajouter une cellule ou application qui charge une nouvelle image locale et affiche classe, probabilite, boite et score IoU lorsqu'une verite terrain est disponible.

## 11. IA symbolique (bonus)

Une composante symbolique est implémentée dans [src/UTILS/symbolic_rules.py](src/UTILS/symbolic_rules.py). Elle prend la probabilité de debris et la boîte prédite, puis applique les règles suivantes :

- vérification que les quatre coordonnées de la boîte sont dans `[0,1]` ;
- vérification que `ymax >= ymin` et `xmax >= xmin` ;
- vérification que la probabilité est dans `[0,1]` ;
- cohérence entre la probabilité, le seuil de décision et la classe prédite ;
- supervision humaine lorsque la probabilité est dans une marge de ±0.10 autour du seuil ;
- supervision humaine lorsqu'un debris est associé à une boîte couvrant moins de 1 % de l'image.

La fonction `apply_symbolic_rules()` retourne le statut `confiant`, `incertain` ou `sortie_invalide`, la classe prédite, la surface normalisée de la boîte, les motivations humaines lisibles et l'indicateur `requires_human_review`. Les règles ne modifient ni les poids, ni la sortie du réseau, ni le seuil utilisé pendant l'entraînement. Elles ajoutent une interprétation explicable aux prédictions du notebook de boîtes.

## 12. Demonstration

Le notebook [src/demo.ipynb](src/demo.ipynb) choisit aleatoirement trois satellites et trois debris dans le jeu de validation. Le modele `satellite_debris_bbox_lronplateau.keras` predit leur classe binaire et leur boite. La boite reelle est tracee en blanc pointille ; la boite predite est verte si l'IoU est au moins 0.5, jaune entre 0.3 et 0.5, et rouge en dessous de 0.3. Le titre devient rouge si la classe predite differe de la classe reelle, et une legende explique les boites.

```bash
jupyter lab src/demo.ipynb
```

La demonstration utilise des images annotees de validation ; elle ne constitue pas encore une interface d'inference pour des images arbitraires fournies par l'utilisateur.

## 13. Reproduction

### Prerequis

- Python 3.10 ou 3.11 recommande ;
- TensorFlow, Keras, pandas, numpy, matplotlib, scikit-learn, PyYAML ;
- JupyterLab ;
- optionnellement CUDA pour un entrainement GPU.

Le fichier [requirements.txt](requirements.txt) contient les dependances principales.

### Installation

Depuis la racine du depot :

```bash
./src/UTILS/setup.sh
source .venv/bin/activate
jupyter lab
```

Le script se trouve dans `src/UTILS/setup.sh` et s'execute sous Linux ou WSL.

### Execution

```bash
PYTHONPATH=src jupyter lab src/Class_detection_training.ipynb
PYTHONPATH=src jupyter lab src/bounding_box_training.ipynb
```

Exécuter les notebooks depuis la racine du depot. Les chemins `data/`, `src/models/` et `src/UTILS/` sont relatifs a cette racine.

Les modes d'entrainement peuvent etre lancees avec :

```bash
PYTHONPATH=src LEARNING_RATE_MODE=hardcoded jupyter lab src/Class_detection_training.ipynb
PYTHONPATH=src LEARNING_RATE_MODE=plateau jupyter lab src/Class_detection_training.ipynb
PYTHONPATH=src LEARNING_RATE_MODE=cosine_decay jupyter lab src/Class_detection_training.ipynb
```

Les poids sont sauvegardes sous `src/models/Class_Detection/` et `src/models/BBOX/`.

## 14. Structure du depot

```text
.
├── README.md
├── CONTRIBUTORS.md
├── CHOIX_MODELES_ET_HYPERPARAMETRES.md
├── requirements.txt
├── data/
├── results/final/
└── src/
    ├── Class_detection_training.ipynb
    ├── bounding_box_training.ipynb
    ├── demo.ipynb
    ├── models/
    └── UTILS/
        ├── setup.sh
        └── symbolic_rules.py
```

## 15. Sources utilisees

- TensorFlow/Keras : documentation officielle des API `tf.data`, `keras.applications.MobileNetV2`, callbacks et metriques.
- Dataset Roboflow : <https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2>
- MobileNetV2 : architecture utilisee via les poids Keras `imagenet`.
- Scikit-learn : rapports de classification, precision, rappel, F1, PR-AUC et matrice de confusion.
- Weights & Biases : suivi optionnel des essais et courbes d'entrainement.
- Roboflow : export YOLOv8, orientation, redimensionnement et augmentations.
- Ressources du cours et notebooks d'exemple fournis dans `sujet_projet/`.
- Assistance IA : analyse du depot, aide a la redaction et realisation de la demonstration. Les experiences et les resultats sont sous la responsabilite des quatre membres du groupe.

## 16. Contribution des membres

Le projet est réalisé par quatre membres :

- Lyse Parriault ;
- Léo Zory ;
- Thomas Mangin ;
- Matthias Lapointe.

La répartition détaillée est présentée dans [CONTRIBUTORS.md](CONTRIBUTORS.md) et couvre les données, le prétraitement, les modèles, l'évaluation, la documentation, la démonstration et l'IA symbolique.


Les configurations et les hyperparamètres des modèles sont détaillés dans [CHOIX_MODELES_ET_HYPERPARAMETRES.md](CHOIX_MODELES_ET_HYPERPARAMETRES.md).

Les notebooks d'entraînement produisent notamment :

- classification : rapport de test, matrice de confusion, courbes et exemples de faux positifs/faux négatifs ;
- boîtes : matrice de confusion, distribution des IoU, courbes et exemples de détections ou boîtes erronées ;
- tous les fichiers produits par ces sections sont dirigés vers `results/final/`.
