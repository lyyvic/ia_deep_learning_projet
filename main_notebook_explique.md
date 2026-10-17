# Explication détaillée de `main.ipynb`

## 1. Vue d'ensemble du notebook

Le notebook construit et évalue un système de reconnaissance d'objets spatiaux. Son objectif principal est de distinguer deux catégories :

- `satellite` : toutes les classes originales sauf `debris` ;
- `debris` : les débris spatiaux.

Le fichier de données `data/data.yaml` contient à l'origine 11 classes. Le notebook transforme donc la classification multi-classe originale en classification binaire. Les annotations sont au format YOLO et décrivent une boîte par image :

```text
class_id x_center y_center width height
```

Les coordonnées YOLO sont normalisées entre `0` et `1`, avec un centre et des dimensions. Elles sont converties dans l'ordre utilisé par le pipeline :

```text
[ymin, xmin, ymax, xmax]
```

Le notebook contient deux usages des données :

1. un pipeline multi-tâches préparant une classe et une boîte englobante ;
2. un pipeline de classification utilisant le jeu équilibré `train_50-50`.

> **Point important :** les cellules 12 et 14 préparent et visualisent simultanément la classe et la boîte englobante, mais la partie entraînement visible dans les cellules 18 à 28 recharge un modèle de **classification binaire**. Le modèle effectivement continué dans la cellule 24 ne reçoit donc que des images et des classes, pas les boîtes.

---

## 2. Utilité de chaque cellule

### Cellule 1 — Titre

Cette cellule Markdown donne le sujet général : détection et classification d'objets spatiaux et de débris.

### Cellule 2 — Description du problème

Elle précise que chaque image contient un seul objet et que le pipeline doit produire deux sorties :

- `class_output`, l'identifiant de classe ;
- `bbox_output`, les quatre coordonnées de la boîte.

Elle documente donc le contrat attendu par le pipeline TensorFlow multi-sorties.

### Cellule 3 — Imports et configuration globale

Cette cellule importe les bibliothèques, fixe la taille des images et la taille des lots, lit `data/data.yaml`, récupère les noms des classes originales et définit les deux classes finales du modèle.

Elle affiche aussi la version TensorFlow et les classes utilisées, ce qui constitue une vérification rapide de l'environnement.

### Cellule 4 — Cellule vide

Cette cellule ne contient aucun code ni texte. Elle sert uniquement de séparation visuelle et n'a aucun effet sur l'exécution.

### Cellule 5 — Introduction à la préparation des DataFrames

Elle annonce la lecture des annotations YOLO et leur conversion en coordonnées de boîte `[ymin, xmin, ymax, xmax]`.

Elle rappelle également que chaque image contient un objet unique : il n'est donc pas nécessaire de compléter plusieurs boîtes avec un mécanisme de padding comme `MAX_BOXES`.

### Cellule 6 — Création des DataFrames

Cette cellule définit `create_dataset_dataframe`, puis l'utilise pour construire les DataFrames d'entraînement, de validation et de test.

Pour chaque image, elle lit la première ligne du fichier d'annotation, convertit la boîte YOLO, regroupe les classes en deux classes binaires et affiche un résumé du jeu d'entraînement.

### Cellule 7 — Affichage de `df_train`

L'expression `df_train` est évaluée par le notebook afin d'afficher le DataFrame complet ou son aperçu interactif. Elle ne transforme pas les données.

### Cellule 8 — Début de l'EDA

Cette cellule introduit l'exploration des données, plus précisément l'étude de la distribution des classes d'entraînement.

### Cellule 9 — Distribution des classes

Elle affiche le nombre d'images `satellite` et `debris` dans `df_train`. Cette mesure permet de constater le déséquilibre entre les deux classes.

### Cellule 10 — Justification de l'équilibrage

Cette cellule explique que les satellites sont beaucoup plus nombreux que les débris. Elle motive donc la création préalable d'un dossier `train_50-50`, dans lequel des rotations d'images de débris ont été utilisées pour obtenir une distribution plus équilibrée.

La cellule documente la stratégie, mais ne réalise pas elle-même les rotations.

### Cellule 11 — Présentation du pipeline `tf.data`

Elle décrit la conversion des DataFrames en objets TensorFlow efficaces : lecture parallèle, redimensionnement, normalisation et dictionnaire de sorties.

### Cellule 12 — Construction des datasets TensorFlow multi-sorties

La fonction `df_to_tf_dataset` transforme les chemins, classes et boîtes du DataFrame en `tf.data.Dataset`.

Elle lit et décode chaque JPEG, redimensionne l'image, normalise ses pixels, mélange uniquement le jeu d'entraînement, regroupe les exemples en lots et précharge les lots suivants.

### Cellule 13 — Introduction à la visualisation

Elle annonce l'affichage d'un lot d'entraînement avec sa classe et sa boîte englobante afin de vérifier visuellement la préparation des annotations.

### Cellule 14 — Vérification visuelle des boîtes

Cette cellule prend un lot, convertit les coordonnées normalisées en pixels, dessine les rectangles verts et affiche le nom de la classe.

Elle permet de détecter rapidement une erreur de conversion, une boîte hors image ou une classe mal associée.

### Cellule 15 — Titre de la conversion HSV

Cette cellule annonce une piste d'analyse des images dans l'espace colorimétrique HSV.

### Cellule 16 — Expérimentation HSV désactivée

Toutes les lignes de cette cellule sont commentées. Elles ne sont donc pas exécutées.

Le code prévu aurait chargé quelques images, converti les pixels RGB en HSV, filtré les pixels selon des plages de teinte, saturation et luminosité, puis comparé les images RGB et HSV filtrées.

### Cellule 17 — Introduction au modèle de classification

Elle annonce la construction d'un réseau qui prédit une classe unique parmi `satellite` et `debris`.

### Cellule 18 — Jeu de classification et rechargement du modèle

Cette cellule importe Keras, définit un pipeline de classification sans boîte englobante, lit le jeu équilibré `train_50-50`, puis recharge `models/satellite_debris_best.keras`.

Le modèle rechargé est prêt à être évalué ou continué dans les cellules suivantes.

### Cellule 19 — Introduction au callback d'aperçu

Elle annonce un callback qui pourrait afficher un exemple de validation par classe après chaque époque.

### Cellule 20 — Définition de `EpochPreviewCallback`

La classe mémorise des exemples de validation, compte les prédictions par classe à chaque époque et affiche une image avec sa classe réelle et sa classe prédite.

Elle contient aussi une intégration W&B conditionnelle pour enregistrer les figures.

> Dans le notebook actuel, la classe est définie mais n'est pas ajoutée à la liste `callbacks` de la cellule 24. La cellule 24 utilise uniquement `ModelCheckpoint` et `EarlyStopping`.

### Cellule 21 — Présentation de Weights & Biases

Cette cellule explique le rôle de W&B : suivre les pertes, métriques et expériences.

### Cellule 22 — Import W&B

Elle importe `wandb` et `WandbMetricsLogger`. L'import du logger est préparé mais ce logger n'est pas utilisé dans les callbacks de la cellule 24.

### Cellule 23 — Introduction à l'entraînement

Elle décrit la poursuite de l'entraînement sur le jeu équilibré et le suivi attendu des métriques.

### Cellule 24 — Poursuite de l'entraînement

Cette cellule recharge le modèle, le compile avec Adam et une entropie croisée sparse, entraîne pendant au plus cinq époques, conserve les meilleurs poids selon `val_accuracy`, arrête tôt si `val_loss` ne progresse plus, puis sauvegarde le modèle.

### Cellule 25 — Introduction aux courbes

Elle annonce la comparaison des courbes de perte et de précision entre entraînement et validation.

### Cellule 26 — Courbes d'apprentissage

Elle lit l'historique Keras et produit deux graphiques : la perte de classification et l'accuracy pour les jeux d'entraînement et de validation.

### Cellule 27 — Introduction à l'évaluation

Elle précise que le test est réservé à la mesure finale et ne doit pas servir à modifier le modèle.

### Cellule 28 — Évaluation finale

Elle calcule les métriques sur `test_classification_ds`, puis les affiche avec quatre chiffres après la virgule.

---

## 3. Explication ligne par ligne

Les extraits ci-dessous reprennent les instructions exécutables du notebook. Les lignes sont commentées individuellement lorsque cela apporte une information utile ; les lignes vides servent uniquement à la lisibilité.

### Cellule 1

```markdown
# Détecteur et Classifieur d'Objets Spatiaux & Débris
Detecteur de Déchets
```

- `# ...` crée le titre principal du notebook.
- `Detecteur de Déchets` ajoute un sous-titre descriptif.

### Cellule 2

```markdown
Ce notebook prépare le pipeline complet de données avec **Pandas** et **TensorFlow**.
Chaque image contient un **unique objet** (satellite ou débris). Le pipeline génère ainsi deux cibles pour chaque image :
- `class_output` : l'identifiant de la classe (parmi 11 classes)
- `bbox_output` : les 4 coordonnées normalisées `[ymin, xmin, ymax, xmax]`
```

- La première ligne identifie Pandas et TensorFlow comme bibliothèques principales de préparation.
- La deuxième ligne pose l'hypothèse d'un seul objet par image.
- La troisième ligne annonce deux sorties par exemple.
- `class_output` représente la cible de classification.
- `bbox_output` représente la cible de localisation.

### Cellule 3

```python
import glob
```

Importe `glob`, utilisé pour rechercher les images correspondant à un motif de chemin.

```python
import os
```

Importe les fonctions de manipulation de chemins et de vérification d'existence de fichiers.

```python
import numpy as np
```

Importe NumPy sous le nom `np`, notamment utilisé pour les tableaux, les types numériques et les compteurs.

```python
import pandas as pd
```

Importe Pandas sous le nom `pd`, utilisé pour construire et manipuler les DataFrames.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```

Importe Matplotlib pour afficher les images et `patches` pour dessiner les rectangles des boîtes.

```python
import tensorflow as tf
```

Importe TensorFlow, utilisé pour les datasets, le prétraitement et l'entraînement.

```python
import yaml
```

Importe le lecteur YAML servant à analyser `data/data.yaml`.

```python
IMG_SIZE = 224
BATCH_SIZE = 16
```

- `IMG_SIZE` impose une image carrée de 224 pixels de côté.
- `BATCH_SIZE` fixe le nombre d'images traité simultanément.

```python
DATA_YAML_PATH = 'data/data.yaml'
```

Stocke le chemin relatif de la configuration du dataset.

```python
with open(DATA_YAML_PATH, 'r', encoding='utf-8') as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)
```

- `open` ouvre le fichier en lecture avec l'encodage UTF-8.
- `with` ferme automatiquement le fichier.
- `yaml.safe_load` convertit le YAML en objets Python sûrs, ici un dictionnaire.

```python
ORIGINAL_CLASS_NAMES = dataset_config['names']
```

Récupère les 11 noms de classes déclarés dans le YAML. L'index de chaque nom correspond à son identifiant YOLO.

```python
CLASS_NAMES = ['satellite', 'debris']
NUM_CLASSES = len(CLASS_NAMES)
```

- `CLASS_NAMES` définit les deux classes finales.
- `NUM_CLASSES` calcule leur nombre, soit `2`.

```python
print(f"TensorFlow version: {tf.__version__}")
print(f"Classes originales : {ORIGINAL_CLASS_NAMES}")
print(f"Classes du modèle : {CLASS_NAMES}")
```

Ces trois lignes affichent la version TensorFlow, les classes originales et les classes regroupées. Elles servent de contrôle de configuration.

### Cellules 4 et 5

La cellule 4 est vide.

Dans la cellule 5 :

- `## 1. Préparation des DataFrames Pandas` crée le titre de section ;
- le paragraphe explique la conversion du format YOLO ;
- la dernière phrase justifie l'absence de padding puisqu'une seule boîte est présente par image.

### Cellule 6 — fonction `create_dataset_dataframe`

```python
def create_dataset_dataframe(image_dir):
```

Définit une fonction qui reçoit le répertoire des images et retourne un DataFrame d'annotations.

```python
    """
    Lit les annotations YOLO et crée un DataFrame avec une cible binaire.
    Tous les objets sauf debris sont regroupés dans la classe satellite.
    """
```

La docstring décrit la lecture YOLO et le regroupement binaire : seul le nom exact `debris` conserve sa propre classe.

```python
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
```

- `os.path.join` construit le motif `image_dir/*.jpg`.
- `glob.glob` récupère les chemins JPEG.
- `sorted` garantit un ordre stable.

```python
    records = []
```

Crée la liste qui contiendra un dictionnaire par image annotée.

```python
    for img_path in image_paths:
```

Parcourt chaque image trouvée.

```python
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
```

Construit le chemin de l'annotation en remplaçant le dossier `images` par `labels` et l'extension JPEG par TXT.

```python
        if os.path.exists(label_path):
```

Ignore les images dont le fichier d'annotation n'existe pas.

```python
            with open(label_path, 'r') as label_file:
                line = label_file.readline().strip()
```

Ouvre l'annotation, lit uniquement sa première ligne et retire les espaces de début et de fin. Cette lecture correspond à l'hypothèse d'un objet unique.

```python
                if line:
                    parts = line.split()
```

Vérifie que la ligne n'est pas vide, puis sépare ses champs selon les espaces.

```python
                    cls_id = int(parts[0])
                    x_c, y_c, box_width, box_height = map(float, parts[1:5])
```

- `cls_id` reçoit l'identifiant entier de la classe YOLO.
- Les quatre valeurs suivantes deviennent le centre et les dimensions normalisés de la boîte.

```python
                    original_class_name = (
                        ORIGINAL_CLASS_NAMES[cls_id]
                        if 0 <= cls_id < len(ORIGINAL_CLASS_NAMES)
                        else 'unknown'
                    )
```

Convertit l'identifiant en nom de classe. Le test d'intervalle évite un accès hors limites et utilise `unknown` pour un identifiant invalide.

```python
                    ymin = max(0.0, y_c - (box_height / 2.0))
                    xmin = max(0.0, x_c - (box_width / 2.0))
                    ymax = min(1.0, y_c + (box_height / 2.0))
                    xmax = min(1.0, x_c + (box_width / 2.0))
```

Convertit le centre et les dimensions YOLO en coins :

- le bord supérieur est le centre vertical moins la moitié de la hauteur ;
- le bord gauche est le centre horizontal moins la moitié de la largeur ;
- les bords inférieur et droit ajoutent ces moitiés ;
- `max` et `min` bornent toutes les coordonnées dans `[0, 1]`.

```python
                    binary_class_id = int(original_class_name == 'debris')
```

Produit `1` pour `debris` et `0` pour toutes les autres classes. Le cast en `int` transforme le booléen en identifiant utilisable par TensorFlow.

```python
                    records.append({
                        'image_path': img_path,
                        'class_id': cls_id,
                        'class_name': original_class_name,
                        'binary_class_id': binary_class_id,
                        'binary_class_name': CLASS_NAMES[binary_class_id],
                        'ymin': ymin,
                        'xmin': xmin,
                        'ymax': ymax,
                        'xmax': xmax,
                    })
```

Ajoute toutes les informations de l'image dans un dictionnaire :

- chemin et classe originale ;
- identifiant et nom binaires ;
- quatre coordonnées normalisées.

```python
    return pd.DataFrame(records)
```

Transforme la liste de dictionnaires en DataFrame et la renvoie.

```python
df_train = create_dataset_dataframe('data/train/images')
df_val = create_dataset_dataframe('data/valid/images')
df_test = create_dataset_dataframe('data/test/images')
```

Construit les DataFrames d'entraînement, de validation et de test à partir des trois répertoires.

```python
print(f"Train : {len(df_train)} images")
print(f"Val   : {len(df_val)} images")
print(f"Test  : {len(df_test)} images")
```

Affiche le nombre de lignes dans chaque DataFrame, donc le nombre d'images annotées retenues.

```python
print("Distribution binaire (Train) :")
print(df_train['binary_class_name'].value_counts())
```

Affiche le nombre d'images de chaque classe binaire dans l'entraînement.

```python
df_train.head()
```

Demande au notebook d'afficher les premières lignes du DataFrame pour contrôler les colonnes et les valeurs.

### Cellules 7 à 11

- La cellule 7 contient seulement `df_train` et affiche le DataFrame.
- La cellule 8 crée le titre `## 2. Exploration des données (EDA)`.
- La cellule 9 affiche à nouveau `df_train['binary_class_name'].value_counts()`, donc la distribution des classes.
- La cellule 10 explique textuellement le déséquilibre et la stratégie de rotations à 45, 60, 90, 180 et 270 degrés. Comme tout est en Markdown, aucune transformation n'est effectuée.
- La cellule 11 documente le pipeline `tf.data`, le chargement parallèle, le redimensionnement, la normalisation et les sorties nommées.

### Cellule 12 — fonction `df_to_tf_dataset`

```python
def df_to_tf_dataset(df, img_size=IMG_SIZE, batch_size=BATCH_SIZE, is_training=True):
```

Définit une fonction de conversion d'un DataFrame en dataset TensorFlow. Les paramètres ont des valeurs par défaut configurables.

```python
    image_paths = df['image_path'].values
    bboxes = df[['ymin', 'xmin', 'ymax', 'xmax']].values.astype(np.float32)
    class_ids = df['binary_class_id'].values.astype(np.int32)
```

Extrait les chemins, les boîtes et les identifiants binaires. Les types NumPy sont choisis pour correspondre aux attentes TensorFlow.

```python
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes, class_ids))
```

Crée un dataset où chaque élément contient le chemin, la boîte et la classe d'une image.

```python
    def load_and_preprocess(img_path, bbox, cls_id):
```

Définit la fonction appelée pour prétraiter un élément.

```python
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
```

Lit les octets du fichier puis décode l'image en trois canaux RGB.

```python
        img = tf.image.resize(img, [img_size, img_size])
```

Redimensionne l'image en carré, par défaut `224 × 224`.

```python
        img = tf.cast(img, tf.float32) / 255.0
```

Convertit les pixels en flottants puis les normalise dans l'intervalle `[0, 1]`.

```python
        return img, {
            'class_output': cls_id,
            'bbox_output': bbox,
        }
```

Retourne l'image et un dictionnaire contenant les deux cibles nommées.

```python
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            reshuffle_each_iteration=True,
        )
```

Mélange uniquement les données d'entraînement. La taille du buffer est limitée à 1000 ou au nombre d'exemples si le dataset est plus petit. Le mélange est renouvelé à chaque passage.

```python
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
```

Applique le prétraitement à chaque élément en laissant TensorFlow choisir le parallélisme adapté.

```python
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

- `batch` regroupe les exemples.
- `prefetch` prépare les lots suivants pendant que le modèle travaille.
- `return` renvoie le dataset prêt à l'emploi.

```python
train_ds = df_to_tf_dataset(df_train, is_training=True)
val_ds = df_to_tf_dataset(df_val, is_training=False)
test_ds = df_to_tf_dataset(df_test, is_training=False)
```

Construit les trois datasets. Seul l'entraînement est mélangé.

```python
print("Datasets TensorFlow construits et prêts pour l'entraînement !")
```

Affiche une confirmation de fin de construction.

### Cellules 13 et 14 — visualisation

La cellule 13 introduit la vérification visuelle.

```python
for images, targets in train_ds.take(1):
```

Prend un seul lot du dataset d'entraînement.

```python
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
```

Crée une grille de huit graphiques et l'aplatit pour accéder aux axes avec un seul index.

```python
    for i in range(min(8, len(images))):
```

Parcourt au plus huit images, même si le dernier lot est incomplet.

```python
        img = images[i].numpy()
        cls_id = int(targets['class_output'][i].numpy())
        ymin, xmin, ymax, xmax = targets['bbox_output'][i].numpy()
```

Convertit les tenseurs du lot en valeurs NumPy, récupère la classe et décompose la boîte.

```python
        h, w, _ = img.shape
        box_y = ymin * h
        box_x = xmin * w
        box_h = (ymax - ymin) * h
        box_w = (xmax - xmin) * w
```

Convertit les coordonnées normalisées en coordonnées pixels adaptées à Matplotlib.

```python
        axes[i].imshow(img)
        rect = patches.Rectangle(
            (box_x, box_y), box_w, box_h,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        axes[i].add_patch(rect)
```

Affiche l'image, crée un rectangle vert sans remplissage et l'ajoute à l'axe.

```python
        axes[i].set_title(
            f"{CLASS_NAMES[cls_id]}",
            color='white',
            backgroundcolor='black',
            fontsize=10,
        )
        axes[i].axis('off')
```

Affiche le nom de la classe et masque les axes pour rendre la grille plus lisible.

```python
    plt.tight_layout()
    plt.show()
    break
```

Ajuste l'espacement, affiche la figure, puis quitte la boucle après le premier lot.

### Cellules 15 et 16 — expérimentation HSV

La cellule 15 est un titre Markdown.

Dans la cellule 16, chaque ligne commence par `#`, donc aucune instruction n'est exécutée. Le code prévu aurait les fonctions suivantes :

- importer `Path`, PIL et `rgb_to_hsv` ;
- sélectionner jusqu'à 20 images de test ;
- définir `hue_range`, `saturation_range` et `value_range` ;
- convertir chaque image RGB en tableau flottant ;
- convertir RGB vers HSV ;
- créer un masque selon les plages HSV ;
- mettre à zéro les pixels hors masque ;
- afficher les images RGB et HSV par pages de dix ;
- libérer chaque figure avec `plt.close` ;
- imprimer le nombre d'images filtrées.

Les valeurs proposées sont des réglages expérimentaux, pas une étape active du pipeline d'entraînement.

### Cellule 18 — pipeline de classification

```python
from tensorflow import keras
from tensorflow.keras import layers
```

Importe l'API Keras et ses couches. `layers` est disponible pour construire un modèle, même si cette cellule ne crée pas de réseau.

```python
def df_to_classification_dataset(
    df,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    is_training=True,
):
```

Définit une variante du pipeline qui ne retourne qu'une classe et non une boîte.

```python
    image_paths = df['image_path'].values
    class_ids = df['binary_class_id'].values.astype(np.int32)
```

Extrait les chemins et les classes binaires, sans extraire de coordonnées.

```python
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_ids))
```

Associe chaque chemin à son identifiant de classe.

```python
    def load_and_preprocess(img_path, class_id):
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, class_id
```

Cette fonction lit, décode, redimensionne et normalise l'image, puis retourne l'image et son identifiant binaire.

```python
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            reshuffle_each_iteration=True,
        )
```

Mélange le dataset d'entraînement avec la même logique que le pipeline multi-sorties.

```python
    return dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

Applique le prétraitement en parallèle, regroupe les images et précharge les lots avant de renvoyer le dataset.

```python
df_train_5050 = create_dataset_dataframe('data/train_50-50/images')
```

Lit le répertoire équilibré préparé en amont.

```python
train_classification_ds = df_to_classification_dataset(
    df_train_5050,
    is_training=True,
)
val_classification_ds = df_to_classification_dataset(df_val, is_training=False)
test_classification_ds = df_to_classification_dataset(df_test, is_training=False)
```

Construit le dataset d'entraînement équilibré et les datasets de validation/test à partir des données originales.

```python
print(f"Train 50/50 : {len(df_train_5050)} images")
print(df_train_5050['binary_class_name'].value_counts())
```

Affiche la taille et la distribution du jeu équilibré.

```python
from pathlib import Path
model_path = Path("models/satellite_debris_best.keras")
model = keras.models.load_model(model_path)
print(f"Modèle rechargé depuis : {model_path}")
```

- `Path` fournit une représentation portable du chemin.
- `model_path` pointe vers le modèle sauvegardé.
- `load_model` restaure l'architecture et les poids.
- le `print` confirme le fichier chargé.

### Cellules 19 à 20 — callback de prévisualisation

La cellule 19 est une introduction Markdown.

```python
class EpochPreviewCallback(keras.callbacks.Callback):
```

Déclare un callback Keras exécutable pendant l'entraînement.

```python
    def __init__(self, dataset, class_names):
        super().__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.examples_by_class = {
            class_id: [] for class_id in range(len(class_names))
        }
```

Initialise le callback, conserve le dataset et les noms, puis prépare une liste d'images pour chaque classe.

```python
        for images, class_ids in dataset:
            for index in range(images.shape[0]):
                class_id = int(class_ids[index].numpy())
                self.examples_by_class[class_id].append(images[index].numpy())
```

Parcourt la validation et stocke chaque image dans la liste correspondant à sa classe réelle.

```python
    def on_epoch_end(self, epoch, logs=None):
        predicted_counts = np.zeros(
            len(self.class_names),
            dtype=np.int32,
        )
```

Définit l'action de fin d'époque et initialise un compteur de prédictions par classe.

```python
        for images, _ in self.dataset:
            predictions = self.model.predict(images, verbose=0)
            predicted_class_ids = np.argmax(predictions, axis=1)
            predicted_counts += np.bincount(
                predicted_class_ids,
                minlength=len(self.class_names),
            )
```

Prédit chaque lot, prend l'indice de probabilité maximale, compte les classes prédites et accumule les résultats.

```python
        print(
            f"Epoch {epoch + 1} - détections validation : "
            f"{self.class_names[0]}={predicted_counts[0]}, "
            f"{self.class_names[1]}={predicted_counts[1]}"
        )
```

Affiche le nombre d'images attribuées à chaque classe à la fin de l'époque.

```python
        for class_id, class_name in enumerate(self.class_names):
            examples = self.examples_by_class[class_id]
            image = examples[np.random.randint(len(examples))][np.newaxis, ...]
```

Sélectionne une image aléatoire parmi les exemples de chaque classe et ajoute une dimension pour former un batch de taille un.

```python
            prediction = self.model.predict(image, verbose=0)
            predicted_class_id = int(np.argmax(prediction[0]))
```

Calcule la prédiction et récupère la classe ayant la plus forte sortie.

```python
            figure, axis = plt.subplots(figsize=(5, 5))
            axis.imshow(image[0])
            axis.set_title(
                f"Epoch {epoch + 1} | Réel : {class_name} "
                f"| Prévu : {self.class_names[predicted_class_id]}"
            )
            axis.axis("off")
            figure.tight_layout()
            plt.show()
```

Construit et affiche une figure montrant l'image, l'époque, la classe réelle et la classe prévue.

```python
            if wandb.run is not None:
                wandb.log({
                    f"epoch_preview_{class_name}": wandb.Image(
                        figure,
                        caption=(
                            f"Epoch {epoch + 1} - exemple {class_name} - "
                            f"prévu : {self.class_names[predicted_class_id]}"
                        ),
                    )
                })
```

Si une exécution W&B existe, enregistre la figure avec une clé propre à la classe et une légende descriptive.

```python
            plt.close(figure)
```

Ferme la figure pour éviter d'accumuler des figures en mémoire.

### Cellules 21 à 23

- La cellule 21 explique le suivi d'expérience W&B.
- Dans la cellule 22, `import wandb` importe le client W&B et `WandbMetricsLogger` importe l'intégration Keras.
- La cellule 23 documente l'entraînement sur le jeu équilibré et le suivi des métriques.

### Cellule 24 — entraînement

```python
from pathlib import Path
from tensorflow import keras
```

Réimporte `Path` et Keras pour rendre la cellule autonome lors d'une exécution séparée.

```python
MODEL_PATH = Path("models/satellite_debris_best.keras")
NEXT_MODEL_PATH = Path("models/satellite_debris_best_v2.keras")
```

Définit le modèle d'entrée et le chemin de sortie de la nouvelle version.

```python
model = keras.models.load_model(MODEL_PATH)
```

Recharge le modèle existant avec son architecture et ses poids.

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
```

- utilise Adam avec un faible taux d'apprentissage ;
- utilise l'entropie croisée sparse pour des labels entiers `0` et `1` ;
- demande à Keras de calculer l'accuracy.

```python
CONTINUATION_EPOCHS = 5
```

Fixe le nombre maximal d'époques supplémentaires à cinq.

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    NEXT_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)
```

Configure la sauvegarde du meilleur modèle selon l'accuracy de validation la plus élevée.

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)
```

Arrête l'entraînement après deux époques sans amélioration de `val_loss` et restaure les meilleurs poids observés.

```python
history = model.fit(
    train_classification_ds,
    validation_data=val_classification_ds,
    epochs=CONTINUATION_EPOCHS,
    callbacks=[checkpoint, early_stopping],
)
```

Lance la poursuite de l'entraînement sur les données 50/50, évalue chaque époque sur la validation et conserve les deux callbacks.

```python
model.save(NEXT_MODEL_PATH)
print(f"Modèle continué sauvegardé dans : {NEXT_MODEL_PATH}")
```

Sauvegarde explicitement le modèle final en plus de la sauvegarde conditionnelle du checkpoint, puis affiche son emplacement.

### Cellules 25 et 26 — courbes

La cellule 25 introduit les courbes.

```python
metrics = history.history
```

Récupère le dictionnaire Keras contenant les valeurs par époque.

```python
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
```

Crée une figure avec deux graphiques côte à côte.

```python
axes[0].plot(history.epoch, metrics["loss"], "o-", label="train")
axes[0].plot(history.epoch, metrics["val_loss"], "o-", label="validation")
axes[0].set_title("Loss de classification")
axes[0].set_xlabel("Epoch")
axes[0].legend()
```

Trace la perte d'entraînement et de validation en fonction du numéro d'époque, puis ajoute un titre, un axe et une légende.

```python
axes[1].plot(history.epoch, metrics["accuracy"], "o-", label="train")
axes[1].plot(history.epoch, metrics["val_accuracy"], "o-", label="validation")
axes[1].set_title("Accuracy de classification")
axes[1].set_xlabel("Epoch")
axes[1].legend()
```

Trace de la même manière l'accuracy d'entraînement et de validation.

```python
plt.tight_layout()
plt.show()
```

Ajuste la mise en page et affiche la figure.

### Cellules 27 et 28 — test final

La cellule 27 rappelle que le test doit rester indépendant de l'apprentissage.

```python
test_results = model.evaluate(test_classification_ds, return_dict=True)
```

Évalue le modèle sur le dataset de test et renvoie les métriques sous forme de dictionnaire nommé.

```python
print("Résultats de classification sur le jeu de test :")
```

Affiche un en-tête pour rendre la sortie lisible.

```python
for metric_name, metric_value in test_results.items():
    print(f"{metric_name}: {metric_value:.4f}")
```

Parcourt chaque métrique et affiche sa valeur avec quatre chiffres après la virgule.

---

## 4. Résumé du flux d'exécution

Dans l'ordre, le notebook :

1. charge les bibliothèques et la configuration YAML ;
2. lit les images et leurs annotations YOLO ;
3. convertit les coordonnées en boîtes normalisées ;
4. convertit les 11 classes originales en deux classes ;
5. inspecte le déséquilibre des classes ;
6. construit un pipeline TensorFlow multi-sorties ;
7. vérifie visuellement les boîtes ;
8. construit un pipeline de classification sur `train_50-50` ;
9. recharge `satellite_debris_best.keras` ;
10. continue son entraînement avec arrêt anticipé ;
11. sauvegarde `satellite_debris_best_v2.keras` ;
12. affiche les courbes ;
13. évalue le résultat final sur le jeu de test.

## 5. Remarques de cohérence du notebook

- La cellule 16 est entièrement commentée : l'expérimentation HSV n'est pas active.
- `EpochPreviewCallback` est utilisé pendant l'entraînement et affiche à chaque époque les vrais positifs `debris` et `satellite` sur le total de leur classe dans la validation équilibrée.
- `WandbMetricsLogger` est importé mais n'apparaît pas dans `callbacks=[checkpoint, early_stopping]`.
- Le pipeline des cellules 12 à 14 prépare des boîtes, tandis que l'entraînement des cellules 18 à 28 est un entraînement de classification uniquement.
- Le modèle chargé doit avoir une sortie compatible avec `SparseCategoricalCrossentropy` et des labels entiers binaires.
- La conversion `img_path.replace('images', 'labels')` suppose que les chemins suivent bien la structure `.../images/...` et `.../labels/...`.
- Le dossier `data/valid_50-50/` contient une sélection fixe de 21 satellites et 21 débris provenant de `data/valid/`. Cette validation équilibrée est utilisée pour les courbes d'entraînement ; la validation originale reste évaluée séparément pour comparaison.
- La cellule d'entraînement peut maintenant construire un MobileNetV2 avec `weights="imagenet"`. La base ImageNet est gelée au premier entraînement et une tête `Dense` à deux sorties est ajoutée pour la classification `satellite`/`debris`. Le prétraitement de classification utilise `keras.applications.mobilenet_v2.preprocess_input`.
