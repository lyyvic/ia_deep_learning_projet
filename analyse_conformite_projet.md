# Analyse de conformité au sujet du projet

## Résumé

Le dépôt couvre désormais une bonne partie du sujet avec un domaine « Images » clairement défini : classifier les objets spatiaux en `satellite` ou `debris`, puis ajouter séparément l’apprentissage des boîtes englobantes dans un second notebook.

Le notebook de classification a été simplifié pour ne traiter que la classification binaire. Les étapes liées aux coordonnées YOLO, aux boîtes et à la localisation ont été retirées de ce pipeline. Elles restent disponibles uniquement dans `bounding_box_training.ipynb`, où elles relèvent explicitement de la détection avec deux sorties.

Les principaux points restant à compléter sont le README de rendu, la structure livrable demandée, une démonstration simple, la consolidation des métriques finales de détection et l’IA symbolique.

---

## Ce qui répond bien au sujet

### 1. Domaine, sujet et jeu de données

- Le sujet appartient au domaine « Images » et porte sur une tâche directement réalisable.
- Les données Roboflow contiennent des images avec annotations YOLO.
- La licence est documentée : **CC BY 4.0**.
- Les ensembles sont séparés en `train`, `valid` et `test`.
- Les données disponibles sont :
  - entraînement : 2105 images ;
  - validation : 239 images ;
  - test : 123 images.
- Dans l’entraînement réel, il y a 1919 satellites pour 186 débris : le déséquilibre est mesuré plutôt que supposé.
- Le notebook regroupe les dix classes satellites en une seule classe `satellite` et conserve `debris` comme classe positive.

### 2. Préparation des données

- Le notebook de classification lit uniquement l’identifiant YOLO nécessaire à la classe.
- Il construit un DataFrame contenant le chemin de l’image, la classe originale et la classe binaire.
- Il utilise `tf.data.Dataset`, `from_tensor_slices`, `shuffle`, `batch`, `prefetch`, `decode_jpeg` et `resize`.
- Les pixels sont adaptés au prétraitement MobileNetV2.
- Les poids de classe compensent le déséquilibre entre satellites et débris.
- Les ensembles d’entraînement, validation et test sont construits séparément.

Le pipeline de classification est maintenant cohérent : une image produit une seule sortie de classe.

### 3. Modèle de classification

- MobileNetV2 pré-entraîné sur ImageNet est utilisé comme backbone.
- Une sortie sigmoïde unique prédit la probabilité que l’objet soit un débris.
- Le code réalise :
  - augmentation aléatoire ;
  - dropout ;
  - entropie croisée binaire ;
  - poids de classe ;
  - `EarlyStopping` ;
  - `ModelCheckpoint` ;
  - plusieurs stratégies de taux d’apprentissage ;
  - suivi avec W&B.
- Plusieurs variantes ont été entraînées : taux fixe, cosine decay et réduction sur plateau.
- Les modèles de classification sont regroupés dans `models/Class_Detection/`.

### 4. Évaluation de classification

- Les métriques adaptées à une classe minoritaire sont présentes :
  - accuracy ;
  - précision ;
  - rappel ;
  - F1-score ;
  - PR-AUC ;
  - matrice de confusion ;
  - courbes d’apprentissage.
- Le seuil de décision peut être analysé sur la validation avant une mesure finale sur le test.
- Les journaux conservent des résultats antérieurs :
  - accuracy jusqu’à 98,37 % ;
  - précision debris jusqu’à 100 % ;
  - rappel debris jusqu’à 81,82 % ;
  - F1 debris jusqu’à 90 % ;
  - PR-AUC debris jusqu’à 0,8501.

Ces valeurs doivent rester accompagnées du faible nombre de débris dans le test : seulement 11 exemples sur 123.

### 5. Pipeline séparé de boîtes englobantes

- `bounding_box_training.ipynb` contient la logique de détection séparée :
  - extraction des coordonnées YOLO ;
  - conversion en `[ymin, xmin, ymax, xmax]` ;
  - sortie binaire de classe ;
  - sortie de quatre coordonnées ;
  - métriques IoU, MAE et accuracy des boîtes ;
  - visualisation des boîtes vraies et prédites ;
  - évaluation combinée classe plus boîte.
- Les modèles sont maintenant dirigés vers `models/BBOX/`.
- La reprise du modèle de classification pointe vers `models/Class_Detection/`.

Cette séparation est plus claire que l’ancien notebook mixte.

---

## Éléments encore éloignés du livrable

### 1. Simplification des 11 classes

Le modèle final ne produit pas les 11 classes originales. Il transforme volontairement le problème en classification satellite/débris.

Cette simplification est défendable, mais elle doit être explicitée dans le rapport comme un choix lié à la problématique retenue. Il faut aussi conserver la correspondance avec les 11 catégories et expliquer pourquoi elles ont été regroupées.

### 2. Résultats non consolidés dans le dépôt

Les notebooks n’ont plus de sorties sauvegardées et les preuves les plus détaillées sont actuellement dans les logs ou W&B.

Avant la remise, il faut produire une synthèse reproductible contenant :

- le modèle retenu ;
- les hyperparamètres finaux ;
- les courbes ;
- le rapport de classification sur test ;
- les résultats IoU du modèle de boîtes ;
- la comparaison des variantes ;
- l’analyse des erreurs.

### 3. Reproductibilité à finaliser

Les chemins des modèles ont été corrigés dans les notebooks principaux :

- classification : `models/Class_Detection/` ;
- boîtes : `models/BBOX/`.

Il reste à vérifier une exécution complète du début à la fin, sans dépendance à un état déjà chargé dans le noyau. Les chemins relatifs supposent l’exécution depuis la racine `leo/`.

Les modèles temporaires ou noms ambigus devraient aussi être renommés : `A_SUPPRIMER`, `cosin_decay`, `lronplateau`, `hardcoded`, `v2`, `v3`, `best_4`.

### 4. Notebooks sans sorties finales

Les deux notebooks ont été sauvegardés sans sorties. Cela évite de conserver des sorties obsolètes, mais le rapport et la démonstration devront éventuellement produire des résultats visibles lors de la présentation.

---

## Ce qui manque encore clairement

### 1. Rapport README

Le dépôt ne contient pas encore de README de rendu détaillé. Le README doit préciser :

- sujet, problématique et objectifs ;
- source, taille, licence et obtention des données ;
- prétraitement et découpage ;
- architecture ;
- entraînement ;
- hyperparamètres testés ;
- courbes ;
- résultats ;
- analyse des erreurs ;
- limites ;
- améliorations ;
- sources ;
- répartition des contributions.

`UTILS/main_notebook_explique.md` peut servir de base, mais il n’est pas le README final demandé.

### 2. Structure de livrable

Le dépôt ne contient pas encore :

- `CONTRIBUTORS` ;
- dossier `src/` ;
- dossier `groupe_N1_N2` ;
- README à la structure e-campus.

Il faut également vérifier que les notebooks finaux sont inclus dans la structure attendue.

### 3. Démonstration déployée

Les notebooks constituent une démonstration consultable, mais pas encore une interface clairement utilisable pour tester une nouvelle image.

La démo peut être :

- une cellule prenant une image locale ;
- une CLI ;
- une application web ;
- ou une interface notebook pré-exécutable.

Elle doit afficher au minimum la classe prédite, la probabilité de débris et, pour le pipeline de détection, la boîte prédite.

### 4. Résultats de détection à consolider

Le notebook de boîtes contient les bonnes métriques, mais les résultats doivent être produits sur le test :

- IoU moyenne ;
- IoU médiane ;
- taux de boîtes avec IoU >= 0,5 ;
- précision et rappel de classe ;
- taux de détection complète correcte ;
- comparaison des modèles ;
- exemples d’échecs.

### 5. Analyse des erreurs

Le rapport doit ajouter une analyse qualitative :

- faux positifs ;
- faux négatifs ;
- petites boîtes ;
- images ambiguës ;
- cas difficiles des classes satellites ;
- effets du redimensionnement ;
- effets des augmentations.

### 6. IA symbolique bonus

Aucune composante symbolique n’est actuellement implémentée :

- aucune règle métier ;
- aucun graphe de connaissances ;
- aucune contrainte logique sur les boîtes ;
- aucun raisonnement explicite sur les décisions.

Le bonus reste donc à réaliser ou explicitement déclaré absent.

---

## Tableau synthétique

| Exigence | État actuel | Niveau |
|---|---|---|
| Domaine et sujet précis | Classification satellite/débris | Bien traité |
| Jeu de données adapté | Roboflow YOLO, licence CC BY 4.0 | Bien traité |
| Prétraitement des images | Redimensionnement, prétraitement MobileNetV2, tf.data | Bien traité |
| Découpage train/validation/test | Présent | Bien traité |
| Gestion du déséquilibre | Poids de classe ; 1919 satellites contre 186 débris | Bien traité |
| Architecture TensorFlow | MobileNetV2, sortie sigmoïde | Bien traité |
| Hyperparamètres | Plusieurs LR testés, augmentation, dropout | À comparer dans un tableau final |
| Courbes d’apprentissage | Code présent | Résultats finaux à consolider |
| Métriques de classification | Accuracy, précision, rappel, F1, PR-AUC | Bien implémenté |
| Détection des boîtes | Notebook séparé, IoU, MAE, boîte correcte | Implémenté ; résultats à publier |
| Analyse des erreurs | Limitée | À améliorer |
| Code reproductible | Chemins principaux corrigés | À vérifier bout en bout |
| README détaillé | Absent | Manquant |
| CONTRIBUTORS | Absent | Manquant |
| Structure `src/` | Absente | Manquant |
| Démo | Notebook uniquement | À finaliser |
| IA symbolique | Absente | Bonus manquant |

---

## Priorités avant rendu

1. Exécuter `Class_detection_training.ipynb` du début à la fin depuis la racine du dépôt.
2. Exécuter `bounding_box_training.ipynb` avec le modèle de classification corrigé dans `models/Class_Detection/`.
3. Produire un tableau comparatif des hyperparamètres et modèles retenus.
4. Sauvegarder les courbes, matrices de confusion, résultats IoU et exemples d’erreurs dans le rendu.
5. Ajouter un README, `CONTRIBUTORS` et organiser le code dans `src/`.
6. Ajouter une démonstration permettant de charger une nouvelle image.
7. Décider puis documenter clairement le statut du bonus IA symbolique.
