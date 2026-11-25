# Choix des modèles et hyperparamètres

Ce document présente les architectures, les stratégies d'apprentissage et les checkpoints retenus pour la classification binaire et la localisation.

## Paramètres communs

| Paramètre | Valeur |
|---|---:|
| Taille d'image | 224 × 224 |
| Taille de lot | 16 |
| Optimiseur | Adam |
| Taux d'apprentissage initial | `1e-3` |
| Nombre maximal d'époques (classification) | 70 |
| Nombre maximal d'époques (boîtes) | 60 |
| Seuil de classification | 0,5 |
| Seuil IoU d'une boîte correcte | 0,5 |

## Classification satellite/débris

### Architecture et entraînement

Le classifieur utilise MobileNetV2 pré-entraîné sur ImageNet, sans sa tête de classification, suivi de `GlobalAveragePooling2D`, d'un `Dropout(0.3)` et d'une sortie sigmoïde. La fonction de perte est `BinaryCrossentropy`. Les poids de classe sont calculés sur l'ensemble d'entraînement pour compenser le déséquilibre entre satellites et débris.

Le checkpoint est conservé selon la PR-AUC sur validation (`val_pr_auc`). L'arrêt anticipé surveille cette même métrique, avec une patience de 65 époques.

### Stratégies comparées

| Stratégie | Configuration |
|---|---|
| Taux fixe | `1e-3` |
| Décroissance cosinus | de `1e-3` à 1 % du taux initial |
| Réduction sur plateau | division par deux après stagnation, minimum `1e-6` |

### Résultats finaux enregistrés sur test

Le modèle retenu est `satellite_debris_cosin_70_retrain.keras`. Au seuil de 0,5, son rapport final indique :

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| Satellite | 98,25 % | 100 % | 99,12 % | 112 |
| Débris | 100 % | 81,82 % | 90,00 % | 11 |

L'accuracy globale est de 98,37 % et la matrice de confusion est `[[112, 0], [2, 9]]`. L'average precision calculée par scikit-learn dans le fichier de résultats est de 0,9202. Les métriques sont à interpréter avec prudence, car le test ne contient que 11 débris.

## Classification et localisation par boîte

### Architecture et entraînement

Le modèle multitâche partage un backbone MobileNetV2 et possède deux sorties :

- `class_output` : probabilité de la classe débris, avec une activation sigmoïde ;
- `bbox_output` : quatre coordonnées normalisées `[ymin, xmin, ymax, xmax]`.

La perte de classification est `BinaryCrossentropy`; celle des boîtes est `Huber`. Les poids de pertes sont respectivement de 1 et 5. Le modèle utilise Adam et le checkpoint est sélectionné selon l'IoU sur validation (`val_bbox_output_iou`), également surveillée par l'arrêt anticipé.

### Résultats finaux enregistrés sur test

Le modèle retenu est `satellite_debris_bbox_lronplateau.keras`.

| Mesure | Résultat |
|---|---:|
| Accuracy de classification | 97,56 % |
| IoU moyenne | 0,6604 |
| IoU médiane | 0,6945 |
| Boîtes avec IoU ≥ 0,5 | 87,80 % |
| Détections avec classe correcte et IoU ≥ 0,5 | 87,80 % |

La performance de localisation est plus faible pour les débris que pour les satellites : l'IoU moyenne est de 0,3566 pour les débris contre 0,6693 pour les satellites. Sur les 11 débris du test, 4 seulement ont une boîte avec IoU ≥ 0,5.

## Protocole d'évaluation et limites

Le seuil de classification est fixé à 0,5. Les seuils alternatifs sont étudiés sur validation, pas ajustés en fonction des résultats du test. Les comparaisons historiques entre variantes sur le test sont descriptives et ne constituent pas une comparaison indépendante pour sélectionner des hyperparamètres.

Les résultats complets, matrices de confusion et analyses d'erreurs sont enregistrés dans `results/final/Classification/` et `results/final/BBOX/`. Le faible nombre de débris dans les ensembles de validation et de test limite la précision des estimations de rappel et d'IoU par classe.
