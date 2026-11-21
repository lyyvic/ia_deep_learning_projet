# Modèles et hyperparamètres

## Paramètres communs

| Paramètre | Valeur |
|---|---|
| Taille d'image | 224 × 224 |
| Taille de lot | 16 |
| Optimiseur | Adam |
| Learning rate initial | `1e-3` |
| Epoques classification | jusqu'à 70 |
| Epoques boîtes | jusqu'à 60 |
| Seuil classe | 0.5 |
| Seuil IoU | 0.5 |

## Classification

Architecture :

```text
MobileNetV2 ImageNet sans tête
-> GlobalAveragePooling2D
-> Dropout(0.3)
-> Dense(1, sigmoid)
```

- Perte : `BinaryCrossentropy`.
- Sortie : probabilité d'être un debris.
- Fine-tuning du backbone MobileNetV2.
- Poids de classe calculés sur le train.
- Métriques : accuracy, précision, rappel, F1, PR-AUC.
- EarlyStopping : `val_pr_auc`, patience 65.
- Checkpoint : meilleur `val_pr_auc`.

| Learning rate | Configuration |
|---|---|
| Fixe | `1e-3` constant |
| Cosinus | décroissance de `1e-3` vers 1 % |
| Plateau | division par 0.5 sur stagnation, minimum `1e-6` |

Résultats déjà observés sur test :

| Variante | Accuracy | Précision debris | Rappel debris | F1 debris |
|---|---:|---:|---:|---:|
| Cosinus | 98.37 % | 100 % | 81.82 % | 90.00 % |
| Hardcoded v3 | 97.56 % | 100 % | 72.73 % | 84.21 % |
| Plateau | 97.56 % | 100 % | 72.73 % | 84.21 % |
| Hardcoded | 91.06 % | 0 % | 0 % | 0 % |

Candidat provisoire : `satellite_debris_cosin_70_retrain.keras`.

## Boîtes englobantes

Architecture :

```text
MobileNetV2 fine-tuné
-> GlobalAveragePooling2D
-> Dropout(0.2)
-> tête classe sigmoid
-> tête boîte sigmoid à quatre valeurs
```

- Classe : `BinaryCrossentropy`, poids 1.
- Boîte : `Huber`, poids 5.
- Sortie : `[ymin, xmin, ymax, xmax]`.
- Métriques : accuracy classe, MAE, IoU moyenne, IoU médiane, IoU ≥ 0.5.
- Checkpoint : meilleur `val_bbox_output_iou`.

Résultats déjà observés sur test :

| Variante | Accuracy classe | IoU moyenne | Boîtes IoU ≥ 0.5 |
|---|---:|---:|---:|
| Cosinus | 98.37 % | 66.22 % | 86.18 % |
| Hardcoded | 98.37 % | 65.75 % | 86.18 % |
| Plateau | 97.56 % | 66.03 % | 87.80 % |

Candidat provisoire : `satellite_debris_bbox_lronplateau.keras`.

## Règles de remise

- Le seuil 0.5 n'est pas optimisé sur le test.
- Les hyperparamètres ne sont pas modifiés après l'évaluation finale.
- Les modèles retenus doivent être confirmés par une nouvelle exécution complète.
- Les anciennes variantes restent des archives expérimentales.
