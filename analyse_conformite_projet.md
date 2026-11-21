# Analyse de conformité

## Sujet

- Domaine : images.
- Tâche : classification satellite/debris puis localisation par boîte `[ymin, xmin, ymax, xmax]`.
- Framework : TensorFlow/Keras.
- Bonus : règles symboliques pour les sorties et la demande de revue humaine.

## Données

- Dataset Roboflow `Space debris - v2`, licence CC BY 4.0.
- 2467 images : 2105 entraînement, 239 validation, 123 test.
- 11 classes originales volontairement regroupées en deux classes : satellite et debris, afin de répondre directement à la mission de tri de l’environnement spatial.
- Train : 1919 satellites et 186 debris.
- Validation : 218 satellites et 21 debris.
- Test : 112 satellites et 11 debris.

Le déséquilibre est conservé sur validation et test. La faible taille du groupe debris demande donc d’interpréter les pourcentages avec prudence, surtout sur test.
Le regroupement est volontaire : les différents noms de satellites ne sont pas nécessaires pour décider si un objet est une opération satellite ou un debris. Ce choix simplifie le projet et évite de demander à un utilisateur non spécialiste de vérifier les identités techniques exactes.

## Prétraitement

- Lectures JPEG avec TensorFlow.
- Redimensionnement en 224 × 224.
- Prétraitement MobileNetV2.
- Conversion YOLO vers `[ymin, xmin, ymax, xmax]`.
- Lots de 16 avec `tf.data`, mélange entraînement et `prefetch`.
- Augmentation de classification : flip horizontal, flip vertical et rotation.
- Pas d’augmentation géométrique pour les boîtes afin de préserver les coordonnées.
- Poids de classe pour la classification.

## Modèles

- Classification : MobileNetV2 pré-entraîné sur ImageNet, pooling global, dropout 0.3 et sortie sigmoïde.
- Boîtes : même backbone, sorties sigmoïdes pour la classe et les quatre coordonnées.
- Perte classe : `BinaryCrossentropy`.
- Perte boîte : `Huber`, poids 5.
- Optimiseur : Adam.
- Seuil de décision : 0.5.
- Seuil de boîte : IoU 0.5.

Les variantes fixe, cosinus et plateau sont conservées dans les notebooks et centralisées dans [CHOIX_MODELES_ET_HYPERPARAMETRES.md](CHOIX_MODELES_ET_HYPERPARAMETRES.md).

## Évaluation

- Classification : accuracy, précision, rappel, F1, PR-AUC et matrice de confusion.
- Boîtes : MAE, IoU moyenne/médiane, pourcentage d’IoU au moins 0.5 et détection complète.
- Erreurs analysées : faux positifs, faux négatifs, petits objets, objets proches des bords et boîtes mal alignées.
- Validation distincte de test : aucune adaptation du seuil sur test.

Les résultats exacts doivent uniquement être repris après une exécution complète vérifiée depuis un nouveau noyau.

## État du livrable

Conforme :

- Sujet et données décrits.
- Pipeline TensorFlow documenté.
- Modèles, entraînement et expérimentations implémentés.
- Métriques adaptées et analyse des erreurs.
- README, contributeurs et instructions reproductibles.
- Démonstration par notebooks.
- IA symbolique avec justification lisible.

Reste à confirmer avant dépôt :

- nom du dossier demandé par e-campus ;
- exécution complète des deux notebooks ;
- sélection définitive des candidats ;
- sauvegarde des courbes et sorties finales ;
- test sur une nouvelle image.

## Points d’attention

- Très peu de debris dans la validation et le test, ce qui nécessite une analyse prudente des pourcentages.
- Une seule boîte principale par image, conforme à la simplification choisie pour ce projet.
- Redimensionnement pouvant réduire les détails des objets très petits.
- Résultats potentiellement sensibles au découpage source et aux augmentations.
- Le regroupement en deux classes est volontaire : il suffit à la mission de tri satellite/debris, sans exiger une identification experte des différents satellites.
