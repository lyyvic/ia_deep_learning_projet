"""Règles symboliques explicables appliquées aux sorties du modèle."""

import math


UNCERTAIN_PROBABILITY_MARGIN = 0.10
SMALL_DEBRIS_AREA_THRESHOLD = 0.01


def apply_symbolic_rules(
    probability_debris,
    bbox,
    decision_threshold=0.5,
    predicted_class=None,
):
    """Interprète une prédiction sans modifier les sorties du réseau profond.

    Paramètres
    ----------
    probability_debris : float
        Probabilité prédite que l'objet soit un debris, dans [0, 1].
    bbox : séquence de quatre flottants
        Boîte au format [ymin, xmin, ymax, xmax], normalisée dans [0, 1].
    decision_threshold : float
        Seuil utilisé pour transformer la probabilité en classe.
    predicted_class : str or None
        Classe éventuellement fournie par une couche externe. Si elle est
        différente de la classe déduite du seuil, une incohérence est signalée.

    Retour
    ------
    dict
        Classe, surface normalisée, décisions symboliques et motivations.
    """
    probability = float(probability_debris)
    coordinates = [float(value) for value in bbox]
    area = None

    derived_class = (
        "debris" if probability >= decision_threshold else "satellite"
    )
    final_predicted_class = predicted_class or derived_class
    reasons = []
    invalid_output = False

    if not math.isfinite(probability):
        invalid_output = True
        reasons.append("La probabilité de debris n'est pas une valeur finie.")
    if not math.isfinite(decision_threshold):
        invalid_output = True
        reasons.append("Le seuil de décision n'est pas une valeur finie.")

    if len(coordinates) != 4:
        invalid_output = True
        reasons.append("La boîte ne contient pas exactement quatre coordonnées.")
    else:
        ymin, xmin, ymax, xmax = coordinates
        finite_coordinates = all(math.isfinite(value) for value in coordinates)

        if not finite_coordinates:
            invalid_output = True
            reasons.append("Une coordonnée de boîte n'est pas une valeur finie.")
        elif any(value < 0.0 or value > 1.0 for value in coordinates):
            invalid_output = True
            reasons.append("Une coordonnée de boîte est hors de [0, 1].")

        if finite_coordinates and (ymax < ymin or xmax < xmin):
            invalid_output = True
            reasons.append("La boîte est géométriquement inversée.")

        if finite_coordinates:
            area = max(0.0, ymax - ymin) * max(0.0, xmax - xmin)

        if probability < 0.0 or probability > 1.0:
            invalid_output = True
            reasons.append("La probabilité de debris est hors de [0, 1].")

    if (
        predicted_class is not None
        and predicted_class not in {"satellite", "debris"}
    ):
        invalid_output = True
        reasons.append("La classe fournie n'appartient pas aux classes du projet.")
    elif (
        predicted_class is not None
        and predicted_class != derived_class
    ):
        invalid_output = True
        reasons.append(
            "La classe prédite est incohérente avec la probabilité et le seuil."
        )

    if invalid_output:
        status = "sortie_invalide"
    elif abs(probability - decision_threshold) <= UNCERTAIN_PROBABILITY_MARGIN:
        status = "incertain"
        reasons.append(
            "La probabilité est proche du seuil de décision : "
            "une supervision humaine est recommandée."
        )
    elif final_predicted_class == "debris" and area is not None and area < SMALL_DEBRIS_AREA_THRESHOLD:
        status = "incertain"
        reasons.append(
            "Un debris a été prédit avec une boîte dont la surface normalisée "
            "est inférieure à 1 % de l'image."
        )
    else:
        status = "confiant"
        reasons.append("Les règles symboliques ne détectent aucune anomalie.")

    if len(coordinates) == 4:
        area_value = area
    else:
        area_value = None

    return {
        "status": status,
        "predicted_class": final_predicted_class,
        "probability_debris": probability,
        "bbox": coordinates,
        "bbox_area": area_value,
        "requires_human_review": status != "confiant",
        "reasons": reasons,
    }
