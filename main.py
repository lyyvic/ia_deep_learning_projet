import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import yaml

# Configuration globale
IMG_SIZE = 224
BATCH_SIZE = 16

DATA_YAML_PATH = 'data/data.yaml'
with open(DATA_YAML_PATH, 'r', encoding='utf-8') as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

ORIGINAL_CLASS_NAMES = dataset_config['names']
CLASS_NAMES = ['satellite', 'debris']
NUM_CLASSES = len(CLASS_NAMES)

print(f"TensorFlow version: {tf.__version__}")
print(f"Classes originales : {ORIGINAL_CLASS_NAMES}")
print(f"Classes du modèle : {CLASS_NAMES}")

def create_dataset_dataframe(image_dir):
    """
    Lit les annotations YOLO et crée un DataFrame avec une cible binaire.
    Tous les objets sauf debris sont regroupés dans la classe satellite.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    records = []

    for img_path in image_paths:
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as label_file:
                line = label_file.readline().strip()
                if line:
                    parts = line.split()
                    cls_id = int(parts[0])
                    x_c, y_c, box_width, box_height = map(float, parts[1:5])
                    original_class_name = (
                        ORIGINAL_CLASS_NAMES[cls_id]
                        if 0 <= cls_id < len(ORIGINAL_CLASS_NAMES)
                        else 'unknown'
                    )

                    ymin = max(0.0, y_c - (box_height / 2.0))
                    xmin = max(0.0, x_c - (box_width / 2.0))
                    ymax = min(1.0, y_c + (box_height / 2.0))
                    xmax = min(1.0, x_c + (box_width / 2.0))

                    binary_class_id = int(original_class_name == 'debris')
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

    return pd.DataFrame(records)

# Création des DataFrames
df_train = create_dataset_dataframe('data/train/images')
df_val = create_dataset_dataframe('data/valid/images')
df_test = create_dataset_dataframe('data/test/images')

print(f"Train : {len(df_train)} images")
print(f"Val   : {len(df_val)} images")
print(f"Test  : {len(df_test)} images")
print("Distribution binaire (Train) :")
print(df_train['binary_class_name'].value_counts())
df_train.head()
print("Distribution des classes binaires (Train) :")
print(df_train['binary_class_name'].value_counts())








def df_to_tf_dataset(df, img_size=IMG_SIZE, batch_size=BATCH_SIZE, is_training=True):
    image_paths = df['image_path'].values
    bboxes = df[['ymin', 'xmin', 'ymax', 'xmax']].values.astype(np.float32)
    class_ids = df['binary_class_id'].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes, class_ids))

    def load_and_preprocess(img_path, bbox, cls_id):
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, {
            'class_output': cls_id,
            'bbox_output': bbox,
        }

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = df_to_tf_dataset(df_train, is_training=True)
val_ds = df_to_tf_dataset(df_val, is_training=False)
test_ds = df_to_tf_dataset(df_test, is_training=False)

print("Datasets TensorFlow construits et prêts pour l'entraînement !")


from tensorflow import keras
from tensorflow.keras import layers


def df_to_classification_dataset(
    df,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    is_training=True,
):
    image_paths = df['image_path'].values
    class_ids = df['binary_class_id'].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_ids))

    def load_and_preprocess(img_path, class_id):
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        img = keras.applications.mobilenet_v2.preprocess_input(img)
        return img, class_id

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            reshuffle_each_iteration=True,
        )

    return dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


df_train_5050 = create_dataset_dataframe('data/train_50-50/images')
train_classification_ds = df_to_classification_dataset(
    df_train_5050,
    is_training=True,
)

# Validation équilibrée: 21 satellites et 21 débris issus de data/valid.
val_original_classification_ds = df_to_classification_dataset(
    df_val,
    is_training=False,
)
val_classification_ds = df_to_classification_dataset(
    df_val,
    is_training=False,
)
test_classification_ds = df_to_classification_dataset(df_test, is_training=False)

print(f"Train 50/50 : {len(df_train_5050)} images")
print(df_train_5050['binary_class_name'].value_counts())
print(f"Validation originale : {len(df_val)} images")
print(df_val['binary_class_name'].value_counts())
print(f"Validation équilibrée : {len(df_val)} images")
print(df_val['binary_class_name'].value_counts())


from pathlib import Path
model_path = Path("models/satellite_debris_best.keras")
model = keras.models.load_model(model_path)
print(f"Modèle rechargé depuis : {model_path}")


class EpochPreviewCallback(keras.callbacks.Callback):
    def __init__(self, dataset, class_names):
        super().__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.examples_by_class = {class_id: [] for class_id in range(len(class_names))}

        for images, class_ids in dataset:
            for index in range(images.shape[0]):
                class_id = int(class_ids[index].numpy())
                self.examples_by_class[class_id].append(images[index].numpy())

    def on_epoch_end(self, epoch, logs=None):
        predicted_counts = np.zeros(len(self.class_names), dtype=np.int32)

        for images, _ in self.dataset:
            predictions = self.model.predict(images, verbose=0)
            predicted_class_ids = np.argmax(predictions, axis=1)
            predicted_counts += np.bincount(
                predicted_class_ids,
                minlength=len(self.class_names),
            )

        print(
            f"Epoch {epoch + 1} - détections validation : "
            f"{self.class_names[0]}={predicted_counts[0]}, "
            f"{self.class_names[1]}={predicted_counts[1]}"
        )

        for class_id, class_name in enumerate(self.class_names):
            examples = self.examples_by_class[class_id]
            image = examples[np.random.randint(len(examples))][np.newaxis, ...]
            prediction = self.model.predict(image, verbose=0)
            predicted_class_id = int(np.argmax(prediction[0]))

            figure, axis = plt.subplots(figsize=(5, 5))
            # MobileNetV2 attend des pixels dans [-1, 1]. Pour l'affichage,
            # reconvertit uniquement une copie vers l'intervalle RGB [0, 1].
            display_image = np.clip((image[0] + 1.0) / 2.0, 0.0, 1.0)
            axis.imshow(display_image)
            axis.set_title(
                f"Epoch {epoch + 1} | Réel : {class_name} "
                f"| Prévu : {self.class_names[predicted_class_id]}"
            )
            axis.axis("off")
            figure.tight_layout()
            plt.show()

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

            plt.close(figure)


            import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.init(project="satellite_debris_classification", name="mobilenetv2_50-50_balanced")
wandb_callback = WandbMetricsLogger(log_freq="epoch")



from pathlib import Path
from tensorflow import keras

NEXT_MODEL_PATH = Path("models_apres/satellite_debris_best_CosineDecay_1e-1__2.keras")
NEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NB_EPOCHS = 10

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None,
)
base_model.trainable = True

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="class_output",
)(x)
model = keras.Model(inputs, outputs)

steps_per_epoch = len(df_train_5050) // BATCH_SIZE
total_steps = steps_per_epoch * NB_EPOCHS

learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-1,
    decay_steps=total_steps,
    alpha=1e-2,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


checkpoint = keras.callbacks.ModelCheckpoint(
    NEXT_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=6,
    restore_best_weights=True,
)

epoch_preview = EpochPreviewCallback(val_classification_ds, CLASS_NAMES)

history = model.fit(
    train_classification_ds,
    validation_data=val_classification_ds,
    epochs=NB_EPOCHS,
    callbacks=[
    checkpoint,
    early_stopping,
    epoch_preview,
    wandb_callback
])
print(f"Modèle ImageNet sauvegardé dans : {NEXT_MODEL_PATH}")



metrics = history.history

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.epoch, metrics["loss"], "o-", label="train")
axes[0].plot(history.epoch, metrics["val_loss"], "o-", label="validation")
axes[0].set_title("Loss de classification")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.epoch, metrics["accuracy"], "o-", label="train")
axes[1].plot(history.epoch, metrics["val_accuracy"], "o-", label="validation")
axes[1].set_title("Accuracy de classification")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = []
y_pred = []

for images, labels in val_classification_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

display.plot(cmap="Blues")
plt.title("Matrice de confusion - validation")
plt.show()