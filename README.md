# Pokemon Image Classification

## Project Overview
This project builds a deep learning model to classify images of Pokémon into 150 different categories. The dataset consists of training images organized into subfolders and test images for classification.

## Dataset Description
- **Train Data:**
  - Organized into 150 subfolders, each named after a Pokémon category.
  - Each subfolder contains around 55 images for training.
- **Test Data:**
  - Contains 2,195 unlabeled images named sequentially (1.jpg to 2195.jpg).
- **Submission Format:**
  - A CSV file with two columns:
    - `Img_name`: Test image filename.
    - `Class Prediction`: Predicted Pokémon category.

## Model Architecture
The model is built using **Keras with TensorFlow** backend and follows a CNN-based architecture:
- Multiple **Conv2D** layers for feature extraction.
- **MaxPooling2D** layers to reduce spatial dimensions.
- **Flatten** layer to convert features into a 1D array.
- **Dense** layers with ReLU activation.
- **Dropout** layer to prevent overfitting.
- Final **Softmax** layer for classification.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(150, activation='softmax')
])
```

## Training the Model
The model is compiled and trained using:
```python
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    verbose=1
)
```

## Improving Model Accuracy
- **Data Augmentation:** Applied transformations like rotation, zoom, and flipping.
- **Adding More Layers:** Deeper CNN architecture.
- **Transfer Learning:** Using pre-trained models like ResNet or EfficientNet.

## Making Predictions
```python
for img_name in tqdm(test_images):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)
    class_label = list(train_generator.class_indices.keys())[pred_class[0]]
    predictions.append([img_name, class_label])
```

## Saving and Loading the Model
To save the model:
```python
model.save("pokemon_classifier.h5")
```
To load and retrain:
```python
from keras.models import load_model
model = load_model("pokemon_classifier.h5")
```

## Submission
Predictions are saved in `submission.csv`:
```python
import pandas as pd
submission = pd.DataFrame(predictions, columns=['Img_name', 'Class Prediction'])
submission.to_csv('submission.csv', index=False)
```

## Dependencies
- Python 3.10+
- TensorFlow/Keras
- NumPy
- Pandas
- OpenCV/PIL

## Usage
1. Install dependencies using `pip install -r requirements.txt`.
2. Train the model using `model.fit()`.
3. Predict Pokémon categories for test images.
4. Submit the results in `submission.csv`.



