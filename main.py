import kagglehub
kagglehub.login()

bttai_ajl_2025_path = kagglehub.competition_download('bttai-ajl-2025')

print('Data source import complete.')

# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Explanation:
# - pandas and numpy: for data manipulation
# - sklearn: for splitting data and encoding labels
# - tensorflow.keras: for building and training the neural network

# 2. Load Data
train_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/test.csv')

# Add .jpg extension to md5hash column to reference the file_name
train_df['md5hash'] += '.jpg'
test_df['md5hash'] += '.jpg'


# Combine label and md5hash to form the correct path
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

# 3. Data Preprocessing
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Split the data into training and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Define image data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define the directory paths
train_dir = '/kaggle/input/bttai-ajl-2025/train/train/'

def create_generator(df, datagen, directory, is_test=False):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=directory,
        x_col='md5hash' if is_test else 'file_path',
        y_col=None if is_test else 'encoded_label',
        target_size=(128, 128),
        batch_size=32,
        class_mode=None if is_test else 'raw',
        shuffle=not is_test,
        validate_filenames=False
    )


val_generator = create_generator(val_data, val_datagen, train_dir)
test_generator = create_generator(test_df, test_datagen, test_dir, is_test=True)


## 4. Build the model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


## 5. Train the Model


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# 6. Make Predictions on Test Data
def preprocess_test_data(test_df, directory):

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        x_col='md5hash',
        y_col=None,
        target_size=(128, 128),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    return test_generator

# Load test data
test_dir = '/kaggle/input/bttai-ajl-2025/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

# 7. Generate Predictions

predictions = model.predict(test_generator)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

submission_df = pd.DataFrame({
    'md5hash': test_df['md5hash'].str.replace('.jpg', '', regex=False),
    'label': predicted_labels
})

print("Final Training Accuracy:", round(history.history['accuracy'][-1], 4))
print("Final Validation Accuracy:", round(history.history['val_accuracy'][-1], 4))


submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

