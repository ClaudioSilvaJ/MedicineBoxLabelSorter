from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Diretórios dos dados
diretorio_treinamento = 'files/imgs_trat/imgs'
diretorio_validacao = 'files/imgs_trat/valid_imgs'


train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 3
train_generator = train_datagen.flow_from_directory(
    diretorio_treinamento,  
    target_size=(300, 150),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    diretorio_validacao, 
    target_size=(300, 150), 
    batch_size=batch_size,
    class_mode='binary'
)

model = Sequential([
    Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(300, 150, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 20
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

model.save('modelo_1_90.h5')


saved_model = load_model('modelo_1_90.h5')

validation_loss, validation_accuracy = saved_model.evaluate(validation_generator)

print(f'Acurácia no conjunto de validação: {validation_accuracy}')
print(f'Perda no conjunto de validação: {validation_loss}')
