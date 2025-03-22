import tensorflow as tf

model = tf.keras.models.load_model('modelo.h5') # Path do modelo

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('modelo_tflite.tflite', 'wb') as f:
    f.write(tflite_model)
