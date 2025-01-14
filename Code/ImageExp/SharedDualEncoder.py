import tensorflow as tf


def create_encoder(height=250, width=250):
    start_size = 64
    input_shape = (height, width, 3)

    base_model = tf.keras.Sequential()
    base_model.add(
        tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                               input_shape=input_shape))
    base_model.add(
        tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                               input_shape=input_shape))
    base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(
        tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu'))
    base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    base_model.add(
        tf.keras.layers.Conv2D(4096, kernel_size=(7, 7), strides=(1, 1), padding='valid',
                               activation='relu'))
    base_model.add(tf.keras.layers.Dropout(0.5))
    base_model.add(
        tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               activation='relu'))
    base_model.add(tf.keras.layers.Dropout(0.5))
    base_model.add(
        tf.keras.layers.Conv2D(2622, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               activation='relu'))

    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Activation('softmax'))
    base_model.load_weights('/Users/manojreddy/Documents/Comparable/Data/vgg_face_weights.h5')

    # for layer in base_model.layers[:-7]:
    #     layer.trainable = False

    base_model_output = tf.keras.layers.Flatten()(base_model.layers[-4].output)
    base_model_output = tf.keras.layers.Dense(256, activation='relu')(base_model_output)
    base_model_output = tf.keras.layers.Dense(1)(base_model_output)

    return tf.keras.Model(inputs=base_model.input, outputs=base_model_output)


class DualEncoderAll(tf.keras.Model):
    def __init__(self, encoder, y_true, **kwargs):
        super(DualEncoderAll, self).__init__(**kwargs)
        self.encoder = encoder
        self.y_true = y_true
        self.temperature = 0.05
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        y = features["Label"]
        self.encoder.trainable = trainable
        return encodings_A, encodings_B, y

    def compute_loss(self, encodings_A, encodings_B, y):
        encodings_A = tf.squeeze(encodings_A)
        encodings_B = tf.squeeze(encodings_B)
        # pred = (encodings_A + encodings_B) / 2
        pred = encodings_A - encodings_B
        y = tf.cast(y, tf.float32)

        # loss = tf.math.abs(y - pred)

        # Hinge loss
        loss = tf.math.maximum(0.0, 1.0 - (y * pred))

        return loss

    def train_step(self, feature, trainable=True):
        with tf.GradientTape() as tape:
            # Forward pass
            encodings_A, encodings_B, y = self(feature, trainable=trainable)
            loss = self.compute_loss(encodings_A, encodings_B, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, feature):
        encodings_A, encodings_B, y = self(feature, trainable=self.trainable)
        loss = self.compute_loss(encodings_A, encodings_B, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        return self.encoder(A) - self.encoder(B)

    def score(self, input):
        return self.encoder(input)

    def save(self, path):
        self.encoder.save_weights(path + "_A")
        self.encoder.save_weights(path + "_B")

    def load(self, path):
        self.encoder.load_weights(path + "_A")
        self.encoder.load_weights(path + "_B")
