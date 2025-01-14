import tensorflow as tf

K = tf.keras.backend


def create_encoder(input_size):
    input = tf.keras.layers.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(32, activation='relu')(input)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    # output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=input, outputs=output)


class DualEncoderAll(tf.keras.Model):
    def __init__(self, source_encoder, target_encoder, y_true, **kwargs):
        super(DualEncoderAll, self).__init__(**kwargs)
        self.encoder_A = source_encoder
        self.encoder_B = target_encoder
        self.y_true = y_true
        self.temperature = 0.05
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, train_A=True, train_B=True):
        encodings_A = self.encoder_A(features["A"], training=train_A)
        encodings_B = self.encoder_B(features["B"], training=train_B)
        y = features["Label"]
        self.encoder_A.trainable = train_A
        self.encoder_B.trainable = train_B
        return encodings_A, encodings_B, y

    def compute_loss(self, encodings_A, encodings_B, y):
        encodings_A = tf.squeeze(encodings_A)
        encodings_B = tf.squeeze(encodings_B)
        # pred = (encodings_A + encodings_B)/2
        pred = encodings_A - encodings_B
        y = tf.cast(y, tf.float32)

        loss = tf.math.abs(y - pred)

        # Hinge loss
        # loss = tf.math.maximum(0.0, 1.0 - (y*pred))

        # Absolute hinge-like loss
        # loss = tf.math.abs(1 - (y * pred))

        # Averaged hinge loss
        # loss_A = tf.math.maximum(0.0, 1 - (y*encodings_A))
        # loss_B = tf.math.maximum(0.0, 1 - (y*encodings_B))
        # loss = (loss_A+loss_B)/2

        # Binary cross-entropy loss
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # loss = tf.math.abs(bce(y, pred))

        # Mean-squared error loss
        # mse = tf.keras.losses.MeanSquaredError()
        # loss = tf.math.abs(mse(y, pred))

        return loss

    def train_step(self, feature, train_A=True, train_B=True):
        with tf.GradientTape() as tape:
            # Forward pass
            encodings_A, encodings_B, y = self(feature, train_A=train_A, train_B=train_B)
            loss = self.compute_loss(encodings_A, encodings_B, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, feature):
        encodings_A, encodings_B, y = self(feature, train_A=False, train_B=False)
        loss = self.compute_loss(encodings_A, encodings_B, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        return self.encoder_A(A) - self.encoder_B(B)

    def score(self, input):
        return (self.encoder_A(input) + self.encoder_B(input)) / 2

    def save(self, path):
        self.encoder_A.save_weights(path + "_A")
        self.encoder_B.save_weights(path + "_B")

    def load(self, path):
        self.encoder_A.load_weights(path + "_A")
        self.encoder_B.load_weights(path + "_B")
