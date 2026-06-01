
import tensorflow as tf
from tensorflow.keras.losses import Loss

class MaskedLoss(Loss):
    def __init__(self, mask_value=0.0, name="masked_loss"):
        super(MaskedLoss, self).__init__(name=name)
        self.mask_value = mask_value

    # <-- IMPORTANT: override `call`, NOT __call__
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 3:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 3:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Valid pixels only
        mask = tf.math.is_finite(y_true)
        mask = tf.logical_and(mask, y_true > 0.0)
        mask = tf.cast(mask, tf.float32)

        y_true = tf.where(mask > 0, y_true, tf.zeros_like(y_true))
        y_pred = tf.where(mask > 0, y_pred, tf.zeros_like(y_pred))

        diff = tf.abs(y_true - y_pred)
        diff = diff * 2.0
        # Smooth L1 / Huber with delta = 1
        loss = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)

        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)

    def get_config(self):
        return {'mask_value': self.mask_value}