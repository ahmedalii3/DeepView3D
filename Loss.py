import tensorflow as tf

class masked_loss:
    def __init__(self, mask_value=0.0):
        self.mask_value = mask_value

    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.cast(y_pred, tf.float32)

        # Valid pixels only

        mask = tf.math.is_finite(y_true)

        mask = tf.logical_and(mask, y_true > 0.0)

        y_true = tf.where(mask, y_true, tf.zeros_like(y_true))

        y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))

        diff = tf.abs(y_true - y_pred)

        # Smooth L1 / Huber with delta = 1

        loss = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)

        mask = tf.cast(mask, tf.float32)

        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)
    
    def get_config(self):
        return {'mask_value': self.mask_value}