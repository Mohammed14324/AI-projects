def combined_loss(y_true, y_pred, smooth=1e-6):
    import tensorflow as tf
    import tensorflow.keras.backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice

    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice_loss + K.mean(cce)