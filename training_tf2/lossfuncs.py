"""
Custom Loss functions and metrics for training/analysis
"""

import tensorflow as tf
from tf_funcs import *

# The following loss functions all expect the lpcnet model to output the lpc prediction


def perceptual_loss(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    # Normalize by dividing by 32768 to bring PCM values to [-1, 1] range
    y_true = y_true / 32768.0
    y_pred = y_pred / 32768.0
    return tf.reduce_mean(tf.square(y_true - y_pred))


def spectral_loss(y_true, y_pred):
    """Spectral loss using STFT magnitude difference"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0
    
    # Handle different tensor shapes - extract only the first channel if multi-channel
    if len(y_true.shape) == 3 and y_true.shape[-1] > 1:
        y_true = y_true[:, :, 0:1]
    if len(y_pred.shape) == 3 and y_pred.shape[-1] > 1:
        y_pred = y_pred[:, :, 0:1]
    
    # Ensure we have 2D tensors for STFT (batch, time)
    if len(y_true.shape) == 3:
        y_true = tf.squeeze(y_true, -1)
    if len(y_pred.shape) == 3:
        y_pred = tf.squeeze(y_pred, -1)
    
    # Compute STFT
    stft_true = tf.signal.stft(y_true, frame_length=512, frame_step=256)
    stft_pred = tf.signal.stft(y_pred, frame_length=512, frame_step=256)
    
    # Magnitude loss
    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)
    
    return tf.reduce_mean(tf.square(mag_true - mag_pred))


def watermark_robustness_loss(y_true, y_pred):
    """Encourages consistent watermark extraction under attacks"""
    # This expects y_true and y_pred to be the extracted bits
    # Penalize differences between original and extracted bits
    return tf.reduce_mean(tf.square(y_true - y_pred))


def snr_loss(clean_signal, watermarked_signal):
    """Signal-to-Noise Ratio loss to control watermark strength"""
    clean_signal = tf.cast(clean_signal, "float32") / 32768.0
    watermarked_signal = watermarked_signal / 32768.0
    
    # Handle different tensor shapes - extract only the first channel if multi-channel
    if len(clean_signal.shape) == 3 and clean_signal.shape[-1] > 1:
        clean_signal = clean_signal[:, :, 0:1]
    if len(watermarked_signal.shape) == 3 and watermarked_signal.shape[-1] > 1:
        watermarked_signal = watermarked_signal[:, :, 0:1]
    
    # Compute noise power
    noise = watermarked_signal - clean_signal
    noise_power = tf.reduce_mean(tf.square(noise))
    
    # Compute signal power  
    signal_power = tf.reduce_mean(tf.square(clean_signal))
    
    # Return negative SNR (we want to maximize SNR, so minimize negative SNR)
    snr = 10 * tf.math.log(signal_power / (noise_power + 1e-8)) / tf.math.log(10.0)
    return -snr


def bit_consistency_loss(original_bits, extracted_bits):
    """Binary cross-entropy with consistency regularization"""
    # Standard BCE
    bce = tf.keras.losses.binary_crossentropy(original_bits, extracted_bits)
    
    # Consistency term - penalize bits that are close to 0.5 (uncertain)
    uncertainty = tf.abs(extracted_bits - 0.5)
    consistency_penalty = tf.reduce_mean(1.0 - 2.0 * uncertainty)  # Higher when bits are near 0.5
    
    return bce + 0.1 * consistency_penalty


def frequency_masking_loss(y_true, y_pred):
    """Perceptual loss with frequency masking (simpler version)"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0
    
    # Compute STFT
    stft_true = tf.signal.stft(tf.squeeze(y_true, -1), frame_length=512, frame_step=256)
    stft_pred = tf.signal.stft(tf.squeeze(y_pred, -1), frame_length=512, frame_step=256)
    
    # Magnitude and phase
    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)
    
    # Weight loss by magnitude (louder frequencies matter more)
    weights = mag_true + 1e-8
    weighted_loss = weights * tf.square(mag_true - mag_pred)
    
    return tf.reduce_mean(weighted_loss) / tf.reduce_mean(weights)


def residual_distribution_loss(y_true, y_pred):
    """Encourage Gaussian-like distribution of watermark residuals"""
    y_pred = tf.cast(y_pred, "float32") / 32768.0
    
    # Compute statistics
    mean = tf.reduce_mean(y_pred)
    var = tf.reduce_mean(tf.square(y_pred - mean))
    
    # Penalize non-zero mean and encourage controlled variance
    mean_penalty = tf.square(mean)
    var_penalty = tf.square(var - 0.01)  # Target small variance
    
    return mean_penalty + var_penalty


def l1_loss(y_true, y_pred):
    """L1 (MAE) loss for time-domain signals"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0
    
    # Handle different tensor shapes - extract only the first channel if multi-channel
    if len(y_true.shape) == 3 and y_true.shape[-1] > 1:
        y_true = y_true[:, :, 0:1]
    if len(y_pred.shape) == 3 and y_pred.shape[-1] > 1:
        y_pred = y_pred[:, :, 0:1]
    
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# Computing the excitation by subtracting the lpc prediction from the target, followed by minimizing the cross entropy
def res_from_sigloss():
    def loss(y_true, y_pred):
        p = y_pred[:, :, 0:1]
        model_out = y_pred[:, :, 2:]
        e_gt = tf_l2u(y_true - p)
        e_gt = tf.round(e_gt)
        e_gt = tf.cast(e_gt, "int32")
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )(e_gt, model_out)
        return sparse_cel

    return loss


# Interpolated and Compensated Loss (In case of end to end lpcnet)
# Interpolates between adjacent embeddings based on the fractional value of the excitation computed (similar to the embedding interpolation)
# Also adds a probability compensation (to account for matching cross entropy in the linear domain), weighted by gamma
def interp_mulaw(gamma=1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, "float32")
        p = y_pred[:, :, 0:1]
        real_p = y_pred[:, :, 1:2]
        model_out = y_pred[:, :, 2:]
        e_gt = tf_l2u(y_true - p)
        exc_gt = tf_l2u(y_true - real_p)
        prob_compensation = tf.squeeze((K.abs(e_gt - 128) / 128.0) * K.log(256.0))
        regularization = tf.squeeze((K.abs(exc_gt - 128) / 128.0) * K.log(256.0))
        alpha = e_gt - tf.math.floor(e_gt)
        alpha = tf.tile(alpha, [1, 1, 256])
        e_gt = tf.cast(e_gt, "int32")
        e_gt = tf.clip_by_value(e_gt, 0, 254)
        interp_probab = (1 - alpha) * model_out + alpha * tf.roll(
            model_out, shift=-1, axis=-1
        )
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )(e_gt, interp_probab)
        loss_mod = sparse_cel + prob_compensation + gamma * regularization
        return loss_mod

    return loss


# Same as above, except a metric
def metric_oginterploss(y_true, y_pred):
    p = y_pred[:, :, 0:1]
    model_out = y_pred[:, :, 2:]
    e_gt = tf_l2u(y_true - p)
    prob_compensation = tf.squeeze((K.abs(e_gt - 128) / 128.0) * K.log(256.0))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha, [1, 1, 256])
    e_gt = tf.cast(e_gt, "int32")
    e_gt = tf.clip_by_value(e_gt, 0, 254)
    interp_probab = (1 - alpha) * model_out + alpha * tf.roll(
        model_out, shift=-1, axis=-1
    )
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(e_gt, interp_probab)
    loss_mod = sparse_cel + prob_compensation
    return loss_mod


# Interpolated cross entropy loss metric
def metric_icel(y_true, y_pred):
    p = y_pred[:, :, 0:1]
    model_out = y_pred[:, :, 2:]
    e_gt = tf_l2u(y_true - p)
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha, [1, 1, 256])
    e_gt = tf.cast(e_gt, "int32")
    e_gt = tf.clip_by_value(e_gt, 0, 254)  # Check direction
    interp_probab = (1 - alpha) * model_out + alpha * tf.roll(
        model_out, shift=-1, axis=-1
    )
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(e_gt, interp_probab)
    return sparse_cel


# Non-interpolated (rounded) cross entropy loss metric
def metric_cel(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    p = y_pred[:, :, 0:1]
    model_out = y_pred[:, :, 2:]
    e_gt = tf_l2u(y_true - p)
    e_gt = tf.round(e_gt)
    e_gt = tf.cast(e_gt, "int32")
    e_gt = tf.clip_by_value(e_gt, 0, 255)

    # print(" =====> METRIC CEL")
    # tf.print("y_true:", y_true.shape)
    # tf.print("y_pred:", y_pred.shape)
    # tf.print("p:",p.shape)
    # tf.print("model_out:", model_out.shape)
    # tf.print("e_ground truth:", e_gt.shape)

    #  =====> METRIC CEL
    # y_true: TensorShape([None, None, None])
    # y_pred: TensorShape([128, 2400, 258])
    # p: TensorShape([128, 2400, 1])
    # model_out: TensorShape([128, 2400, 256])
    # e_ground truth: TensorShape([128, 2400, None])

    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(e_gt, model_out)
    return sparse_cel


# Variance metric of the output excitation
def metric_exc_sd(y_true, y_pred):
    p = y_pred[:, :, 0:1]
    e_gt = tf_l2u(y_true - p)
    sd_egt = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(
        e_gt, 128
    )
    return sd_egt


def loss_matchlar():
    def loss(y_true, y_pred):
        model_rc = y_pred[:, :, :16]
        # y_true = lpc2rc(y_true)
        loss_lar_diff = K.log((1.01 + model_rc) / (1.01 - model_rc)) - K.log(
            (1.01 + y_true) / (1.01 - y_true)
        )
        loss_lar_diff = tf.square(loss_lar_diff)
        return tf.reduce_mean(loss_lar_diff, axis=-1)

    return loss
