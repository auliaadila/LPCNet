"""
Custom Loss functions and metrics for training/analysis
"""

from tf_funcs import *
import tensorflow as tf

# The following loss functions all expect the lpcnet model to output the lpc prediction

# Computing the excitation by subtracting the lpc prediction from the target, followed by minimizing the cross entropy
def res_from_sigloss():
    def loss(y_true,y_pred):
        p = y_pred[:,:,0:1]
        model_out = y_pred[:,:,2:]
        e_gt = tf_l2u(y_true - p)
        e_gt = tf.round(e_gt)
        e_gt = tf.cast(e_gt,'int32')
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,model_out)
        return sparse_cel
    return loss

# Interpolated and Compensated Loss (In case of end to end lpcnet)
# Interpolates between adjacent embeddings based on the fractional value of the excitation computed (similar to the embedding interpolation)
# Also adds a probability compensation (to account for matching cross entropy in the linear domain), weighted by gamma
def interp_mulaw(gamma = 1):
    def loss(y_true,y_pred):
        y_true = tf.cast(y_true, 'float32')
        p = y_pred[:,:,0:1]
        real_p = y_pred[:,:,1:2]
        model_out = y_pred[:,:,2:]
        e_gt = tf_l2u(y_true - p)
        exc_gt = tf_l2u(y_true - real_p)
        prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
        regularization = tf.squeeze((K.abs(exc_gt - 128)/128.0)*K.log(256.0))
        alpha = e_gt - tf.math.floor(e_gt)
        alpha = tf.tile(alpha,[1,1,256])
        e_gt = tf.cast(e_gt,'int32')
        e_gt = tf.clip_by_value(e_gt,0,254) 
        interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
        loss_mod = sparse_cel + prob_compensation + gamma*regularization
        return loss_mod
    return loss

# Same as above, except a metric
def metric_oginterploss(y_true,y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,2:]
    e_gt = tf_l2u(y_true - p)
    prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) 
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    loss_mod = sparse_cel + prob_compensation
    return loss_mod

# Interpolated cross entropy loss metric
def metric_icel(y_true, y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,2:]
    e_gt = tf_l2u(y_true - p)
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) #Check direction
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    return sparse_cel

# Non-interpolated (rounded) cross entropy loss metric
def metric_cel(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,2:]
    e_gt = tf_l2u(y_true - p)
    e_gt = tf.round(e_gt)
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,255) 
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,model_out)
    return sparse_cel

# Variance metric of the output excitation
def metric_exc_sd(y_true,y_pred):
    p = y_pred[:,:,0:1]
    e_gt = tf_l2u(y_true - p)
    sd_egt = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(e_gt,128)
    return sd_egt

def loss_matchlar():
    def loss(y_true,y_pred):
        model_rc = y_pred[:,:,:16]
        #y_true = lpc2rc(y_true)
        loss_lar_diff = K.log((1.01 + model_rc)/(1.01 - model_rc)) - K.log((1.01 + y_true)/(1.01 - y_true))
        loss_lar_diff = tf.square(loss_lar_diff)
        return tf.reduce_mean(loss_lar_diff, axis=-1)
    return loss


# ==================================
# Watermark Loss Functions
# ==================================


def watermark_extraction_loss():
    """
    Watermark extraction loss: L_w = (1/N) * Σ(w - w_o)²
    Measures how well the extractor can recover the original watermark bits
    """

    def loss(y_true, y_pred):
        """
        Args:
            y_true: Original watermark bits (N,) - values in {0,1}
            y_pred: Extracted watermark bits (N,) - values in [0,1]
        """
        # Convert to same data type
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # MSE between original and extracted bits
        extraction_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return extraction_loss

    return loss


def watermark_embedding_loss():
    """
    Watermark embedding loss: L_se = (1/M) * Σ((s_w)_i - s_o)²
    Ensures watermarked signal is perceptually similar to original
    """

    def loss(y_true, y_pred):
        """
        Args:
            y_true: Original clean audio signal (M,)
            y_pred: Watermarked audio signal (M,)
        """
        # MSE between original and watermarked audio
        embedding_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return embedding_loss

    return loss


def adversarial_generator_loss():
    """
    Adversarial loss for generator: L_adv = -log(σ(D(s_w)))
    Encourages watermarked speech to fool the discriminator
    """

    def loss(y_true, y_pred):
        """
        Args:
            y_true: Not used (placeholder)
            y_pred: Discriminator output for watermarked speech [0,1]
        """
        # We want discriminator to think watermarked speech is real (close to 0)
        # So we minimize -log(1 - D(s_w)) = maximize log(1 - D(s_w))
        eps = 1e-7  # Small epsilon to avoid log(0)
        adv_loss = -tf.reduce_mean(tf.math.log(1 - y_pred + eps))
        return adv_loss

    return loss


def adversarial_discriminator_loss():
    """
    Discriminator loss: L_d = -log(σ(D(s))) - log(1 - σ(D(s_w)))
    Train discriminator to distinguish real from watermarked speech
    """

    def loss(y_true, y_pred):
        """
        Args:
            y_true: Ground truth labels - 0 for real, 1 for watermarked
            y_pred: Discriminator predictions [0,1]
        """
        eps = 1e-7
        y_true = tf.cast(y_true, tf.float32)

        # Binary cross-entropy loss
        disc_loss = -(
            y_true * tf.math.log(y_pred + eps)
            + (1 - y_true) * tf.math.log(1 - y_pred + eps)
        )
        return tf.reduce_mean(disc_loss)

    return loss


def combined_watermark_loss(alpha_extract=1.0, alpha_embed=0.1, alpha_adv=0.01):
    """
    Combined loss function for joint watermark training

    Args:
        alpha_extract: Weight for extraction loss
        alpha_embed: Weight for embedding loss
        alpha_adv: Weight for adversarial loss
    """
    extract_loss_fn = watermark_extraction_loss()
    embed_loss_fn = watermark_embedding_loss()
    adv_loss_fn = adversarial_generator_loss()

    def loss(y_true, y_pred):
        """
        Args:
            y_true: [original_bits, original_audio, disc_dummy]
            y_pred: [extracted_bits, watermarked_audio, disc_output]
        """
        # Unpack inputs
        orig_bits, orig_audio, _ = y_true
        extract_bits, wm_audio, disc_out = y_pred

        # Compute individual losses
        L_extract = extract_loss_fn(orig_bits, extract_bits)
        L_embed = embed_loss_fn(orig_audio, wm_audio)
        L_adv = adv_loss_fn(None, disc_out)

        # Weighted combination
        total_loss = (
            alpha_extract * L_extract + alpha_embed * L_embed + alpha_adv * L_adv
        )

        return total_loss

    return loss


# Metrics for monitoring watermark performance
def bit_error_rate_metric(y_true, y_pred):
    """
    Compute Bit Error Rate (BER) for watermark extraction
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)  # Round to {0,1}

    # BER = (number of wrong bits) / (total bits)
    wrong_bits = tf.reduce_sum(tf.abs(y_pred - y_true))
    total_bits = tf.cast(tf.size(y_true), tf.float32)
    ber = wrong_bits / total_bits
    return ber


def signal_to_noise_ratio_metric(y_true, y_pred):
    """
    Compute SNR between original and watermarked audio
    """
    # SNR = 10 * log10(||s||² / ||s-s_w||²)
    signal_power = tf.reduce_mean(tf.square(y_true))
    noise_power = tf.reduce_mean(tf.square(y_true - y_pred))

    snr_db = 10 * tf.math.log(signal_power / (noise_power + 1e-10)) / tf.math.log(10.0)
    return snr_db


def perceptual_embedding_loss(alpha_mse=1.0, alpha_spectral=0.5, alpha_temporal=0.3):
    """
    Perceptual loss for watermark embedding that goes beyond simple MSE
    Improves training of WatermarkAddition learnable mask and overall quality

    Args:
        alpha_mse: Weight for time-domain MSE loss
        alpha_spectral: Weight for spectral consistency loss
        alpha_temporal: Weight for temporal structure preservation
    """

    def loss(y_true, y_pred):
        """
        Args:
            y_true: Original clean audio (pcm)
            y_pred: Watermarked audio (pcm_w)
        """
        # 1. Time-domain MSE loss (basic embedding quality)
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))

        # 2. Spectral consistency loss (preserve frequency content)
        # Use first-order differences as crude spectral approximation
        y_true_diff = y_true[:, 1:] - y_true[:, :-1]
        y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        spectral_loss = tf.reduce_mean(tf.square(y_pred_diff - y_true_diff))

        # 3. Temporal structure preservation (maintain signal dynamics)
        # Use second-order differences to capture temporal patterns
        if tf.shape(y_true_diff)[1] > 1:
            y_true_diff2 = y_true_diff[:, 1:] - y_true_diff[:, :-1]
            y_pred_diff2 = y_pred_diff[:, 1:] - y_pred_diff[:, :-1]
            temporal_loss = tf.reduce_mean(tf.square(y_pred_diff2 - y_true_diff2))
        else:
            temporal_loss = 0.0

        # Weighted combination for perceptual quality
        total_loss = (
            alpha_mse * mse_loss
            + alpha_spectral * spectral_loss
            + alpha_temporal * temporal_loss
        )

        return total_loss

    return loss
