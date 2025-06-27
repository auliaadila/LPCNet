# extractor_broadcast.py  (time axis now flexible)

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    ReLU,
)
from tensorflow.keras.models import Model


def res_block(x, filters, k, name):
    y = Conv1D(filters, k, padding="same", groups=filters, name=f"{name}_dw")(x)
    y = BatchNormalization(name=f"{name}_bn1")(y)
    y = ReLU(name=f"{name}_relu1")(y)
    y = Conv1D(filters, 1, padding="same", name=f"{name}_pw")(y)
    y = BatchNormalization(name=f"{name}_bn2")(y)
    return ReLU(name=f"{name}_out")(Add()([x, y]))


def build_chunk_extractor(
    time_len=None,  # None  → accept any length
    filters=(32, 64, 128, 128),
    kernel_size=7,
    dropout=0.3,
    bits=64,
):
    """
    (B, time_len, 1) waveform  →  (B, bits) posteriors
    Default time_len=None means the model works with 2400-sample
    chunks during training **and** accepts other lengths at test time.
    """

    audio = Input(shape=(time_len, 1), name="pcm_in")  # time_len=None OK

    # ── stem ──────────────────────────────────────────────
    x = Conv1D(filters[0], kernel_size, padding="same", name="stem_conv")(audio)
    x = BatchNormalization(name="stem_bn")(x)
    x = ReLU(name="stem_relu")(x)

    # ── residual stack + strided pooling ─────────────────
    for i, f in enumerate(filters):
        x = res_block(x, f, kernel_size, name=f"res{i + 1}")
        if i < len(filters) - 1:
            x = AveragePooling1D(2, name=f"pool{i + 1}")(x)  # ↓2 each stage

    # ── global stats over the whole chunk ────────────────
    x = GlobalAveragePooling1D(name="global_pool")(x)  # (B, filters[-1])

    # ── MLP head ──────────────────────────────────────────
    x = Dense(128, activation="relu", name="dense1")(x)
    x = Dropout(dropout)(x)
    logits = Dense(bits, activation="sigmoid", name="bits")(x)  # (B, bits)

    return Model(audio, logits, name="extractor_64_per_chunk")


# smoke-test with a batch that is (128, 2400, 1)
if __name__ == "__main__":
    model = build_chunk_extractor()  # time_len=None
    model.summary()

    dummy = tf.zeros((128, 2400, 1))
    out = model(dummy)
    print("output shape:", out.shape)  # (128, 64)


# # Adapted from Pavlovic

# import tensorflow as tf
# from tensorflow.keras.layers import (Input, Conv1D, Conv2D, BatchNormalization,
#                                      ReLU, LeakyReLU, Flatten, Add, AveragePooling1D,
#                                      GlobalAveragePooling1D,
#                                      Dense, Dropout)
# from tensorflow.keras.models import Model

# BITS_PER_FRAME = 64

# def create_extractor(time_len=2400,
#                      bits_per_frame=BITS_PER_FRAME):
#     # time_len = 2400 if Flatten
#     # time_len = None if Flatten or GlobalAveragePool

#     audio = Input(shape=(time_len, 1), name="pcm_in")

#     dconv_1 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(audio)
#     dconv_1 = BatchNormalization()(dconv_1)
#     dconv_1 = LeakyReLU(alpha=0.2)(dconv_1)

#     dconv_2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(dconv_1)
#     dconv_2 = BatchNormalization()(dconv_2)
#     dconv_2 = LeakyReLU(alpha=0.2)(dconv_2)

#     dconv_3 = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(dconv_2)
#     dconv_3 = BatchNormalization()(dconv_3)
#     dconv_3 = LeakyReLU(alpha=0.2)(dconv_3)

#     dconv_4 = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(dconv_3)
#     dconv_4 = BatchNormalization()(dconv_4)
#     dconv_4 = LeakyReLU(alpha=0.2)(dconv_4)

#     dconv_5 = Conv1D(filters=128, kernel_size=5, strides=2, padding='same')(dconv_4)
#     dconv_5 = BatchNormalization()(dconv_5)
#     dconv_5 = LeakyReLU(alpha=0.2)(dconv_5)

#     dconv_6 = Conv1D(filters=128, kernel_size=5, strides=2, padding='same')(dconv_5)
#     dconv_6 = BatchNormalization()(dconv_6)
#     dconv_6 = LeakyReLU(alpha=0.2)(dconv_6)
#     print("dconv_6:", dconv_6.shape)

#     flatten = Flatten()(dconv_6)
#     # flatten = GlobalAveragePooling1D(name="global_pool")(dconv_6)

#     logits = Dense(bits_per_frame, activation="sigmoid", name="bits")(flatten)  # (B, bits)
#     print("logits:", logits.shape)

#     return Model(audio, logits, name="wm_extractor")

# if __name__ == "__main__":
#     model = create_extractor()          # time_len=None
#     model.summary()

#     dummy = tf.zeros((128, 2400, 1))
#     out   = model(dummy)
#     print("output shape:", out.shape)        # (128, 64)
