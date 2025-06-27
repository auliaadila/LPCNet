import numpy as np
import tensorflow as tf

# Import your custom layers
from watermark_layers import WatermarkAddition, WatermarkEmbedding


def main():
    # Configuration
    batch_size = 2
    frame_size = 160
    bits_per_frame = 64
    num_frames = 5  # number of frames per example
    total_samples = frame_size * num_frames

    # --- 1) Prepare synthetic inputs -------------------------------------
    # msg_bits: shape (B, F, bits_per_frame), values 0 or 1
    msg_bits = np.random.randint(
        0, 2, size=(batch_size, num_frames, bits_per_frame), dtype=np.int32
    )

    # residual: shape (B, T, 1)
    residual = np.random.randn(batch_size, total_samples, 1).astype(np.float32)

    print("Input msg_bits shape:", msg_bits.shape)
    print("Input residual shape:", residual.shape)

    # --- 2) Instantiate layers ------------------------------------------
    embed_layer = WatermarkEmbedding(
        frame_size=frame_size,
        bits_per_frame=bits_per_frame,
        alpha_init=0.05,
        trainable_alpha=False,
    )
    add_layer = WatermarkAddition(learnable_mask=True, beta=0.1, filters=32)

    # Build the layers by calling once (this also allocates weights)
    wm_component = embed_layer([msg_bits, residual])  # shape (B, T, 1)
    marked_residual = add_layer([residual, wm_component])

    # --- 3) Inspect outputs ---------------------------------------------
    print("Watermark component shape:", wm_component.shape)
    print(
        "Watermark component stats:  min={:.5f}, max={:.5f}, mean={:.5f}".format(
            wm_component.numpy().min(),
            wm_component.numpy().max(),
            wm_component.numpy().mean(),
        )
    )

    print("Marked residual shape:", marked_residual.shape)
    print(
        "Marked residual stats:    min={:.5f}, max={:.5f}, mean={:.5f}".format(
            marked_residual.numpy().min(),
            marked_residual.numpy().max(),
            marked_residual.numpy().mean(),
        )
    )


if __name__ == "__main__":
    # Ensure eager execution (default in TF2)
    tf.config.run_functions_eagerly(True)
    main()
