"""
Enhanced DataLoader for Watermark Training
Handles multiple outputs: [m_out, residual_w, pcm_w] and provides
appropriate targets for watermark loss functions
"""

import numpy as np
from dataloader import LPCNetLoader, lpc2rc
from tensorflow.keras.utils import Sequence

BITS_PER_FRAME = 64


class WatermarkLPCNetLoader(LPCNetLoader):
    """
    Extended LPCNetLoader for watermark training

    Provides targets for:
    - LPCNet synthesis loss (m_out)
    - Watermark embedding loss (pcm vs pcm_w)
    - Watermark extraction loss (bits_in vs extracted_bits)
    """

    def __init__(
        self,
        data,
        features,
        periods,
        batch_size,
        bits_in=None,
        e2e=False,
        lookahead=2,
        phase=1,
    ):
        """
        Args:
            phase: Training phase (1-4) determines output format
        """
        super().__init__(data, features, periods, batch_size, bits_in, e2e, lookahead)
        self.phase = phase

    def __getitem__(self, index):
        """
        Returns data formatted for watermark training phases

        Phase 1: LPCNet + DSS (no attack)
        Phase 2: + DNN extraction (no attack)
        Phase 3: + adversarial training (with attack)
        Phase 4: Full joint training
        """

        # Get base data from parent class
        data = self.data[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size], :, :
        ]
        in_data = data[:, :, :1]  # Input PCM
        out_data = data[:, :, 1:]  # Target PCM

        features = self.features[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size],
            :,
            :-16,
        ]
        periods = self.periods[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size], :, :
        ]
        bits_in = self.bits_in[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size], :, :
        ]

        # Handle LPC coefficients
        if self.lookahead > 0:
            lpc = self.features[
                self.indices[index * self.batch_size : (index + 1) * self.batch_size],
                4 - self.lookahead : -self.lookahead,
                -16:,
            ]
        else:
            lpc = self.features[
                self.indices[index * self.batch_size : (index + 1) * self.batch_size],
                4:,
                -16:,
            ]

        # Prepare inputs - consistent across all phases
        inputs = [in_data, features, periods, bits_in]
        if not self.e2e:
            inputs.append(lpc)

        # Prepare outputs based on training phase
        if self.phase == 1:
            # Phase 1: LPCNet + DSS (no attack)
            # Targets: [m_out_target, residual_w_target, pcm_w_target]
            outputs = [
                out_data,  # LPCNet synthesis target
                None,  # No residual target needed
                in_data,  # Embedding target: pcm_w should match original pcm
            ]

        elif self.phase == 2:
            # Phase 2: + DNN extraction (no attack)
            # Return inputs and separate targets for extractor training
            outputs = [
                out_data,  # LPCNet synthesis target
                None,  # No residual target needed
                in_data,  # Embedding target
                bits_in.reshape(bits_in.shape[0], -1),  # Extraction target (flattened)
            ]

        elif self.phase == 3:
            # Phase 3: + adversarial training (with attack)
            # Add discriminator targets
            batch_size = in_data.shape[0]
            real_labels = np.zeros((batch_size, 1))  # Real speech = 0
            fake_labels = np.ones((batch_size, 1))  # Watermarked = 1

            outputs = [
                out_data,  # LPCNet synthesis target
                None,  # No residual target
                in_data,  # Embedding target
                bits_in.reshape(bits_in.shape[0], -1),  # Extraction target
                real_labels,  # Discriminator target for real
                fake_labels,  # Discriminator target for fake
            ]

        elif self.phase == 4:
            # Phase 4: Full joint training (future implementation)
            outputs = [out_data, None, in_data]

        return (inputs, outputs)


class WatermarkEvaluationLoader(Sequence):
    """
    DataLoader for watermark evaluation and testing
    Loads pre-generated watermarked audio and corresponding metadata
    """

    def __init__(
        self, watermarked_audio_path, original_audio_path, bits_path, batch_size=32
    ):
        """
        Args:
            watermarked_audio_path: Path to .npy file with watermarked audio
            original_audio_path: Path to .npy file with original audio
            bits_path: Path to .npy file with watermark bits
            batch_size: Batch size for evaluation
        """

        self.watermarked_audio = np.load(watermarked_audio_path)
        self.original_audio = np.load(original_audio_path)
        self.bits = np.load(bits_path)
        self.batch_size = batch_size

        # Ensure consistent shapes
        assert self.watermarked_audio.shape[0] == self.original_audio.shape[0]
        assert self.watermarked_audio.shape[0] == self.bits.shape[0]

        self.n_samples = self.watermarked_audio.shape[0]
        self.n_batches = self.n_samples // self.batch_size

        print(f"Evaluation data: {self.n_samples} samples, {self.n_batches} batches")
        print(f"Audio shape: {self.watermarked_audio.shape}")
        print(f"Bits shape: {self.bits.shape}")

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Get batch data
        wm_audio = self.watermarked_audio[start_idx:end_idx]
        orig_audio = self.original_audio[start_idx:end_idx]
        bits = self.bits[start_idx:end_idx]

        # Ensure proper shapes for model input
        if len(wm_audio.shape) == 2:
            wm_audio = np.expand_dims(wm_audio, -1)
        if len(orig_audio.shape) == 2:
            orig_audio = np.expand_dims(orig_audio, -1)
        if len(bits.shape) == 3:
            bits = bits.reshape(bits.shape[0], -1)  # Flatten

        return {
            "watermarked_audio": wm_audio,
            "original_audio": orig_audio,
            "bits": bits,
        }


def create_attack_loader(clean_loader, attack_types=["noise", "compression"]):
    """
    Create a loader that applies attacks to clean data

    Args:
        clean_loader: Base loader providing clean data
        attack_types: List of attack types to apply

    Returns:
        Generator that yields attacked data
    """

    def apply_gaussian_noise(audio, snr_db=20):
        """Apply Gaussian noise at specified SNR"""
        signal_power = np.mean(audio**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise

    def apply_compression(audio, factor=0.8):
        """Apply simple compression by scaling dynamic range"""
        return np.tanh(audio * factor) / factor

    def apply_lowpass_filter(audio, cutoff=0.8):
        """Apply simple lowpass filter (crude approximation)"""
        # Very simple lowpass: moving average
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        if len(audio.shape) == 3:  # (batch, time, channels)
            filtered = np.zeros_like(audio)
            for b in range(audio.shape[0]):
                for c in range(audio.shape[2]):
                    filtered[b, :, c] = np.convolve(audio[b, :, c], kernel, mode="same")
            return filtered
        else:
            return np.convolve(audio, kernel, mode="same")

    attack_functions = {
        "noise": apply_gaussian_noise,
        "compression": apply_compression,
        "lowpass": apply_lowpass_filter,
    }

    for inputs, outputs in clean_loader:
        # Apply random attack
        attack_type = np.random.choice(attack_types)
        attack_func = attack_functions[attack_type]

        # Attack is typically applied to the input audio
        attacked_inputs = list(inputs)
        attacked_inputs[0] = attack_func(inputs[0])  # Attack input PCM

        yield (attacked_inputs, outputs)


if __name__ == "__main__":
    # Test the enhanced dataloader
    print("Testing WatermarkLPCNetLoader...")

    # Create dummy data for testing
    batch_size = 4
    n_samples = 100
    frame_size = 160
    n_frames = 15

    dummy_data = np.random.randn(n_samples, frame_size * n_frames, 2)
    dummy_features = np.random.randn(n_samples, n_frames + 4, 36)
    dummy_periods = np.random.randn(n_samples, n_frames, 1)

    # Test different phases
    for phase in [1, 2, 3]:
        print(f"\n=== Testing Phase {phase} ===")
        loader = WatermarkLPCNetLoader(
            dummy_data, dummy_features, dummy_periods, batch_size, phase=phase
        )

        inputs, outputs = loader[0]
        print(f"Inputs: {len(inputs)} items")
        for i, inp in enumerate(inputs):
            if inp is not None:
                print(f"  Input {i}: {inp.shape}")

        print(f"Outputs: {len(outputs)} items")
        for i, out in enumerate(outputs):
            if out is not None:
                print(f"  Output {i}: {out.shape}")

    print("\nDataLoader test completed!")

