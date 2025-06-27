import numpy as np
from tensorflow.keras.utils import Sequence
from ulaw import lin2ulaw


def lpc2rc(lpc):
    # print("shape is = ", lpc.shape)
    order = lpc.shape[-1]
    rc = 0 * lpc
    for i in range(order, 0, -1):
        rc[:, :, i - 1] = lpc[:, :, -1]
        ki = rc[:, :, i - 1 : i].repeat(i - 1, axis=2)
        lpc = (lpc[:, :, :-1] - ki * lpc[:, :, -2::-1]) / (1 - ki * ki)
    return rc


BITS_PER_FRAME = 64


class LPCNetLoader(Sequence):
    def __init__(
        self, data, features, periods, batch_size, bits_in=None, e2e=False, lookahead=2
    ):
        self.batch_size = batch_size
        self.nb_batches = (
            np.minimum(np.minimum(data.shape[0], features.shape[0]), periods.shape[0])
            // self.batch_size
        )
        self.data = data[: self.nb_batches * self.batch_size, :]
        self.features = features[: self.nb_batches * self.batch_size, :]
        self.periods = periods[: self.nb_batches * self.batch_size, :]
        self.e2e = e2e
        self.lookahead = lookahead
        self.bps = BITS_PER_FRAME
        # placeholder – real bits are created in on_epoch_end()
        self.bits_table = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches * self.batch_size)
        np.random.shuffle(self.indices)

        # new random bit table for the forthcoming epoch
        rng = np.random.default_rng()
        self.bits_table = rng.integers(
            0, 2, size=(self.nb_batches * self.batch_size, self.bps), dtype="int32"
        )

    def __getitem__(self, index):
        data = self.data[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size], :, :
        ]
        in_data = data[:, :, :1]
        out_data = data[:, :, 1:]
        features = self.features[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size],
            :,
            :-16,
        ]
        periods = self.periods[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size], :, :
        ]

        actual_batch_size = len(data)

        bits_in = self.bits_table[
            self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        ]

        dummy = np.zeros_like(in_data, dtype="float32")
        outputs = {
            "pdf": out_data,
            "residual_w": dummy,  # No target needed - use regularization loss only
            "pcm_w": in_data,  # Target: clean PCM (compare watermarked vs clean)
            "bits_pred": bits_in,  # Target: original bits
        }
        inputs = [in_data, features, periods, bits_in]
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
        if self.e2e:
            outputs["pdf"] = [out_data, lpc2rc(lpc)]
        else:
            inputs.append(lpc)
        return (inputs, outputs)

    def __len__(self):
        return self.nb_batches
