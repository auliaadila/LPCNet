## TO DO: Check how it compares with train_lpcnet
#!/usr/bin/env python3
"""
Stage-1: fine-tune LPCNet to cope with α·e·m perturbations.
* Loads your patched model from `joint/model_joint.py`
* Streams data from embed.tfrecord
* Freezes most layers (keeps 'alpha_dense' trainable)
* Saves `checkpoints/lpcnet_embed.h5`
"""

import os, argparse, tensorflow as tf
from tensorflow.keras.optimizers import Adam

import joint.model_joint as mj          # your patched model
from lpcnet import decoder_from_model   # optional sanity check

CHUNK = 2400
BATCH = 32
EPOCHS = 3

# --- TFRecord parsing -------------------------------------------------- #
def _parse(serialized):
    spec = {
        'pcm'     : tf.io.FixedLenFeature([], tf.string),
        'feat'    : tf.io.FixedLenFeature([], tf.string),
        'pitch'   : tf.io.FixedLenFeature([], tf.string),
        'lpcoeffs': tf.io.FixedLenFeature([], tf.string),
        'pn'      : tf.io.FixedLenFeature([], tf.string),
        'sigma'   : tf.io.FixedLenFeature([], tf.string),
        'bits'    : tf.io.FixedLenFeature([], tf.string)
    }
    ex = tf.io.parse_single_example(serialized, spec)
    pcm   = tf.io.decode_raw(ex['pcm'], tf.float32)
    feat  = tf.reshape(tf.io.decode_raw(ex['feat'], tf.float32),  [-1,20])
    pitch = tf.reshape(tf.io.decode_raw(ex['pitch'], tf.float32), [-1,1])
    lpc   = tf.reshape(tf.io.decode_raw(ex['lpcoeffs'], tf.float32),[-1,16])
    pn    = tf.reshape(tf.io.decode_raw(ex['pn'], tf.int8),        [-1,1])
    sigma = tf.reshape(tf.io.decode_raw(ex['sigma'], tf.float32),  [-1,1])
    bits  = tf.reshape(tf.io.decode_raw(ex['bits'], tf.int8),      [-1,64])

    # LPCNet expects shape (batch, time, feat)
    return (pcm, feat, pitch, lpc, tf.cast(pn, tf.float32),
            sigma, tf.cast(bits, tf.float32)), None  # m_out is internal

def make_dataset(path, batch=BATCH):
    ds = tf.data.TFRecordDataset(path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1024).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrecord', required=True,
                    help='embed.tfrecord produced by gen_dataset.py')
    ap.add_argument('--baseline', default='checkpoints/lpcnet_clean.h5')
    ap.add_argument('--out',      default='checkpoints/lpcnet_embed.h5')
    args = ap.parse_args()

    dataset = make_dataset(args.tfrecord)

    model, _, _ = mj.new_lpcnet_model(
        rnn_units1=384,
        rnn_units2=16,
        nb_used_features=20,
        batch_size=BATCH,
        training=True,   # IMPORTANT
        adaptation=False,
        quantize=False,
        flag_e2e=False
    )
    model.load_weights(args.baseline)
    print('Loaded baseline weights:', args.baseline)

    # freeze everything except alpha_dense
    for layer in model.layers:
        if layer.name != 'alpha_dense':
            layer.trainable = False
    model.get_layer('alpha_dense').trainable = True
    model.summary(line_length=95)

    # categorical CE is already attached inside m_out
    model.compile(optimizer=Adam(1e-4))
    model.fit(dataset, epochs=EPOCHS)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print('Saved fine-tuned model:', args.out)

if __name__ == '__main__':
    main()
