import tensorflow as tf

# Run eagerly so you see the real error (very useful!)
tf.config.run_functions_eagerly(True)

# Then compile the model again
model.compile(optimizer=opt, loss=[...], metrics=[...])

# Then fit
model.fit(loader, epochs=nb_epochs, validation_split=0.0, callbacks=callbacks)