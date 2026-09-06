import tensorflow as tf
import numpy as np

# dataset generation

np.set_printoptions(suppress=True, precision=2)

N_DATA_SIZE = 10000

# celsius to Delisle https://en.wikipedia.org/wiki/Delisle_scale conversion (I picked something that I thought had a neat linear relationship).


def celsius_to_delisle(celsius):
    return -1.5 * celsius + 150


SEED = 42
rng = np.random.default_rng(seed=SEED)
dataset_arr = np.round(rng.uniform(low=-273.15, high=260.0, size=N_DATA_SIZE), 2)
delisle_arr = celsius_to_delisle(dataset_arr)
print(f"dataset_arr: {dataset_arr}")
print(f"delisle_arr: {delisle_arr}")


# define model
# also why are we using tensorflow in the year 2026 like 90% of all modern everything uses pytorch now, isn't this suposed to be a modern "advanced machine learning" class?
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])


model.compile(optimizer=tf.keras.optimizers.AdamW(0.1), loss="mean_squared_error")

history = model.fit(dataset_arr, delisle_arr, epochs=500, verbose=False)

# 40c should be 90 Delisle, 0c should be 150 Delisle, and 100c should be 0 Delisle.
test_input = np.array([40.0, 0.0, 100.0])
predictions = model.predict(test_input)
actual = celsius_to_delisle(test_input)

print(f"Test input: {test_input}")
print(f"Predicted Delisle values: {predictions}")
print(f"Actual Delisle values: {actual}")
