import tensorflow_datasets as tfds
import tensorflow as tf
import nevergrad as ng
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras

# MIN_INPUT_DIM, MAX_INPUT_DIM = 1,2
BATCHES_PER_TEST = 1
BATCH_SIZE = 1
CODE_SIZE = 64


@tf.function
def resize_matrix(flat_in, cppn, matrix_shape, label, tensor):
    flat_matrix = tf.vectorized_map(lambda ij: cppn(
                                    tf.expand_dims(tf.concat([label, ij, tensor], 0), 0)),
                            get_cartesian_product(*matrix_shape))
    return B.dot(flat_in, tf.reshape(flat_matrix, matrix_shape))


@tf.function
def interface(cppn, tensor, out_shape, label):
    in_size = get_size(tf.shape(tensor))
    out_size = get_size(out_shape)
    code = resize_matrix(tf.reshape(tensor, [1, -1]), cppn, (in_size, CODE_SIZE), label, tensor)
    code = resize_matrix(code, cppn, (CODE_SIZE, CODE_SIZE), label, tensor)
    output = resize_matrix(code, cppn, (CODE_SIZE, out_size), label, tensor)
    return tf.reshape(tf.keras.activations.hard_sigmoid(output), out_shape)


@tf.function
def get_cartesian_product(a, b, normalize=True):
    a = tf.range(a)
    a = tf.cast(a, tf.float32)
    b = tf.range(b)
    if normalize:
        a = a / tf.math.reduce_max(a)
        b = b / tf.math.reduce_max(b)
    b = tf.cast(b, tf.float32)
    return tf.reshape(tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1), (-1, 2))


@tf.function
def get_size(shape):
    return tf.foldl(lambda a, d: a * d, tf.convert_to_tensor(shape, dtype=tf.int32))


def test(cppn, mnist, max_batches):
    total_loss = 0.
    for batch_number, batch in mnist.enumerate():
        predictions = tf.concat([interface(cppn,
                               tf.random.normal((tf.constant(1),)),
                               tf.shape(image),
                               tf.cast(label / 9, tf.float32)) for image, label in zip(tf.split(batch["image"], BATCH_SIZE), tf.split(batch["label"], BATCH_SIZE))], 0)
        images = tf.cast(batch["image"], tf.float32) / 255.
        loss = tf.image.ssim(images, predictions, 1.)
        total_loss = total_loss + tf.reduce_mean(loss) * -1.
        if batch_number > max_batches:
            break
    return total_loss / BATCHES_PER_TEST


def get_model():
    input = K.Input((4,))  # label, i, j -> ij_value

    output1 = L.Dense(32, 'hard_sigmoid')(input)
    output2 = L.Dense(32, 'hard_sigmoid')(input)

    output3 = L.Dense(32, 'hard_sigmoid')(input)
    output3 = L.Dense(32, 'hard_sigmoid')(output3)

    output5 = L.Dense(32, 'hard_sigmoid')(input)
    output5 = L.Dense(32, 'hard_sigmoid')(output5)
    output5 = L.Dense(32, 'hard_sigmoid')(output5)
    output5 = L.Dense(32, 'hard_sigmoid')(output5)

    output8 = L.Dense(32, 'hard_sigmoid')(input)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)
    output8 = L.Dense(32, 'hard_sigmoid')(output8)

    output = L.Concatenate()([input, output1, output2, output3, output5, output8]) # 1, 1, 2, 3, 5
    output = L.Dense(1, 'hard_sigmoid')(output)
    return K.Model(input, output)


def main():
    cppn = get_model()
    cppn.summary()
    print("nevergrad optimizers:")
    print(list(sorted(ng.optimizers.registry.keys())))
    print("cppn weight shapes:")
    [print(weight.shape) for weight in cppn.trainable_weights]
    vars = [ng.var.Array(*tuple(weight.shape)).bounded(-10, 10) for weight in cppn.trainable_weights]
    optimizer = ng.optimizers.registry["RotationInvariantDE"](instrumentation=ng.Instrumentation(*vars))
    model_number = 0
    best_loss = 123456789.
    mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN).repeat().shuffle(128).batch(BATCH_SIZE)
    max_batches = BATCHES_PER_TEST - 2
    while True:
        weights = optimizer.ask()
        cppn.set_weights([tf.convert_to_tensor(arg, dtype=tf.float32) for arg in weights.args])
        total_loss = test(cppn, mnist, max_batches).numpy().item()
        optimizer.tell(weights, total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            cppn.save("./best.h5")
        print(model_number, "loss:", total_loss, "best:", best_loss)
        model_number += 1


if __name__ == "__main__":
    main()
