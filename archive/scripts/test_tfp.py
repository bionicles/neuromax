import tensorflow_probability as tfp
import tensorflow as tf

K = tf.keras
L = K.layers
tfpl = tfp.layers

IN_SHAPES = [(128, 128, 4), (256, 8), (100,)]
OUT_SHAPES = [(32, 16), (2, ), (4, 8, 16)]
SAMPLE_SHAPES = [1, (1, 1), (1, 2, 3)]
BATCH_SIZES = [1, 2, 4, 8]


def main(append=True):
    for BATCH_SIZE in BATCH_SIZES:
        for IN_SHAPE in IN_SHAPES:
            if append:
                IN_SHAPE = (1,) + IN_SHAPE
            for OUT_SHAPE in OUT_SHAPES:
                print(f"\nBATCH_SIZE {BATCH_SIZE}")
                print(f"IN_SHAPE {IN_SHAPE} ---> OUT_SHAPE {OUT_SHAPE}")
                sensor = K.Sequential([
                    K.Input(shape=IN_SHAPE, batch_size=BATCH_SIZE),
                    L.Dense(tfpl.MixtureNormal.params_size(2, OUT_SHAPE)),
                    tfpl.MixtureNormal(2, OUT_SHAPE)])
                random_in = tf.random.normal(IN_SHAPE)
                out = sensor(random_in)
                print(f"requested out.shape: {OUT_SHAPE}")
                print(f"got out.shape: {out.shape}")
                print(f"got out.batch_shape: {out.batch_shape}")
                print(f"got out.evemt_shape: {out.event_shape}")
                for SAMPLE_SHAPE in SAMPLE_SHAPES:
                    sample = out.sample(SAMPLE_SHAPE)
                    sample = extract_event(sample, SAMPLE_SHAPE, out.batch_shape)
                    print("sample.shape", sample.shape)
                    print("sample.shape == OUT_SHAPE",
                          sample.shape == OUT_SHAPE)


def extract_event(sample, sample_shape, batch_shape):
    len2discard = 1 if isinstance(sample_shape, int) else len(sample_shape)
    if isinstance(batch_shape, int):
        len2discard += 1
    else:
        len2discard += len(batch_shape)
    for _ in range(len2discard):
        sample = sample[0]
    return sample


# warmup to clear the wall of text
x = 1234. * tf.ones((10, 10))
tf.print(x)
del x

main()
main(append=False)
