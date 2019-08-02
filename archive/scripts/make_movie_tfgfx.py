import tensorflow_graphics as gfx
import tensorflow as tf

lambertian = gfx.rendering.reflectance.lambertian
phong = gfx.rendering.reflectance.phong


def make_movie(keyframes, length, save_path):
    frames_per_keyframe = length / tf.shape(keyframes)[0]
    frames = interpolate(keyframes, frames_per_keyframe)
    movie = tf.vectorized_map(draw_frame, frames)
    gfx.tensor2movie(movie, save_path)


# generate frames from keyframes
def interpolate(keyframes, frames_per_keyframe):
    return tf.unnest(tf.vectorized_map(
        lambda i: interpolate_pair(keyframes, i, frames_per_keyframe),
        tf.range(len(keyframes))))


def interpolate_pair(keyframes, n, frames_per_keyframe):
    key_1, key_2 = keyframes[n - 1], keyframes[n]
    d_v_d_i = (key_1 - key_2) / frames_per_keyframe
    return tf.vectorized_map(
        lambda d_i: translate_objects(key_1, d_v_d_i * d_i),
        tf.range(frames_per_keyframe))


def translate_objects(xyz, v):
    return tf.vectorized_map(
        lambda i: xyz[i] + v[i],
        tf.range(tf.shape(xyz)[0]))


# draw the frame and project it onto a camera plane
def draw_frame(xyz):
    atoms = tf.vectorized_map(draw_atom, xyz)
    camera = gfx.zoom_to_fit()
    return gfx.project(atoms, camera)


def draw_atom(xyz):
    return gfx.geometry.sphere(xyz, radiance=[lambertian, phong])
