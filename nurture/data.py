import tensorflow as tf
import tensorflow_datasets as tfds
from attrdict import AttrDict

from tools import get_spec

DEFAULT_DATASET = "mnist"

blacklist = [
    "chexpert", "celeb_a_hq", "bigearthnet", "abstract_reasoning", "celeb_a",
    "amazon_us_reviews", "higgs", "lm1b", "trivia_qa", "wikipedia", "nsynth",
    "wmt14_translate", "wmt15_translate", "wmt16_translate", "wmt17_translate",
    "wmt18_translate", "wmt19_translate", "wmt_t2t_translate", "cats_vs_dogs",
    "bair_robot_pushing_small", "moving_mnist", "starcraft_video", "ucf101",
    "cifar10_corrupted", "clevr", "coco", "coco2014", "kitti", "lsun", "dtd",
    "colorectal_histology_large", "eurosat", "food101", "image_label_folder",
    "imagenet2012", "imagenet2012_corrupted", "mnist_corrupted", "resisc45",
    "open_images_v4", "patch_camelyon", "pet_finder", "quickdraw_bitmap",
    "so2sat", "stanford_dogs", "sun397"]


def get_images(agent, key=DEFAULT_DATASET):
    data, info = tfds.load(key, split="train", with_info=True)
    n_classes = info.features["label"].num_classes

    def unpack(element):
        image, label = element['image'], element['label']
        hw = (agent.image_spec.shape[0], agent.image_spec.shape[1])
        image = tf.image.resize(tf.cast(image, tf.float32), hw)
        label = tf.cast(tf.one_hot(label, n_classes), tf.float32)
        return image, label
    data = data.map(unpack)
    data = data.batch(agent.batch)
    data = data.repeat(5)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    out_specs = [get_spec(n=n_classes, format="onehot")]
    in_specs = [agent.image_spec, agent.image_spec, out_specs[0], agent.loss_spec]
    return AttrDict(
        key=key, data=data, loss=agent.classifier_loss,
        in_specs=in_specs, out_specs=out_specs)
