import tensorflow_datasets as tfds
from attrdict import AttrDict
import tensorflow as tf
import random

from tools import get_spec, log
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


def prepare_data(agent):
    data_key, loss_fn = "chexpert", None
    datasets_list = tfds.list_builders()
    while data_key in blacklist:
        if len(blacklist) is len(list):
            return "fuck"
        data_key = random.choice(datasets_list)
        try:
            data, info = tfds.load(data_key, split="train", with_info=True)
            features = info.features
            fkeys = features.keys()
            if "image" not in fkeys and "label" not in fkeys:
                raise Exception("not simple")
            in_specs, out_specs = [agent.image_spec], []
            for fkey in fkeys:
                feature = features[fkey]
                log(fkey, feature, color="red")
                if isinstance(feature, tfds.features.ClassLabel):
                    loss_fn = tf.keras.losses.categorical_crossentropy
                    out_spec = get_spec(n=feature.num_classes, format="onehot")
                    out_specs.append(out_spec)
            if not loss_fn:
                raise Exception("not simple")

            def unpack(e):
                return e['image'], e['label']
            data_env = DataEnv(data, unpack, loss_fn)
            return AttrDict(in_specs=in_specs, out_spec=out_specs,
                            env=data_env, key=data_key, is_data=True)

        except Exception as e:
            log(data_key, e, color="red")
            blacklist.append(data_key)


class DataEnv:

    def __init__(self, dataset, unpack, loss_fn):
        self.dataset = dataset
        self.iter = self.dataset.__iter__()
        self.loss_fn = loss_fn

    def reset(self):
        element = self.iter.next()
        inputs, self.y_true = self.unpack(element)
        return inputs

    def step(self, y_pred):
        loss = self.loss_fn(self.y_true, y_pred)
        return "nothing", -loss, True, "nothing"
