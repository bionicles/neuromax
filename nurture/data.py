import tensorflow_datasets as tfds
from attrdict import AttrDict
from tensorflow import cast, float32, expand_dims

from tools import get_spec, log, get_onehot

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


def prepare_data(agent):
    log("prepare_data", color="red")
    data_key, loss_fn = DEFAULT_DATASET, agent.classifier_loss
    data, info = tfds.load(data_key, split="train", with_info=True)
    data = data.batch(agent.batch_size)
    data = data.prefetch(2)
    features = info.features

    out_spec = get_spec(n=features["label"].num_classes, format="onehot")
    in_specs, out_specs = [agent.image_spec], [out_spec]
    options = list(range(10))

    def unpack(element):
        image, label = element['image'], element['label']
        image = cast(image, float32)
        label = get_onehot(int(label), options)
        label = cast(label, float32)
        label = expand_dims(label, 0)
        return image, label
    data_env = DataEnv(data, unpack, loss_fn)
    log("data_env", data_env, color="red")
    return AttrDict(
        in_specs=in_specs, out_specs=out_specs,
        env=data_env, key=data_key, is_data=True)


class DataEnv:

    def __init__(self, dataset, unpack, loss_fn):
        self.dataset = dataset
        self.iter = self.dataset.__iter__()
        self.loss_fn = loss_fn
        self.unpack = unpack

    def reset(self):
        element = self.iter.next()
        inputs, self.y_true = self.unpack(element)
        return inputs

    def step(self, y_pred):
        log("DataEnv.step y_true", self.y_true, "y_pred", y_pred)
        loss = self.loss_fn(self.y_true, y_pred)
        return None, -loss, True, None
