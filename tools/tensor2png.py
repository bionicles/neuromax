from imageio import imsave
import os
import tensorflow as tf


def convert_tensor_to_image(tensor, image_name, path="."):
    img_array = tf.squeeze(tensor, axis=0).numpy()
    if img_array.shape[-1] is 4:
        save_path = os.path.join(path, image_name+".png")
    else:
        save_path = os.path.join(path, image_name+".jpg")
    imsave(save_path, img_array)
    print(save_path)
