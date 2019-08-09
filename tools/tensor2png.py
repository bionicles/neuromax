from imageio import imsave
import os 
import tensorflow as tf

def convert_tensor_to_image(tensor, image_name, path="."):
    img_array = tf.squeeze(tensor, axis=0).numpy()
    save_path = os.path.join(path, image_name+".png") if img_array.shape[-1] == "4" else os.path.join(path, image_name+".jpg") 
    imsave(save_path, img_array)
    print(save_path)