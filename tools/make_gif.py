from datetime import datetime
from natsort import natsorted
import imageio
import os

from .debug import log


def make_gif(folder_path="archive/nets", out_path="archive/movies",
             image_x=1024, image_y=1024, brand_y=128,
             brand="Bit Pharma", movie=True, framerate=24.):
    image_size = f"{image_x}x{image_y}"
    dirfiles = []
    for filename in os.listdir(folder_path):
        log("parsed", filename)
        name, ext = os.path.splitext(filename)
        if name == 'model' or ext != ".png":
            continue
        log("convert", filename, "to jpg")
        cmd_string = f"""
                convert '{folder_path}/{name}.png'
                -gravity Center -thumbnail '{image_size}' -extent {image_size}
                -font Courier -gravity Center -undercolor '#00000080'
                -size {image_x}x{brand_y} -fill white caption:'{brand}'
                +swap -append -gravity Center '{folder_path}/{name}.jpg'
                """
        os.system(cmd_string)
        dirfiles.append(f"{folder_path}/{name}.jpg")
    dirfiles = natsorted(dirfiles)
    [log(dirfile) for dirfile in dirfiles]
    now = str(datetime.now()).replace(" ", "_")
    if movie:
        movie_path = f"archive/movies/net-{now}.mp4"
        writer = imageio.get_writer(movie_path, fps=framerate)
        [writer.append_data(imageio.imread(dirfile)) for dirfile in dirfiles]
        writer.close()
        log(f"SAVED MOVIE TO {movie_path}")
    else:
        images = [imageio.imread(dirfile) for dirfile in dirfiles]
        gif_path = os.path.join(out_path, f"net-{now}.gif")
        imageio.mimsave(gif_path, images, duration=(1/framerate))
        log(f"SAVED GIF TO {gif_path}")
