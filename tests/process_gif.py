from PIL import Image
import numpy as np


def test_processImage():
    file_name = "images_paper/charleston_93_04_160_3_0.gif"

    num_key_frames = 10
    im = Image.open(file_name)
    transparency = im.info['transparency']
    p = im.getpalette()
    frames = im.n_frames

    with Image.open(file_name) as im:
        for i in range(num_key_frames):
            # if not im.getpalette():
            #    im.putpalette(list(np.uint32(p)))

            im.seek(frames // num_key_frames * i)
            im.convert('P').save('images_paper/charleston_{}.pdf'.format(i),
                                 palette=list(np.uint32(p)))
