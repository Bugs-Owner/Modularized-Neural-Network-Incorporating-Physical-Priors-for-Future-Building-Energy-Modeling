import imageio
import glob
import os

image_path = '../Saved/Training_Image/5_31'
image_list = sorted(glob.glob(image_path+'/*.png') , key=os.path.getmtime)
images = [imageio.imread(filename) for filename in image_list]
imageio.mimsave(os.path.join(image_path, '403baseline.gif'), images, fps=1.5)
