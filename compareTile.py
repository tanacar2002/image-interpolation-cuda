import numpy as np
import cv2
import matplotlib.pyplot as plt

files = ["./samples/2_gpu_tex_nn_out.jpg", "./samples/2_gpu_tex_lin_out.jpg", "./samples/2_gpu_tex_cub_v2_out.jpg", "./samples/2_gpu_tex_lan_v3_out.jpg"]

ROI = [7000, 16000, 500, 500]
zooms = [2000, 500, 100]
imgs = []
for file in files:
    imgs.append(cv2.imread(file)[:,:,::-1]) #ROI[0]:ROI[0]+ROI[2], ROI[1]:ROI[1]+ROI[3],

print(imgs[0].shape)

fig, axs = plt.subplots(len(zooms),len(imgs))
for i, zoom in enumerate(zooms):
    for j, img in enumerate(imgs):
        axs[i,j].imshow(imgs[j][ROI[0]-zoom:ROI[0]+zoom, ROI[1]-zoom:ROI[1]+zoom,:])
        axs[i,j].axis("off")
        axs[i,j].set_aspect('equal')
        

axs[0,0].set_title('Nearest N.')
axs[0,1].set_title('Bilinear')
axs[0,2].set_title('Bicubic')
axs[0,3].set_title('Lanczos')

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()