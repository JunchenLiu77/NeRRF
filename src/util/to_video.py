import cv2
import os
from IPython import embed
import imageio
from skimage.transform import resize

# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def img2video(name, fps=40):
    image_ori = cv2.imread(name + "/50.png")
    video_size = (image_ori.shape[1], image_ori.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video.mp4", fourcc, fps, video_size, True)
    list_file = os.listdir(name)

    # list_file.sort(key=lambda x:int(x[:-9]))
    # list_file = list_file[50:110]

    list_file.sort(key=lambda x: int(x[:-4]))
    for i in list_file:
        filename = os.path.join(name, i)
        frame = cv2.imread(filename)
        print(filename)
        video.write(frame)
    video.release()


img2video("visuals/round1")
# img2video('./dataloader/tmp/blender_indoor_bunny/image')


# root = './dataloader/tmp/evals/'

# cur = 'd'

# psnr=.0
# ssim=.0
# for i in range(1,3):
#     img_gt = imageio.imread(root+cur+'_gt_'+str(i)+'.jpg')
#     img = imageio.imread(root+cur+'_'+str(i)+'.jpg')
#     img = resize(img,(270,480,3))
#     img_gt = resize(img_gt,(270,480,3))
#     psnr += compare_psnr(img,img_gt,data_range=1)
#     ssim += compare_ssim(img,img_gt,multichannel=True, data_range=1)
#     #print(psnr)
#     #print(ssim)

# print(psnr/2)
# print(ssim/2)
