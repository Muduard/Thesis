import os
import cv2
from metric_utils import mse,psnr
from skimage.metrics import peak_signal_noise_ratio as psnr2
from skimage.metrics import mean_squared_error as mse2
from image_similarity_measures.quality_metrics import ssim
import matplotlib.pyplot as plt
import numpy as np
insfolder = "./results/tesi_reg" #net_loss
#insfolder = "./results/final_reg_2net_loss" #2net_no_loss
#tv = "./results/tesi_tv" #loss
#insfolder = "./results/tesi/24"
#canny = "./results/tesi_canny_notv"
#cannytv = "./results/tesi_canny"
base = "./results/tesi_base" # nothing
#nonblind = "./results/tesi_canny_noise4"
#insfolder = "./results/final_reg_net_loss"
#insfolder = "./results/final_tesi_3000"

originalfolder = "./datasets/original_tesi"

originals = os.listdir(originalfolder)
originals_mat = {}
for o in originals:
    originals_mat[o[:-4]] = (cv2.imread(os.path.join(originalfolder,o)))


results_mat = []

m = []
p = []
s = []

def evaluate_dataset(path):
    mses= []
    ssims= []
    psnrs = []
    results = list(filter(lambda x: x[-5:] == "x.png",os.listdir(path)))
    for r in results:
        img = cv2.imread(os.path.join(path,r))
        key = ""
        if "blurred" in r:
            key = r[:-6]
        else:
            key = r[:2]
        mses.append(mse(originals_mat[key],img) / (img.shape[0]*img.shape[1]))
        ssims.append(ssim(originals_mat[key],img))
        psnrs.append(psnr(originals_mat[key],img))
    return (mses,psnrs,ssims)

mc,pc,sc, = evaluate_dataset(base)
print(np.mean(mc))
print(np.mean(pc))
print(np.mean(sc))
m.append(mc)
p.append(pc)
s.append(sc)

mc,pc,sc, = evaluate_dataset(insfolder)
m.append(mc)
p.append(pc)
s.append(sc)
print(np.mean(mc))
print(np.mean(pc))
print(np.mean(sc))
"""
mc,pc,sc, = evaluate_dataset(canny)
m.append(mc)
p.append(pc)
s.append(sc)
print(np.mean(mc))
print(np.mean(pc))
print(np.mean(sc))
mc,pc,sc, = evaluate_dataset(cannytv)
m.append(mc)
p.append(pc)
s.append(sc)
print(np.mean(mc))
print(np.mean(pc))
print(np.mean(sc))

mc,pc,sc, = evaluate_dataset(nonblind)

print(np.mean(mc))
print(np.mean(pc))
print(np.mean(sc))"""
fig, ax = plt.subplots(1,3)
ax[0].boxplot(m)
#ax = plt.boxplot(m)
ax[0].set_xticklabels(['SelfDeblur',r'$Selfdeblur_{gtv}'])
ax[0].set_title('MSE')
ax[1].boxplot(p)
ax[1].set_title('PSNR')
#ax = plt.boxplot(m)
ax[1].set_xticklabels(['SelfDeblur',r'$Selfdeblur_{gtv}'])
#plt.boxplot(p)

ax[2].boxplot(s)
#ax = plt.boxplot(m)
ax[2].set_title('SSIM')
ax[2].set_xticklabels(['SelfDeblur',r'$Selfdeblur_{gtv}'])
#plt.boxplot(s)
plt.show()
"""
plt.clf()
plt.boxplot(psnrs)
plt.show()"""