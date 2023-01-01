import cv2
import numpy as np

source_p = r'./lena.bmp'
result_p = r'./1c_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape

for c in range(w):
    for r in range(c+1, h):
        tmp = np.array(img[r, c, :])
        img[r, c, :] = img[c, r, :]
        img[c, r, :] = tmp

cv2.imwrite(result_p, img)




