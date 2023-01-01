import cv2
import numpy as np

source_p = r'./lena.bmp'
result_p = r'./1b_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape

for c in range(w):
    swap_c = w - c - 1
    if swap_c > c:
        for r in range(h):
            tmp = np.array(img[r, c, :])
            img[r, c, :] = img[r, swap_c, :]
            img[r, swap_c, :] = tmp

cv2.imwrite(result_p, img)




