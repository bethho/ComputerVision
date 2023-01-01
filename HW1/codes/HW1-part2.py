import cv2
import numpy as np

## (d) rotate lena.bmp 45 degrees clockwise
source_p = r'./lena.bmp'
result_rot_p = r'./2d_rot_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape
center = (w / 2, h / 2)

M = cv2.getRotationMatrix2D(center, -45, 1.0)
img_rot = cv2.warpAffine(img, M, (w, h))

cv2.imwrite(result_rot_p, img_rot)

## (e) shrink lena.bmp in half
source_p = r'./lena.bmp'
result_resize_p = r'./2e_resize_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape

result_resize = cv2.resize(img, (w//2, h//2))
img_result = np.zeros(img.shape)
img_result[h//2-h//4:h//2+h//4, w//2-w//4:w//2+w//4, :] = result_resize
cv2.imwrite(result_resize_p, img_result)

## (f) binarize lena.bmp at 128 to get a binary image
source_p = r'./lena.bmp'
result_binary_p = r'./2f_binary_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape
# check three channels are the same
a = img[:,:,0] - img[:,:,1]
b = img[:,:,1] - img[:,:,2]
c = img[:,:,0] - img[:,:,2]

if (np.max(a)+np.max(b)+np.max(c)) == 0:
    for c in range(w):
        for r in range(h):
            if img[r, c, 0] > 128:
                img[r, c, 0] = 255
                img[r, c, 1] = 255
                img[r, c, 2] = 255
            else:
                img[r, c, 0] = 0
                img[r, c, 1] = 0
                img[r, c, 2] = 0
cv2.imwrite(result_binary_p, img)
