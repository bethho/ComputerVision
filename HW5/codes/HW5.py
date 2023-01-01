import cv2
import numpy as np

def dilation(img, kernel, origin):
    h,w = img.shape
    hh = len(kernel)
    ww = len(kernel[0])
    output = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            max_elem = 0
            for a in range(hh):
                for b in range(ww):
                    m = i - (a - origin[0])
                    n = j - (b - origin[1])
                    if m < h and m >= 0 and n < w and n >= 0:
                        max_elem = max(max_elem, img[m][n]+kernel[a][b])
            output[i][j] = max_elem
    return output

def erosion(img, kernel, origin):
    h,w = img.shape
    hh = len(kernel)
    ww = len(kernel[0]) 
    output = np.zeros((h, w))
    pad_h = hh-origin[0]
    pad_w = ww-origin[1]
    for i in range(pad_h, h-pad_h):
        for j in range(pad_w, w-pad_w):
            min_elem = 255
            for a in range(hh):
                for b in range(ww):
                    m = i + (a - origin[0])
                    n = j + (b - origin[1])
                    if m < h and m >= 0 and n < w and n >= 0:
                        min_elem = min(min_elem, img[m][n]-kernel[a][b])
            output[i][j] = min_elem
    return output

def opening(img, kernel, origin):
    output = erosion(img, kernel, origin)
    output = dilation(output, kernel, origin)
    return output

def closing(img, kernel, origin):
    output = dilation(img, kernel, origin)
    output = erosion(output, kernel, origin)
    return output

img = cv2.imread('lena.bmp', 0)

# kernel
kernel = [
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
]
origin = [2, 2]

cv2.imwrite('dilation.bmp', dilation(img, kernel, origin))
cv2.imwrite('erosion.bmp', erosion(img, kernel, origin))
cv2.imwrite('opening.bmp', opening(img, kernel, origin))
cv2.imwrite('closing.bmp', closing(img, kernel, origin))