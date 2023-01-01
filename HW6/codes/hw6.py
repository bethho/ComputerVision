import cv2
import numpy as np

def binarize(img, threshold = 128):
    h, w = img.shape
    output = np.zeros((h, w), dtype = np.int)

    for i in range(h):
        for j in range(w):
            if img[i][j] >= threshold:
                output[i][j] = 255
    return output

def downsample(img):
    scale = 8
    h, w = img.shape
    img_out = np.zeros((h//scale, w//scale))
    
    for i in range(h//scale):
        for j in range(w//scale):
            img_out[i, j] = img[i*scale, j*scale]
            
    return img_out

def h_f(b, c, d, e):
    out = ''
    if ((b == c) and ((d != b) or (e != b))):
        out = 'q'
    elif ((b == c) and ((d == b) and (e == b))):
        out = 'r'
    else:
        out = 's'
    return out

def f(a1, a2, a3, a4):
    data = np.array([a1, a2, a3, a4])
    r_num = np.sum(data == 'r')
    q_num = np.sum(data == 'q')
    out = 0
    if r_num == 4:
        out = 5
    else:
        out = q_num
    
    return out

def Yokoi_4connected(img):
    h, w = img.shape
    img_tmp = np.zeros((h+2, w+2))
    img_tmp[1:1+h, 1:1+w] = img
    output = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = i + 1
            y = j + 1
            if (img_tmp[x, y] > 0):
                a1 = h_f(img_tmp[x, y], img_tmp[x, y+1], img_tmp[x-1, y+1], img_tmp[x-1, y])
                a2 = h_f(img_tmp[x, y], img_tmp[x-1, y], img_tmp[x-1, y-1], img_tmp[x, y-1])
                a3 = h_f(img_tmp[x, y], img_tmp[x, y-1], img_tmp[x+1, y-1], img_tmp[x+1, y])
                a4 = h_f(img_tmp[x, y], img_tmp[x+1, y], img_tmp[x+1, y+1], img_tmp[x, y+1])
                output[i, j] = f(a1, a2, a3, a4)
            
    return output

img = cv2.imread('lena.bmp', 0)
img = binarize(img)
img = downsample(img)
output = Yokoi_4connected(img)

h, w = output.shape

for i in range(h):
    s = ''
    for j in range(w):
        s = s + '%s' % (int(output[i, j]) if output[i, j] else ' ') + ' '
    print("%s" % s)