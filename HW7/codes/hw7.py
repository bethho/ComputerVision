import cv2
import numpy as np

def binarize(img, threshold = 128):
    h, w = img.shape
    output = np.zeros((h, w), dtype = np.int)

    for i in range(h):
        for j in range(w):
            if img[i][j] >= threshold:
                output[i][j] = 1
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

def Yokoi_num(img_tmp):
    x = 1; y = 1
    a1 = h_f(img_tmp[x, y], img_tmp[x, y+1], img_tmp[x-1, y+1], img_tmp[x-1, y])
    a2 = h_f(img_tmp[x, y], img_tmp[x-1, y], img_tmp[x-1, y-1], img_tmp[x, y-1])
    a3 = h_f(img_tmp[x, y], img_tmp[x, y-1], img_tmp[x+1, y-1], img_tmp[x+1, y])
    a4 = h_f(img_tmp[x, y], img_tmp[x+1, y], img_tmp[x+1, y+1], img_tmp[x, y+1])
    out = f(a1, a2, a3, a4)
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
                output[i, j] = Yokoi_num(img_tmp[x-1:x+2, y-1:y+2])
            
    return output

def pair_relate_op(Yokoi):
    h, w = Yokoi.shape
    Yokoi_padding = np.zeros((h+2, w+2))
    Yokoi_padding[1:h+1, 1:w+1] = Yokoi
    result = np.zeros((h,w))
    # q: 1; p: 2
    for r in range(h):
        for c in range(w):
            x = c+1; y = r+1
            if (Yokoi[r, c] > 0):
                if (Yokoi[r, c] == 1):
                    nebor_4 = np.array([Yokoi_padding[y, x+1], Yokoi_padding[y-1, x], Yokoi_padding[y, x-1], Yokoi_padding[y+1, x]])
                    edge_num = np.sum(nebor_4 == 1)
                    result[r, c] = 2 if edge_num > 0 else 1
                else:
                    result[r, c] = 1
    return result

def shrink(pair_relate_img, symbol_img):
    h, w = symbol_img.shape
    symbol_img_padding = np.zeros((h+2, w+2))
    symbol_img_padding[1:h+1, 1:w+1] = symbol_img
    for r in range(h):
        for c in range(w):
            x = c+1; y = r+1
            if(pair_relate_img[r, c] == 2):
                isedge = Yokoi_num(symbol_img_padding[y-1:y+2, x-1:x+2])
                symbol_img_padding[y, x] = 0 if isedge == 1 else 1
    return symbol_img_padding[1:h+1, 1:w+1]

def thinning(symbol_img):
    output = np.zeros(symbol_img.shape)
    while(np.sum(np.abs(symbol_img - output))):
        output = np.array(symbol_img)
        # Yokoi_4connected
        Yokoi = Yokoi_4connected(symbol_img)
        # Pair Relationship Operator
        pair_relate_img = pair_relate_op(Yokoi)
        # shrink
        symbol_img = shrink(pair_relate_img, symbol_img)
    return output

img = cv2.imread('lena.bmp', 0)
img = binarize(img)
img = downsample(img)
cv2.imwrite('./downsample.png', (img*255).astype(np.uint8))
cv2.imwrite('./thinning.png', (thinning(img)*255).astype(np.uint8))