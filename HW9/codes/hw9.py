import cv2
import numpy as np

def roberts_op(img, thresh=30):
    img = img
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            r1 = padding_img[r+1, c+1] - padding_img[r, c]
            r2 = padding_img[r+1, c] - padding_img[r, c+1]
            output_img[r, c] = 255 if (r1**2 + r2**2)**0.5 < thresh else 0
    return output_img

def prewitt_op(img, thresh=24):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 1
            y = c + 1
            p1 = -np.sum(padding_img[x-1, y-1:y+2]) + np.sum(padding_img[x+1, y-1:y+2])
            p2 = -np.sum(padding_img[x-1:x+2, y-1]) + np.sum(padding_img[x-1:x+2, y+1])
            output_img[r, c] = 255 if (p1**2 + p2**2)**0.5 < thresh else 0
    return output_img

def sobel_op(img, thresh=38):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 1
            y = c + 1
            s1 = np.sum(np.array([-1,-2,-1])*padding_img[x-1, y-1:y+2] + np.array([1,2,1])*padding_img[x+1, y-1:y+2])
            s2 = np.sum(np.array([-1,-2,-1])*padding_img[x-1:x+2, y-1] + np.array([1,2,1])*padding_img[x-1:x+2, y+1])
            output_img[r, c] = 255 if (s1**2 + s2**2)**0.5 < thresh else 0
    return output_img

def freichen_grad_op(img, thresh=30):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 1
            y = c + 1
            s1 = np.sum(np.array([-1,-2**0.5,-1])*padding_img[x-1, y-1:y+2] + np.array([1,2**0.5,1])*padding_img[x+1, y-1:y+2])
            s2 = np.sum(np.array([-1,-2**0.5,-1])*padding_img[x-1:x+2, y-1] + np.array([1,2**0.5,1])*padding_img[x-1:x+2, y+1])
            output_img[r, c] = 255 if (s1**2 + s2**2)**0.5 < thresh else 0
    return output_img

def kirshcompass_op(img, thresh=135):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 1
            y = c + 1
            k0 = np.sum(np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])*padding_img[x-1:x+2, y-1:y+2])
            k1 = np.sum(np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])*padding_img[x-1:x+2, y-1:y+2])
            k2 = np.sum(np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])*padding_img[x-1:x+2, y-1:y+2])
            k3 = np.sum(np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])*padding_img[x-1:x+2, y-1:y+2])
            k4 = np.sum(np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])*padding_img[x-1:x+2, y-1:y+2])
            k5 = np.sum(np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])*padding_img[x-1:x+2, y-1:y+2])
            k6 = np.sum(np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])*padding_img[x-1:x+2, y-1:y+2])
            k7 = np.sum(np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])*padding_img[x-1:x+2, y-1:y+2])
            max_ = max(k0,k1,k2,k3,k4,k5,k6,k7)
            output_img[r, c] = 255 if max_ < thresh else 0
    return output_img

def Robincompass_op(img, thresh=43):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 1
            y = c + 1
            r0 = np.sum(np.array([[-1,-0,1],[-2,0,2],[-1,0,1]])*padding_img[x-1:x+2, y-1:y+2])
            r1 = np.sum(np.array([[0,1,2],[-1,0,1],[-2,-1,0]])*padding_img[x-1:x+2, y-1:y+2])
            r2 = np.sum(np.array([[1,2,1],[0,0,0],[-1,-2,-1]])*padding_img[x-1:x+2, y-1:y+2])
            r3 = np.sum(np.array([[2,1,0],[1,0,-1],[0,-1,-2]])*padding_img[x-1:x+2, y-1:y+2])
            r4 = np.sum(np.array([[1,0,-1],[2,0,-2],[1,0,-1]])*padding_img[x-1:x+2, y-1:y+2])
            r5 = np.sum(np.array([[0,-1,-2],[1,0,-1],[2,1,0]])*padding_img[x-1:x+2, y-1:y+2])
            r6 = np.sum(np.array([[-1,-2,-1],[0,0,0],[1,2,1]])*padding_img[x-1:x+2, y-1:y+2])
            r7 = np.sum(np.array([[-2,-1,0],[-1,0,1],[0,1,2]])*padding_img[x-1:x+2, y-1:y+2])
            max_ = max(r0,r1,r2,r3,r4,r5,r6,r7)
            output_img[r, c] = 255 if max_ < thresh else 0
    return output_img

def NevatiaBabu_op(img, thresh=12500):
    output_img = np.zeros(img.shape)
    padding_img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    padding_img = padding_img.astype(np.float32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            x = r + 2
            y = c + 2
            N0 = np.sum(np.array([[100,100,100,100,100],[100,100,100,100,100],[0,0,0,0,0],[-100,-100,-100,-100,-100],[-100,-100,-100,-100,-100]])*padding_img[x-2:x+3, y-2:y+3])
            N2 = np.sum(np.array([[100,100,100,32,-100],[100,100,92,-78,-100],[100,100,0,-100,-100],[100,78,-92,-100,-100],[100,-32,-100,-100,-100]])*padding_img[x-2:x+3, y-2:y+3])
            N4 = np.sum(np.array([[-100,32,100,100,100],[-100,-78,92,100,100],[-100,-100,0,100,100],[-100,-100,-92,78,100],[-100,-100,-100,-32,100]])*padding_img[x-2:x+3, y-2:y+3])
            N1 = np.sum(np.array([[100,100,100,100,100],[100,100,100,78,-32],[100,92,0,-92,-100],[32,-78,-100,-100,-100],[-100,-100,-100,-100,-100]])*padding_img[x-2:x+3, y-2:y+3])
            N3 = np.sum(np.array([[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100],[-100,-100,0,100,100]])*padding_img[x-2:x+3, y-2:y+3])
            N5 = np.sum(np.array([[100,100,100,100,100],[-32,78,100,100,100],[-100,-92,0,92,100],[-100,-100,-100,-78,32],[-100,-100,-100,-100,-100]])*padding_img[x-2:x+3, y-2:y+3])
            max_ = max(N0,N1,N2,N3,N4,N5)
            output_img[r, c] = 255 if max_ < thresh else 0
    return output_img

img = cv2.imread('lena.bmp', 0)
cv2.imwrite('roberts_op_30.bmp', roberts_op(img))
cv2.imwrite('prewitt_op_24.bmp', prewitt_op(img))
cv2.imwrite('sobel_op_38.bmp', sobel_op(img))
cv2.imwrite('freichen_grad_op_30.bmp', freichen_grad_op(img))
cv2.imwrite('kirshcompass_op_135.bmp', kirshcompass_op(img))
cv2.imwrite('Robincompass_op_43.bmp', Robincompass_op(img))
cv2.imwrite('NevatiaBabu_op_12500.bmp', NevatiaBabu_op(img))