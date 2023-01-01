import cv2
import random
import numpy as np

def getGaussionNoise_Img(image, amp=10):
    gaussionNoise_Img = np.array(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            noise = int(gaussionNoise_Img[r, c] + amp*random.gauss(0, 1))
            if noise > 255:
                noise = 255
            gaussionNoise_Img[r, c] = noise
    return gaussionNoise_Img

def getSaltAndPeperNoise_Img(image, thresh=0.05):
    saltandpepper_Img = np.array(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            radom_value = random.uniform(0, 1)
            if (radom_value <= thresh):
                saltandpepper_Img[r, c] = 0
            elif (radom_value >= 1-thresh):
                saltandpepper_Img[r, c] = 255
        
    return saltandpepper_Img

def box_filter(image, kernal_size):
    filter_img = np.array(image)
    padding_pixel = kernal_size // 2
    padding_img = cv2.copyMakeBorder(image, padding_pixel, padding_pixel, padding_pixel, padding_pixel, cv2.BORDER_REPLICATE)
    for r in range(image.shape[0]):
        for c in range(img.shape[1]):
            x = r + padding_pixel
            y = c + padding_pixel
            filter_img[r, c] = int(np.sum(padding_img[x-padding_pixel:x+padding_pixel+1, y-padding_pixel:y+padding_pixel+1])/(kernal_size*kernal_size))
    return filter_img

def median_filter(image, kernal_size):
    filter_img = np.array(image)
    padding_pixel = kernal_size // 2
    padding_img = cv2.copyMakeBorder(image, padding_pixel, padding_pixel, padding_pixel, padding_pixel, cv2.BORDER_REPLICATE)
    for r in range(image.shape[0]):
        for c in range(img.shape[1]):
            x = r + padding_pixel
            y = c + padding_pixel
            filter_img[r, c] = int(np.median(padding_img[x-padding_pixel:x+padding_pixel+1, y-padding_pixel:y+padding_pixel+1]))
    return filter_img

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

def SNR(image, image_noise):
    image = image/255
    image_noise = image_noise/255
    us = np.sum(image)/(image.shape[0]*image.shape[1])
    VS = np.sum((image - us)**2)/(image.shape[0]*image.shape[1])
    un = np.sum(image_noise-image)/(image_noise.shape[0]*image_noise.shape[1])
    VN = np.sum((image_noise-image-un)**2)/(image_noise.shape[0]*image_noise.shape[1])
    SNR = 20*np.log10((VS**0.5)/(VN**0.5))
    return SNR

img = cv2.imread('lena.bmp', 0)
# generator
guassnoise_img_10 = getGaussionNoise_Img(img, 10)
guassnoise_img_30 = getGaussionNoise_Img(img, 30)
saltpepper_img_01 = getSaltAndPeperNoise_Img(img, 0.1)
saltpepper_img_005 = getSaltAndPeperNoise_Img(img, 0.05)
cv2.imwrite('guassnoise_img_10_%f.bmp' % (SNR(img, guassnoise_img_10)), guassnoise_img_10)
cv2.imwrite('guassnoise_img_30_%f.bmp' % (SNR(img, guassnoise_img_30)), guassnoise_img_30)
cv2.imwrite('saltpepper_img_01_%f.bmp' % (SNR(img, saltpepper_img_01)), saltpepper_img_01)
cv2.imwrite('saltpepper_img_005_%f.bmp' % (SNR(img, saltpepper_img_005)), saltpepper_img_005)

# box_filter
guassnoise_img_10_box3 = box_filter(guassnoise_img_10, 3)
guassnoise_img_10_box5 = box_filter(guassnoise_img_10, 5)
guassnoise_img_30_box3 = box_filter(guassnoise_img_30, 3)
guassnoise_img_30_box5 = box_filter(guassnoise_img_30, 5)
cv2.imwrite('guassnoise_img_10_box3_%f.bmp' % (SNR(img, guassnoise_img_10_box3)), guassnoise_img_10_box3)
cv2.imwrite('guassnoise_img_10_box5_%f.bmp' % (SNR(img, guassnoise_img_10_box5)), guassnoise_img_10_box5)
cv2.imwrite('guassnoise_img_30_box3_%f.bmp' % (SNR(img, guassnoise_img_30_box3)), guassnoise_img_30_box3)
cv2.imwrite('guassnoise_img_30_box5_%f.bmp' % (SNR(img, guassnoise_img_30_box5)), guassnoise_img_30_box5)

saltpepper_img_01_box3 = box_filter(saltpepper_img_01, 3)
saltpepper_img_01_box5 = box_filter(saltpepper_img_01, 5)
saltpepper_img_005_box3 = box_filter(saltpepper_img_005, 3)
saltpepper_img_005_box5 = box_filter(saltpepper_img_005, 5)
cv2.imwrite('saltpepper_img_01_box3_%f.bmp' % (SNR(img, saltpepper_img_01_box3)), saltpepper_img_01_box3)
cv2.imwrite('saltpepper_img_01_box5_%f.bmp' % (SNR(img, saltpepper_img_01_box5)), saltpepper_img_01_box5)
cv2.imwrite('saltpepper_img_005_box3_%f.bmp' % (SNR(img, saltpepper_img_005_box3)), saltpepper_img_005_box3)
cv2.imwrite('saltpepper_img_005_box5_%f.bmp' % (SNR(img, saltpepper_img_005_box5)), saltpepper_img_005_box5)

# median_filter
guassnoise_img_10_med3 = median_filter(guassnoise_img_10, 3)
guassnoise_img_10_med5 = median_filter(guassnoise_img_10, 5)
guassnoise_img_30_med3 = median_filter(guassnoise_img_30, 3)
guassnoise_img_30_med5 = median_filter(guassnoise_img_30, 5)
cv2.imwrite('guassnoise_img_10_med3_%f.bmp' % (SNR(img, guassnoise_img_10_med3)), guassnoise_img_10_med3)
cv2.imwrite('guassnoise_img_10_med5_%f.bmp' % (SNR(img, guassnoise_img_10_med5)), guassnoise_img_10_med5)
cv2.imwrite('guassnoise_img_30_med3_%f.bmp' % (SNR(img, guassnoise_img_30_med3)), guassnoise_img_30_med3)
cv2.imwrite('guassnoise_img_30_med5_%f.bmp' % (SNR(img, guassnoise_img_30_med5)), guassnoise_img_30_med5)

saltpepper_img_01_med3 = median_filter(saltpepper_img_01, 3)
saltpepper_img_01_med5 = median_filter(saltpepper_img_01, 5)
saltpepper_img_005_med3 = median_filter(saltpepper_img_005, 3)
saltpepper_img_005_med5 = median_filter(saltpepper_img_005, 5)
cv2.imwrite('saltpepper_img_01_med3_%f.bmp' % (SNR(img, saltpepper_img_01_med3)), saltpepper_img_01_med3)
cv2.imwrite('saltpepper_img_01_med5_%f.bmp' % (SNR(img, saltpepper_img_01_med5)), saltpepper_img_01_med5)
cv2.imwrite('saltpepper_img_005_med3_%f.bmp' % (SNR(img, saltpepper_img_005_med3)), saltpepper_img_005_med3)
cv2.imwrite('saltpepper_img_005_med5_%f.bmp' % (SNR(img, saltpepper_img_005_med5)), saltpepper_img_005_med5)

# opening-then-closing; closing-then opening
kernel = [
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
]
origin = [2, 2]

guassnoise_img_10_opclo = closing(opening(guassnoise_img_10, kernel, origin), kernel, origin)
guassnoise_img_10_cloop = opening(closing(guassnoise_img_10, kernel, origin), kernel, origin)
guassnoise_img_30_opclo = closing(opening(guassnoise_img_30, kernel, origin), kernel, origin)
guassnoise_img_30_cloop = opening(closing(guassnoise_img_30, kernel, origin), kernel, origin)
cv2.imwrite('guassnoise_img_10_opclo_%f.bmp' % (SNR(img, guassnoise_img_10_opclo)), guassnoise_img_10_opclo)
cv2.imwrite('guassnoise_img_10_cloop_%f.bmp' % (SNR(img, guassnoise_img_10_cloop)), guassnoise_img_10_cloop)
cv2.imwrite('guassnoise_img_30_opclo_%f.bmp' % (SNR(img, guassnoise_img_30_opclo)), guassnoise_img_30_opclo)
cv2.imwrite('guassnoise_img_30_cloop_%f.bmp' % (SNR(img, guassnoise_img_30_cloop)), guassnoise_img_30_cloop)

saltpepper_img_01_opclo = closing(opening(saltpepper_img_01, kernel, origin), kernel, origin)
saltpepper_img_01_cloop = opening(closing(saltpepper_img_01, kernel, origin), kernel, origin)
saltpepper_img_005_opclo = closing(opening(saltpepper_img_005, kernel, origin), kernel, origin)
saltpepper_img_005_cloop = opening(closing(saltpepper_img_005, kernel, origin), kernel, origin)
cv2.imwrite('saltpepper_img_01_opclo_%f.bmp' % (SNR(img, saltpepper_img_01_opclo)), saltpepper_img_01_opclo)
cv2.imwrite('saltpepper_img_01_cloop_%f.bmp' % (SNR(img, saltpepper_img_01_cloop)), saltpepper_img_01_cloop)
cv2.imwrite('saltpepper_img_005_opclo_%f.bmp' % (SNR(img, saltpepper_img_005_opclo)), saltpepper_img_005_opclo)
cv2.imwrite('saltpepper_img_005_cloop_%f.bmp' % (SNR(img, saltpepper_img_005_cloop)), saltpepper_img_005_cloop)