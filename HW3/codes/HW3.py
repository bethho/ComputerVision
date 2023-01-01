import cv2
import numpy as np
import matplotlib.pyplot as plt

# function
def colorhistogram(img):
    h, w = img.shape[:2]
    histogram = np.zeros((256))
    
    for c in range(w):
        for r in range(h):
            histogram[int(img[r, c, 0])] += 1
    
    return histogram


def drawhistogram(histogram, result_p):
    index = np.array([i for i in range(256)])
    plt.bar(index, histogram, color ='maroon', width = 0.4)

    plt.xlabel("bins")
    plt.ylabel("pixels")
    plt.title("histogram")
    plt.savefig(result_p) 
    plt.show()
    plt.close()

def calCDF(histogram):
    CDF = np.zeros((256))
    sum_ = np.sum(histogram)
    for i in range(256):
        # CDF[i] = np.sum(histogram[:i+1])/sum_
        if i == 0:
            CDF[0] = histogram[0]/sum_
        else:
            CDF[i] = CDF[i-1] + histogram[i]/sum_
        
    return CDF

def histogramequalization(img, CDF):
    if len(img.shape) > 2:
        img_gray = img[:,:,0]
    else:
        img_gray = img
        
    h, w = img_gray.shape
    for r in range(h):
        for c in range(w):
            light = int(img_gray[r, c])
            img_gray[r, c] = 255 * CDF[light]
    
    img_gray = img_gray.reshape(h, w, 1)
    img_gray = np.concatenate((img_gray, img_gray, img_gray), axis = -1)
    
    return img_gray


# (a) original image and its histogram
source_p = r'./lena.bmp'
result_p = r'./a_histogram.jpg'
img = cv2.imread(source_p)
histogram = colorhistogram(img)
drawhistogram(histogram, result_p)

# (b) image with intensity divided by 3 and its histogram
source_p = r'./lena.bmp'
result_p = r'./b_img.bmp'
histogram_p = r'./b_histogram.jpg'
img = cv2.imread(source_p)
# img = img//3
h, w, _ = img.shape
for r in range(h):
    for c in range(w):
        img[r, c, 0] = img[r, c, 0] // 3 
        img[r, c, 1] = img[r, c, 0]
        img[r, c, 2] = img[r, c, 0]
cv2.imwrite(result_p, img)
histogram = colorhistogram(img)
drawhistogram(histogram, histogram_p)

# (c) image after applying histogram equalization to (b) and its histogram
source_p = r'./b_img.bmp'
result_p = r'./c_img.bmp'
histogram_p = r'./c_histogram.jpg'
img = cv2.imread(source_p)
histogram = colorhistogram(img)
CDF = calCDF(histogram)
img_result = histogramequalization(img, CDF)
cv2.imwrite(result_p, img_result)
histogram = colorhistogram(img_result)
drawhistogram(histogram, histogram_p)