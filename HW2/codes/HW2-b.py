import cv2
import numpy as np
import matplotlib.pyplot as plt

histogram = np.zeros((256))
index = np.array([i for i in range(256)])

source_p = r'./lena.bmp'
result_p = r'./b_result.jpg'
img = cv2.imread(source_p)

h, w, _ = img.shape

for c in range(w):
    for r in range(h):
        histogram[int(img[r, c, 0])] += 1

plt.bar(index, histogram, color ='maroon', width = 0.4)
 
plt.xlabel("bins")
plt.ylabel("pixels")
plt.title("histogram")
plt.savefig(result_p) 
plt.show()
# plt.plot(index, histogram)
# plt.savefig(result_p) 
# plt.show()
