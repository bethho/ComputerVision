import cv2
import numpy as np

def change_index(map, component, k):
    h, w, = map.shape
    for e in component:
        r = e // w
        c = e % w
        map[r, c] = k

def binary_img(img_source, thr=128):
    h, w, _ = img_source.shape
    img = np.array(img_source)
    for c in range(w):
        for r in range(h):
            if img[r, c, 0] < thr:
                img[r, c, 0] = 0
                img[r, c, 1] = 0
                img[r, c, 2] = 0
            else:
                img[r, c, 0] = 255
                img[r, c, 1] = 255
                img[r, c, 2] = 255
    return img

source_p = r'./lena.bmp'
result_p = r'./result_c.bmp'
img_source = cv2.imread(source_p)
h, w, _ = img_source.shape

# produce binary img
img = binary_img(img_source)

# initial
i = 1
map = np.zeros(img_source.shape[:2])
component_item = {}

# 4 connected
for r in range(h):
    for c in range(w):
        if img[r, c, 0] == 255:
            if c > 0 and r > 0:
                if map[r-1, c] * map[r, c-1] > 0:
                    if map[r-1, c] < map[r, c-1]:
                        map[r, c] = map[r-1, c]
                        component_item[map[r, c]].append(r*w+c)
                        component_item[map[r, c]].extend(component_item[map[r, c-1]])
                        tmp = map[r, c-1]
                        change_index(map, component_item[map[r, c-1]], map[r, c])
                        del component_item[tmp]
                    
                    elif map[r-1, c] > map[r, c-1]:
                        map[r, c] = map[r, c-1]
                        component_item[map[r, c]].append(r*w+c)
                        component_item[map[r, c]].extend(component_item[map[r-1, c]])
                        tmp = map[r-1, c]
                        change_index(map, component_item[map[r-1, c]], map[r, c])
                        del component_item[tmp]

                    else:
                        map[r, c] = map[r, c-1]
                        component_item[map[r, c]].append(r*w+c)

                elif map[r-1, c] > 0 or map[r, c-1] > 0:
                    map[r, c] = max(map[r-1, c], map[r, c-1])
                    component_item[map[r, c]].append(r*w+c)

                else:
                    map[r, c] = i
                    component_item[map[r, c]] = [r*w+c]
                    i += 1

            elif c > 0 and r == 0:
                if map[r, c-1] > 0:
                    map[r, c] = map[r, c-1]
                    component_item[map[r, c]].append(r*w+c)
                
                else:
                    map[r, c] = i
                    component_item[map[r, c]] = [r*w+c]
                    i += 1

            elif r > 0 and c == 0:
                if map[r-1, c] > 0:
                    map[r, c] = map[r-1, c]
                    component_item[map[r, c]].append(r*w+c)
                
                else:
                    map[r, c] = i
                    component_item[map[r, c]] = [r*w+c]
                    i += 1
            else:
                map[r, c] = i
                component_item[map[r, c]] = [r*w+c]
                i += 1

# calculate boundingbox and centroid
thr_area = 500
for key, value in component_item.items():
    
    if len(value) >= thr_area:
        # calculate centroid
        y = np.array(value) // w
        y = int(np.mean(y))
        x = np.array(value) % w
        x = int(np.mean(x))

        # draw a circle
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # calculate boundingbox
        y_max = np.max(np.array(value) // w)
        y_min = np.min(np.array(value) // w)
        x_max = np.max(np.array(value) % w)
        x_min = np.min(np.array(value) % w)

        # draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3, cv2.LINE_AA)

cv2.imwrite(result_p, img)
