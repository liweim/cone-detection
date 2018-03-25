import skimage.io
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

def std_color(img, plant_color):
    std_green = 50
    if plant_color == -1 or abs(plant_color-std_green) < 10:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]

    h_tmp = np.copy(h).astype(int)
    h_tmp -= (plant_color - std_green)
    for r in range(h.shape[0]):
        for c in range(h.shape[1]):
            if h_tmp[r,c] < 0:
                h_tmp[r,c] += 180
            if h_tmp[r,c] > 179:
                h_tmp[r,c] -= 180
    hsv[:,:,0] = h_tmp
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # print(plant_color)
    return img

def read_img(img_path):
    img = skimage.io.imread(img_path).astype(np.uint8)
    if img.shape[2] > 3:
        img = np.ascontiguousarray(img[:,:,:3])

    if np.max(img) > 255.0:
        raise Exception('Only 8 bit images supported')
    return img[:,:,::-1]

def write_txt(txt,point,way='w'):
    point = np.array(point)
    with open(txt,way) as f:
        for i in range(len(point)):
            f.write(str(point[i,0])+' '+str(point[i,1]))
            f.write('\n')

def write_txt_any(txt_path, data, way='w'):
    with open(txt_path, way) as f:
        for d in data:
            for i in range(len(d)):
                f.write(str(d[i])+' ')
            f.write('\n')

# def read_txt(txt_path):
#     with open(txt_path, 'r') as f:
#         s=f.readlines()
#         point=np.zeros((len(s),2))
#         for i in range(len(s)):
#             arr=s[i].split(' ')
#             point[i,:]=arr
#     return point

def read_txt(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.split(' '))
    return data
    
def read_txt_any(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(' ')
            data.append(arr[:-1])
    return data

def getPolygonBounds(points):
    length = len(points)
    top = down = left = right = points[0]
    for i in range(1, length):
        if points[i][0] > top[0]:
            top = points[i]
        elif points[i][0] < down[0]:
            down = points[i]
        else:
            pass
        if points[i][1] > right[1]:
            right = points[i]
        elif points[i][1] < left[1]:
            left = points[i]
        else:
            pass

    point0 = [top[0], left[1]]
    point1 = [top[0], right[1]]
    point2 = [down[0], right[1]]
    point3 = [down[0], left[1]]
    polygonBounds = [point0, point1, point2, point3]
    return polygonBounds

def isPointInRect(point, polygonBounds):
    if point[0] >= polygonBounds[3][0] and point[0] <= polygonBounds[0][0] and point[1] >= polygonBounds[3][1] and point[1] <= polygonBounds[2][1]:
        return True
    else:
        return False

def isPointsInPolygons(xyset,polygonset):
    inliers = []
    outliers = []
    for points in polygonset:
        polygonBounds = getPolygonBounds(points)
        for point in xyset:
            if not isPointInRect(point, polygonBounds):
                continue

            length = len(points)
            p = point
            p1 = points[0]
            flag = False
            for i in range(1, length):
                p2 = points[i]
                if (p[0] == p1[0] and p[1] == p1[1]) or (p[0] == p2[0] and p[1] == p2[1]):
                    inliers.append(p)
                    break
                if (p2[1] < p[1] and p1[1] >= p[1]) or (p2[1] >= p[1] and p1[1] < p[1]):
                    if (p2[1] == p1[1]):
                        x = (p1[0] + p2[0])/2
                    else:
                        x = p2[0] - (p2[1] - p[1])*(p2[0] - p1[0])/(p2[1] - p1[1])
                    if (x == p[0]):
                        inliers.append(p)
                        break
                    if (x > p[0]):
                        flag = not flag
                    else:
                        pass
                else:
                    pass

                p1 = p2
            if flag:
                inliers.append(p)
            else:
                outliers.append(p)
    inliers = np.array(inliers).astype(int)
    outliers = np.array(outliers).astype(int)
    return inliers, outliers
