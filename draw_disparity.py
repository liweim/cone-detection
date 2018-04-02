import numpy as np
import cv2
import pandas as pd
import xml.etree.ElementTree as ET

colors = [(0, 255, 255), (255, 0, 0), (0, 165, 255)]

def draw_disparity():
    csv_path = 'tmp/result/1513418744.114132.csv'
    img_path = 'tmp/disparity.png'

    img_source = cv2.imread(img_path)
    img = cv2.resize(img_source, (640, 360))
    # img = np.zeros((360, 640, 3)).astype(np.uint8)
    # img[:,:,0] = img[:,:,1] = img[:,:,2] = img_source

    pts = pd.read_csv(csv_path)
    pts = np.array(pts)
    for pt in pts:
        x = int(pt[0])
        y = int(pt[1])
        label = int(pt[2])
        cv2.circle(img, (x, y), 3, colors[label], -1)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    cv2.imwrite('tmp/draw_disparity.png', img)

def draw_bbx():
    xml_path = 'annotations/skidpad1/1513418744.114132.xml'
    # img_path = 'annotations/skidpad1/1513418744.114132.png'
    img_path = 'tmp/roi.png'
    img = cv2.imread(img_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    cones = []
    for member in root.findall('object'):
        label = member[0].text
        y1 = int(member[4][0].text)
        x1 = int(member[4][1].text)
        y2 = int(member[4][2].text)
        x2 = int(member[4][3].text)
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        length = int((y2-y1)/2*1.5)
        cones.append([x, y, length])

    for x, y, length in cones:
        cv2.rectangle(img,(y-length,x-length),(y+length,x+length),(0,0,255),1)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    cv2.imwrite('tmp/roi.png', img)

draw_bbx()