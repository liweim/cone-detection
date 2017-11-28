import numpy as np
import pandas as pd
import os
from os.path import join
import xml.etree.ElementTree as ET
import glob

def get_bbox(xml_path):
    dirname = os.path.split(xml_path)[0]
    column_name = ['class', 'xmin', 'ymin', 'xmax', 'ymax']

    for xml_file in glob.glob(xml_path + '/*.xml'):
        basename = os.path.split(xml_file)[1]
        filename = os.path.splitext(basename)[0]
        bbox = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                if member[0].text == 'yellow' or member[0].text == 'blue':
                    value = (member[0].text,
                             int(member[4][0].text),
                             int(member[4][1].text),
                             int(member[4][2].text),
                             int(member[4][3].text)
                             )
                    bbox.append(value)
            except ValueError:
                pass

        bbox_df = pd.DataFrame(bbox, columns=column_name)
        bbox_df.to_csv(join(dirname, 'bbox', filename+'.csv'), index=None)
    return


if __name__ == '__main__':
    get_bbox(xml_path='video2/annotations_right')
