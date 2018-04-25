import ogr, osr, os, sys
import pandas as pd
import gdal
import argparse

def load_coords_from_file(path_to_file):
    coords = pd.read_csv(path_to_file, header=None, delim_whitespace=True)
    points = list()
    for index, row in coords.iterrows():
        if len(row.values) == 3:
            points.append((row.values[1], row.values[0], row.values[2]))
        elif len(row.values) == 4:
            points.append((row.values[1], row.values[0], row.values[3], row.values[2]))
        else:
            points.append((row.values[1], row.values[0]))

    return points

def pixel2coord(x, y):
    """Returns global coordinates from pixel x, y coords"""
    xp = x_res * x + xoff
    yp = y_res * y + yoff
    return(xp, yp)

def csv2shp(input, output, geotiff):
    tiff_ds = gdal.Open(geotiff)
    xoff, x_res, x_rot, yoff, y_rot, y_res = tiff_ds.GetGeoTransform()

    driver = ogr.GetDriverByName("GeoJSON")
    if os.path.exists(output):
        driver.DeleteDataSource(output)

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    shp_ds = driver.CreateDataSource(output)
    layer = shp_ds.CreateLayer(output, wgs84, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("Size", ogr.OFTInteger))

    for p in load_coords_from_file(input):
        if len(p) == 4:
            x_min = x_res * p[0] + xoff
            y_min = y_res * p[1] + yoff
            x_max = x_res * p[2] + xoff
            y_max = y_res * p[3] + yoff

            geom = ogr.Geometry(ogr.wkbLinearRing)
            geom.AddPoint(x_min, y_min)
            geom.AddPoint(x_min, y_max)
            geom.AddPoint(x_max, y_max)
            geom.AddPoint(x_max, y_min)
            geom.AddPoint(x_min, y_min)
        else:
            x_coord = x_res * p[0] + xoff
            y_coord = y_res * p[1] + yoff
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(x_coord, y_coord)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geom)

        if len(p) == 3:
            feature.SetField("Size", p[2])

        layer.CreateFeature(feature)
        feature = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, help="Input file with coordinates")
    parser.add_argument("--o", type=str, help="Output geojson file")
    parser.add_argument("--t", type=str, help="Path to geotiff to georeference with")

    args = parser.parse_args()

    csv2shp(args.i, args.o, args.t)
