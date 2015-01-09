# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 14:24:51 2014

Test and compare MTL and single-task Facial Landmark detector. This test will 
perform on validation set. The images in validation set have been sorted 
descendingly according to pose_yaw

@author: james
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
import ImageDraw
from PIL import Image
import sys
sys.path.insert(0, "/home/james/Documents/caffe-master/distribute/python")
import caffe

def landmarkFromRow(row):
    result = [0] * 10
    for index in range(4, 14):
        result[index - 4] = float(row[index])
    return result

def landmarkFromPredic(prediction):
    result = [0] * 10
    for index in range(0, 10):
        result[index] = prediction[0][index][0][0]
    return result
        
def distance(groundTruth, prediction):
    result = 0
    reference = np.sqrt(np.power((groundTruth[0] - groundTruth[2]), 2) + np.power((groundTruth[1] - groundTruth[3]), 2))
    for i in range(0, 5):
        x = 2 * i
        y = 2 * i + 1
        result = result + np.sqrt(np.power((groundTruth[x] - prediction[x]), 2) + np.power((groundTruth[y] - prediction[y]), 2))
    result = result / 5
    result = result / reference
    return result
    
def fivePointDis(groundTruth, prediction):
    result = []
    reference = np.sqrt(np.power((groundTruth[0] - groundTruth[2]), 2) + np.power((groundTruth[1] - groundTruth[3]), 2))
    for i in range(0, 5):
        x = 2 * i
        y = 2 * i + 1
        dis = np.sqrt(np.power((groundTruth[x] - prediction[x]), 2) + np.power((groundTruth[y] - prediction[y]), 2))
        dis = dis / reference
        result.append(dis)
    return result

def saveObj(points, outPath, filename, tile, index):
    prefix = int2index(index)
    with open(outPath + r"/" + prefix + filename + r".obj", 'w') as f:
        f.write("#vertices\n")
        f.write("v " + str(points[0]*tile) + " " + str(points[1]*tile) + " 0\n")
        f.write("v " + str(points[2]*tile) + " " + str(points[3]*tile) + " 0\n")
        f.write("v " + str(points[4]*tile) + " " + str(points[5]*tile) + " 0\n")
        f.write("v " + str(points[6]*tile) + " " + str(points[7]*tile) + " 0\n")
        f.write("v " + str(points[8]*tile) + " " + str(points[9]*tile) + " 0")

def int2index(index):
    if (index / 10 == 0):
        return "0000"+str(index)
    elif (index / 100 == 0):
        return "000"+str(index)
    elif (index / 1000 == 0):
        return "00"+str(index)
    elif (index / 10000 == 0):
        return "0"+str(index)
    elif (index / 100000 == 0):
        return str(index)
    else:
        raise ValueError("This is not good index value")

def drawPoint(im, x, y):
    draw = ImageDraw.Draw(im)
    draw.point((x-1, y-1), fill = 0)
    draw.point((x-1, y), fill = 0)
    draw.point((x-1, y+1), fill = 0)
    draw.point((x, y-1), fill = 0)
    draw.point((x, y), fill = 1)
    draw.point((x, y+1), fill = 0)
    draw.point((x+1, y-1), fill = 0)
    draw.point((x+1, y), fill = 0)
    draw.point((x+1, y+1), fill = 0)
    return im
    
def drawLandmark(im, landmark, tile):
    for i in range(0, 5):
        xindex = 2 * i
        yindex = 2 * i + 1
        x = landmark[xindex] * tile
        y = landmark[yindex] * tile
        im = drawPoint(im, x, y)
    im.show()

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def left_right_range(errorlist):
    result = []
    # From 1 to 168
    tmp = 0
    for i in range(0, 168):
        tmp = tmp + errorlist[i]
    tmp = tmp / 168
    result.append(tmp)
    # From 169 to 5954
    tmp = 0
    for i in range(168, 5954):
        tmp = tmp + errorlist[i]
    tmp = tmp / 5786
    result.append(tmp)
    # From 5955 to 23091
    tmp = 0
    for i in range(5954, 23091):
        tmp = tmp + errorlist[i]
    tmp = tmp / 17137
    result.append(tmp)
    # From 23092 to 26197
    tmp = 0
    for i in range(23091, 26197):
        tmp = tmp + errorlist[i]
    tmp = tmp / 3106
    result.append(tmp)
    # From 26198 to 27602
    tmp = 0
    for i in range(26197, 27602):
        tmp = tmp + errorlist[i]
    tmp = tmp / 1405
    result.append(tmp)
    
    return result

def main():
    tile = 40;
    pose_norm = 60
    imdata = np.zeros((1, 1, tile, tile))
    
    imageRoot = r"/home/james/Documents/James/data/test"
    csvPath = r"/home/james/Documents/James/data/multilabellandmark_truth_test_yaw_de.csv"
    imageList = glob.glob(imageRoot + r"/*.jpg")
    numImages = len(imageList)
    xlabel = range(0, numImages)
    
    MODEL_FILE_MTL = r"/home/james/Documents/Master_Thesis/mtl_landmark_test/deploy_mtl.prototxt"
    MODEL_FILE_LANDMARK = r"/home/james/Documents/Master_Thesis/mtl_landmark_test/deploy_landmark.prototxt"
    PRETRAINED_MTL_List = [r"/home/james/Documents/Master_Thesis/mtl_landmark_test/multilabellandmark2000_lr5_re_iter_828000.caffemodel"]
    PRETRAINED_LANDMARK_List = [r"/home/james/Documents/Master_Thesis/mtl_landmark_test/landmark2000_lr5_train_iter_828000.caffemodel"]
    
    for index in range(0, 1):
        net_mtl = caffe.Net(MODEL_FILE_MTL, PRETRAINED_MTL_List[index])
        net_mtl.set_phase_test()
        net_mtl.set_mode_cpu()
        net_mtl.set_input_scale('data', 0.00390625)
        
        net_landmark = caffe.Net(MODEL_FILE_LANDMARK, PRETRAINED_LANDMARK_List[index])
        net_landmark.set_phase_test()
        net_landmark.set_mode_cpu()
        net_landmark.set_input_scale('data', 0.00390625)
        
        error_mtl_curve = []
        error_landmark_curve = []
        
        i = 0
        gt_path = r"/home/james/Documents/James/obj/gt"
        ensure_dir_exists(gt_path)
        mtl_path = r"/home/james/Documents/James/obj/mtl2000_lr5_b1"
        ensure_dir_exists(mtl_path)
        lm_path = r"/home/james/Documents/James/obj/lm2000_lr5_b1"
        ensure_dir_exists(lm_path)        
        with open(csvPath, 'rb') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                filename = row[0]
                imagePath = imageRoot + r"/" + filename + r".jpg"
                im = Image.open(imagePath)
                arr = np.array(im.getdata(), np.uint8).reshape(im.size[0], im.size[1])
                imdata[0, 0, :, :] = arr
                imdata = imdata.astype('float32')
                prediction_mtl = net_mtl.forward(data = imdata)
                prediction_landmark = net_landmark.forward(data = imdata)
                groundTruth = landmarkFromRow(row)
                predic_mtl = landmarkFromPredic(prediction_mtl['ip2'])
                predic_landmark = landmarkFromPredic(prediction_landmark['ip2'])
                #saveObj(groundTruth, gt_path, filename, tile, i)
                #saveObj(predic_mtl, mtl_path, filename, tile, i)
                #saveObj(predic_landmark, lm_path, filename, tile, i)
                i = i + 1
                
                #drawLandmark(im, predic_mtl, tile)
                #drawLandmark(im, predic_landmark, tile)
                error_mtl = distance(groundTruth, predic_mtl)
                error_landmark = distance(groundTruth, predic_landmark)
                
                #five_error_mtl = fivePointDis(groundTruth, predic_mtl)
                #five_error_landmark = fivePointDis(groundTruth, predic_landmark)
                
                #print error_mtl
                #print error_landmark
                #print five_error_mtl
                #print five_error_landmark
                
                error_mtl_curve.append(error_mtl)
                error_landmark_curve.append(error_landmark)
        error_mtl_left_right = left_right_range(error_mtl_curve)
        error_landmark_left_right = left_right_range(error_landmark_curve)
        print error_mtl_left_right
        print error_landmark_left_right        
        print reduce(lambda x, y: x + y, error_mtl_curve) / len(error_mtl_curve)
        print reduce(lambda x, y: x + y, error_landmark_curve) / len(error_landmark_curve)    
        #plt.plot(xlabel, error_mtl_curve)
        #plt.plot(xlabel, error_landmark_curve)
if __name__ == '__main__':
    main()
