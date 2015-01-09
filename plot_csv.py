# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:09:13 2014

Plot result after log parser
Hopefully we could find any clue to set early stopping correctly

@author: JamesGuo
"""
import csv
import matplotlib.pyplot as ptl

def readCSV(csv_path, column):
    index = 0
    result = []
    with open(csv_path, "rb") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if (index > 0):
                value = float(row[column])
                result.append(value)
            index = index + 1
    return result

def main():
    # column 0 is number of iterations
    # column 1 is time in seconds
    # column 2 is learning rate
    # column 3 is value of early stop layer
    # column 4 is loss_landmark
    # column 5 is loss_pose
    csv_path = r"/Users/JamesGuo/Documents/MasterThesis/caffe_mtl_lr5_resume_log_test.csv"   
    loss_lm = readCSV(csv_path, 4)
    loss_pose = readCSV(csv_path, 5)
    xlabel = range(241, 2001)
    ptl.plot(xlabel, loss_lm, 'r')
    ptl.plot(xlabel, loss_pose, 'b')
    

if __name__ == "__main__":
    main()
