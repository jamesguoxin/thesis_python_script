#-------------------------------------------------------------------------------
# Name:        ptpt_error.py
# Purpose:     Compare the 5 points shapes obtained with the different facial
#               points detectors to the 5 points ground truth shape
#
# Author:      Magali Violero
#
# Created:     15/04/2014
# Copyright:   (c) nViso 2014
#-------------------------------------------------------------------------------

from __future__ import division

import os
import csv
import pygame
from pygame.locals import *
import math
from sets import Set
from shape import Shape
from vector import Point
from matplotlib.pyplot import *
from numpy import *

def distance(point1,point2):
    x1 = point1.x
    y1 = point1.y
    x2 = point2.x
    y2 = point2.y
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_ref(shape,indexes):

    points = shape.get_points(indexes)
    return distance(points[0],points[1])

def ptpt_error(points1, points2, ref):

        n = len(points1)
        if (n!=len(points2)):
            print "error : lists of points should have the same length, here", n, "and", len(points2)
            pass
        else:
            ptpt_error = 0.0
            for i in range(0,n):
                ptpt_error += distance(points1[i],points2[i])/ref
            return ptpt_error/n

def plot_curve(e,err,color,class_label):

    res = []
    for jj in range(0,len(e)):
        res.append(100*len([x for x in err if x<e[jj]])/len(err))
    plot(e,res,color,linewidth=2,label=class_label)

def main():

    pygame.init()

    # Directories
    obj_dir = r"/home/james/Documents/Master_Thesis/mtl_landmark_test/obj"

    # Methods
    method = ['mtl2000_lr5_noes', 'lm2000_lr5']
    label_method = ['mtl2000_lr5_noes', 'lm2000_lr5']

    # Indexes
    indexes = range(0,5)

    # Errors
    total_err = []
    for i in range(0,len(method)):
        N = 0
        method_err = []

        for file in os.listdir(obj_dir+'/'+method[i]):
            if file.endswith(".obj"):
                identity = file.split(".")[0]
                N+=1
                #print method[i], N, identity

                # Load gt
                shape_gt = Shape()
                shape_gt.read_obj(obj_dir + "/gt/" + identity + ".obj")
                shape_gt_points = shape_gt.get_points(indexes)
                ref = get_ref(shape_gt,[0,1])   # compute distance between two eyes as reference

                # Load method obj
                shape_detect = Shape()
                shape_detect.read_obj(obj_dir + "/" + method[i] + "/" + identity + ".obj")
                shape_detect_points = shape_detect.get_points(indexes)

                # Compute pt to pt error for all overlapping points and by facial subset
                method_err.append(ptpt_error(shape_gt_points,shape_detect_points,ref))

        total_err.append(method_err)

    # Cumulative curves

    e = linspace(0,0.4,80)
    mean_vector = []
    std_vector = []
    color = ['k','b','Purple','r','g','c','m','Brown','Orange','Yellow','Silver']
    ax = subplot(122)
    for ii in range(0,len(method)):
        err = [x for x in total_err[ii] if x<1]
        plot_curve(e,err,color[ii],label_method[ii])
        mean_vector.append(mean(err))
        std_vector.append(std(err))
    grid("on")
    title("Cumulative curves", y = 1.03)
    xlabel("Pt-pt error to ground truth shape\n(normalized by interocular distance)")
    ax.set_xticks([0,0.1,0.2,0.3,0.4])
    ylabel("% of images")
    ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    legend(loc=6)

    # Average error
    mean_vector = list(mean_vector)
    std_vector = list(std_vector)
    i = arange(len(mean_vector))
    bar_width = 0.2
    ax = subplot(121)
    #print mean_vector
    bar(i+bar_width/2,mean_vector,bar_width,color=color)#,yerr=std_vector)
    xticks(i+bar_width, label_method, rotation=25)
    grid(axis='y', color='w', linestyle='-')
    for x,y in zip(i,mean_vector):
        annotate(str(round(y,3)), (x+bar_width,y+0.5), ha='center', va='top',color='k')
    title("Average normalized pt-pt error", y = 1.03)
##    xlabel("method")
    ylabel("Average pt-pt error to ground truth shape\n(normalized by interocular distance)")
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.yaxis.set_label_coords(-0.15, 0.5)

    savefig(r"/home/james/Documents/Master_Thesis/mtl_landmark_test/results_new20150104.jpg")
    show()

if __name__ == '__main__':
    main()
