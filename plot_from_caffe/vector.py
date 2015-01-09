#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Hugo
#
# Created:     26/04/2012
# Copyright:   (c) Hugo 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import math
import re

class Point:
    'Represents a 3D vector.'
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, val):
        return Point( self.x + val.x, self.y + val.y, self.z + val.z )

    def __sub__(self,val):
        return Point( self.x - val.x, self.y - val.y, self.z - val.z )

    def __iadd__(self, val):
        self.x = val.x + self.x
        self.y = val.y + self.y
        self.z = val.z + self.z
        return self

    def __isub__(self, val):
        self.x = self.x - val.x
        self.y = self.y - val.y
        self.z = self.z - val.z
        return self

    def __div__(self, val):
        return Point( self.x / val, self.y / val, self.z / val )

    def __mul__(self, val):
        return Point( self.x * val, self.y * val, self.z * val )

    def __idiv__(self, val):
        self.x = self.x / val
        self.y = self.y / val
        self.z = self.z / val
        return self

    def __imul__(self, val):
        self.x = self.x * val
        self.y = self.y * val
        self.z = self.z * val
        return self

    def rotateXY(self, angle):
        new_x = self.x * math.cos(angle) - self.y * math.sin(angle)
        new_y = self.x * math.sin(angle) + self.y * math.cos(angle)
        self.x, self.y = new_x, new_y
        return self

    def rotateXYaroundPoint(self, origin, angle):
        delta = self - origin
        dest = (origin + delta.rotateXY(angle))
        self.x = dest.x
        self.y = dest.y
        return self

    def __getitem__(self, key):
        if( key == 0):
            return self.x
        elif( key == 1):
            return self.y
        elif( key == 2):
            return self.z
        else:
            raise Exception("Invalid key to Point %d" % key)

    def __setitem__(self, key, value):
        if( key == 0):
            self.x = value
        elif( key == 1):
            self.y = value
        elif( key == 2):
            return self.z
        else:
            raise Exception("Invalid key to Point")

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def cross_product(self, b):
        return Point(   self.y * b.z - self.z * b.y,
                        self.z * b.x - self.x * b.z,
                        self.x * b.y - self.y * b.x)



    def distance_sqrd( self, point2 ):
        'Returns the distance between two points squared. Marginally faster than distance()'
        return ( (self.x-point2.x)**2 + (self.y-point2.y)**2 + (self.z-point2.z)**2)

    def distanceXY(self, point2):
        return math.sqrt( (self.x-point2.x)**2 + (self.y-point2.y)**2)

    def distance( self, point2 ):
        'Returns the distance between two points'
        return math.sqrt( self.distance_sqrd(point2) )

    def length_sqrd( self ):
        'Returns the length of a vector sqaured. Faster than Length(), but only marginally'
        return self.x**2 + self.y**2 + self.z**2

    def length( self ):
        'Returns the length of a vector'
        return math.sqrt( self.length_sqrd() )

    def normalize(self ):
        'Returns a new vector that has the same direction as vec, but has a length of one.'
        if( self.x == 0. and self.y == 0. and self.z == 0.):
            return self
        return self / self.length()

    def dot( self,b ):
        'Computes the dot product of a and b'
        return self.x*b.x + self.y*b.y + self.z*b.z

    def project_onto( self,v ):
        'Projects w onto v.'
        return v * w.dot(v) / v.length_sqrd()

    def middle(self, point2):
        return Point(x=(self.x + point2.x)/2. , y=(self.y + point2.y)/2. , z=(self.z + point2.z)/2.)


def main():
    pass

if __name__ == '__main__':
    main()
