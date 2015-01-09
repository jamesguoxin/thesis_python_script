#-------------------------------------------------------------------------------
# Name:        shape.py
# Purpose:       Data structure and methods to store and make operations on AAM shapes
#               (basically it's a set of vertices and a set of triangles)
#
# Author:      Hugo Penedones
#
# Created:     26/04/2012
# Copyright:   (c) nViso 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numbers
import math
import re
import os
import copy
import sys

from vector import Point


class Shape:
    def __init__(self, width = None, height = None):
        self.points       = []
        self.triangles    = []
        self.is_relative  = True
        self.width          = width
        self.height         = height

    def get_indexes(self):
        return range(0, len(self.points))

    def get_points(self, indexes):
        plist = []
        for i in indexes:
            plist.append(self.points[i])
        return plist

    def get_points_2d_tuple(self, indexes):
        plist = []
        for i in indexes:
            plist.append((self.points[i][0], self.points[i][1]))
        return plist

    def get_points_3d_tuple(self, indexes):
        plist = []
        for i in indexes:
            plist.append((self.points[i][0], self.points[i][1], self.points[i][2]))
        return plist

    def add_point(self, x, y, z):
        assert(isinstance(x,numbers.Number))
        assert(isinstance(y,numbers.Number))
        assert(isinstance(z,numbers.Number))
        self.points.append(Point(x,y,z))

    def del_point(self, index):
        assert(self.point_index_exists(index))
        del self.points[index]
        self.del_all_triangles_with_point(index)
        self.update_triangle_indexes(index, -1)

    def move_point(self, index, x, y, z = 0):
        assert(self.point_index_exists(index))
        self.points[index] += Point(x,y,z)

    def del_points_not_respecting(self, condition):
        todelete = []
        for i in range(0, len(self.points)):
             if( not condition(i, self.points[i])):
                todelete.append(i - len(todelete))  #  when things are deleted list will shrink, so we have to change the indexes accordingly
        for p in todelete:
            self.del_point(p)

    def del_all_points(self):
        del self.points[:]
        del self.triangles[:]

    def point_index_exists(self, index):
        return (index >= 0 and index < len(self.points))

    def add_triangle(self, pA, pB, pC):
        assert(self.point_index_exists(pA))
        assert(self.point_index_exists(pB))
        assert(self.point_index_exists(pC))
        assert(pA != pB)
        assert(pA != pC)
        assert(pB != pC)
        if not self.triangle_exists(pA,pB,pC):
            self.triangles.append([pA,pB,pC])

    def triangle_index_exists(self, index):
        return (index >= 0 and index < len(self.triangles))

    def get_triangle_index(self, pA, pB, pC):
        for i, t in enumerate(self.triangles):
            if (pA in t) and (pB in t) and (pC in t):
                return i
        return -1

    def triangle_exists(self, pA, pB, pC):
        return self.get_triangle_index(pA, pB, pC) >= 0

    def del_triangle_by_index(self,t_index):
        assert(self.triangle_index_exists(t_index))
        del self.triangles[t_index]

    def del_triangle(self,pA, pB, pC):
        i = self.get_triangle_index(pA, pB,pC)
        if i>=0:
            self.del_triangle_by_index(i)
            return True
        else:
            return False

    def del_all_triangles_with_point(self, p_index):
        self.triangles = [t for t in self.triangles if not p_index in t ]

    def update_triangle_indexes(self, p_old_index, p_diff):
        for t in self.triangles:
            for i in range(0,len(t)):
                if t[i] >= p_old_index:
                    t[i] += p_diff

    def get_nearest_point(self, x, y, z):
        min_dist = float("inf")
        nearest = -1
        target = Point(x,y,z)
        for i in range(0,len(self.points)):
            dist = self.points[i].distanceXY(target)
            if dist < min_dist:
                min_dist = dist
                nearest = i
        return (nearest, min_dist)

    def get_center_of_mass_of_selection(self, selection):
        acc = Point(0,0,0)
        for p in selection:
            assert(self.point_index_exists(p))
            acc += self.points[p]
        return acc / len(selection)

    def get_center_of_mass(self):
        return self.get_center_of_mass_of_selection(self.get_indexes())

    def get_ptpt_error(self, ref_shape, indexes):
        ptpt_error = []
        for i in indexes:
            x1 = self.points[i].x
            y1 = self.points[i].y
            x2 = ref_shape.points[i].x
            y2 = ref_shape.points[i].y
            ptpt_error.append( ((x1 - x2)**2 + (y1 - y2)**2)**0.5 )

        avg_ptpt_error = 0.0
        for error in ptpt_error:
            avg_ptpt_error += error

        return ( avg_ptpt_error / len(ptpt_error) )


    def scale_coord_of_selection(self,coord, factor, selection):
        for i in selection:
            self.points[i][coord] *= factor

    def scale_coord(self, coord, factor):
        self.scale_coord_of_selection(coord, factor, self.get_indexes())

    def scale_selection(self, factors, selection):
        for i, v in enumerate(factors):
            self.scale_coord_of_selection(i,v, selection)

    def scale_norm_selection(self, selection=None):
        if not selection:
            selection=self.get_indexes()

        # Compute average distance to center
        center_mass = self.get_center_of_mass_of_selection(selection)
        avg_dist = 0.0
        for i in selection:
            avg_dist += self.points[i].distance(center_mass)
        scale_factor = 1.0 / ( avg_dist / len(selection) )
        self.scale( [ scale_factor, scale_factor, scale_factor] )

    def scale(self, factors):
        self.scale_selection(factors, self.get_indexes())


    def face_normal(self, triangle):
        pA = self.points[triangle[0]]
        pB = self.points[triangle[1]]
        pC = self.points[triangle[2]]
        vAB = pB - pA
        vBC = pC - pB
        return vAB.cross_product(vBC)

    def invert_triangle_normal(self, t_index):
        assert(self.triangle_index_exists(t_index))
        temp_0 = self.triangles[t_index][0]
        temp_1 = self.triangles[t_index][1]
        self.triangles[t_index] = [temp_1,temp_0,self.triangles[t_index][2]]

    def fix_triangle_normals(self):
        for i,t in enumerate(self.triangles):
            if self.face_normal(t)[2] < 0:
                self.invert_triangle_normal(i)

    def static_scale(self, factors, selection = None):
        if not selection:
            selection=self.get_indexes()
        init_center_mass = self.get_center_of_mass_of_selection(selection)
        self.scale_selection(factors, selection)
        curr_center_mass = self.get_center_of_mass_of_selection(selection)
        self.shift_selection_pos( init_center_mass - curr_center_mass, selection)

    def shift_coord_of_selection(self, coord, dist, selection ):
        for i in selection:
            self.points[i][coord] += dist

    def shift_coord(self, coord, dist ):
        self.shift_coord_of_selection(coord, dist, self.get_indexes())

    def shift(self, vect):
        self.shift_selection_pos(vect, self.get_indexes())

    def shift_minus(self, vect):
        self.shift_selection_neg(vect, self.get_indexes())

    def shift_selection_neg(self, vect, selection):
        for i in selection:
            self.points[i] -= vect

    def shift_selection_pos(self, vect, selection):
        for i in selection:
            self.points[i] += vect

    def shift_selection(self, vect, selection):
        return self.shift_selection_pos(vect, selection)


    def mirror(self, coord):
        self.mirror_selection(coord, self.get_indexes())

    def mirror_selection(self, coord, selection):
        scale = 1
        if not self.is_relative:
            if coord == 0:
                scale = self.width
            else:
                scale = self.height
        for p in selection:
            self.points[p][coord] = scale - self.points[p][coord]

    def get_coord(self, coord):
        return [row[coord] for row in self.points]


    def center_selection(self, selection = None):
        if not selection:
            selection=self.get_indexes()
        shift = self.get_center_of_mass_of_selection(selection)
        self.shift_selection_neg( shift, selection)

    def rotate_selection(self, angle, selection):
        center_mass = self.get_center_of_mass_of_selection(selection)
        for i in selection:
            self.points[i].rotateXYaroundPoint(center_mass, angle)


    def to_absolute(self, pic_width, pic_height):
        for p in self.points:
            p.x *= pic_width
            p.y *= pic_height
        self.is_relative = False
        self.width = pic_width
        self.height = pic_height

    def to_relative(self,pic_width, pic_height):
        assert(pic_height > 0)
        assert(pic_width > 0)
        for p in self.points:
            p.x /= pic_width
            p.y /= pic_height
        self.is_relative = True
        self.width  = 1
        self.height = 1


    def fit_in_scale(self, min_scale, max_scale):
        assert(max_scale > min_scale)
        for coord in range(0,3):
            coord_vals = self.get_coord(coord)
            min_v = min(coord_vals)
            max_v = max(coord_vals)
            self.shift_coord(coord, -min_v )
            if (max_v > min_v): # prevent division by zero
                self.scale_coord(coord, (max_scale-min_scale)/(float(max_v)-min_v))
            self.shift_coord(coord, min_scale)


    # very rough and hardcoded version of format convertor, specific to nvFaces
    # TODO: generalize for other shapes, taking the matching triangle
    def init_from_shape(self, other_shape):
        other_eye_dist = other_shape.points[14].distance(other_shape.points[22])
        my_eye_dist    = self.points[14].distance(self.points[22])
        f1 = other_eye_dist/my_eye_dist
        other_face_height = other_shape.points[33].distance(other_shape.points[34])
        my_face_height = self.points[33].distance(self.points[34])
        f2 = other_face_height/my_face_height
        self.scale([f1,f2,1])
        delta = other_shape.points[14] - self.points[14]
        self.shift(delta)
        for i,p in enumerate(other_shape.points):
            self.points[i] = p




    def add_border(self, orig_width, orig_height, border):
        new_height = float(orig_height+2*border)
        new_width  = float(orig_width+2*border)
        self.scale( [orig_width/new_width, orig_height/new_height,1])
        if self.is_relative:
            self.shift( Point(border/new_width, border/new_height, 0) )
        else:
            self.shift( Point(border, border, 0) )

    def read_pts(self, pts_filepath):
        self.del_all_points()
        img_link = None
        pts_file = open(pts_filepath, 'r')
        number_rexpr = re.compile(r"[-+]?\d*\.\d+|\d+")
        for line in pts_file:
            numbers_list = number_rexpr.findall(line)
            if len(numbers_list) == 2 :
                if(self.width != None or self.height != None):
                    x = float(numbers_list[0]) / self.width
                    y = (self.height - float(numbers_list[1])) / self.height
                    z = 0.0;
                    self.add_point(x,y,z)
                else:
                    x = float(numbers_list[0])
                    y = 1.0 - float(numbers_list[1])
                    z = 0.0;
                    self.add_point(x,y,z)
        pts_file.close()

    # Assuming a very specific format. Without z coordinate
    # Only reading coordinates. Discarding paths.
    def read_asf(self, asf_filepath):
        self.del_all_points()
        img_link = None
        asf_file = open(asf_filepath, 'r')
        number_rexpr = re.compile(r"\d+\.*\d*\s+")
        for line in asf_file:
            numbers_list = number_rexpr.findall(line)
            if len(numbers_list) >= 6 :
                if(self.width != None or self.height != None):
                    x = float(numbers_list[2]) / self.width
                    y = 1-float(numbers_list[3]) / self.height
                    z = 0.0;
                    self.add_point(x,y,z)
                else:
                    x = float(numbers_list[2])
                    y = 1-float(numbers_list[3])
                    z = 0.0;
                    self.add_point(x,y,z)
            else:
                linelower = line.lower().strip()
                if linelower.endswith('.jpg') or linelower.endswith('.png') or linelower.endswith('.bmp'):
                    img_link = linelower
        asf_file.close()
        return img_link

    def write_asf_from_template(self, asf_template, out_filepath, img_link=None):
        template_file = open(asf_template, 'r')
        all_lines = template_file.readlines()
        template_file.close()

        out_file = open(out_filepath, 'w+')
        number_rexpr = re.compile(r"\d+\.*\d*\s+")
        i = 0
        for line in all_lines:
            numbers_list = number_rexpr.findall(line)
            if len(numbers_list) >= 6 :
                numbers_list[2] = '%.8f' % self.points[i][0]
                numbers_list[3] = '%.8f' % (1-self.points[i][1])
                i += 1
                out_file.write('\t'.join(map(str.strip, numbers_list)) + '\n')
            else:
                linelower = line.lower()
                if img_link and (('jpg' in linelower ) or ('png'in linelower) or ('bmp' in linelower)):
                        out_file.write(img_link)
                else:
                        out_file.write(line)
        out_file.close()


    def write_asf(self, out_filepath, img_link=None):
        if not img_link:
            img_link = os.path.basename(out_filepath)[:-4]+'.jpg'
        out_file = open(out_filepath, 'w+')
        out_file.write("""######################################################################
#
# AAM Shape File - outputed by nViso aamtool
#
######################################################################

#
# number of model points
#
%d

#
# model points
#
# format: <path#> <type> <x rel.> <y rel.> <point#> <connects from> <connects to> <user1> <user2> <user3>
#
""" % self.npoints())
        for i,p in enumerate(self.points):
            out_file.write("%d  4    %.6f    %.6f    %d        %d    1    0.000000 0.000000 0.000000\n" % (i,p.x, p.y, i,i))
        out_file.write("""
#
# host image
#
%s
""" % img_link)
        out_file.flush()
        out_file.close()



    def read_obj(self, obj_filepath):
    #    is_in_unit_cube = False
        i = 0
        self.del_all_points()
        obj_file = open(obj_filepath, 'r')
        lines = obj_file.readlines()
        for line in lines:
            i = i + 1
            if line.startswith('vt'):
                # skip texture information
                cols = line.split()
            elif line.startswith('vn'):
                # skip normal information
                cols = line.split()
            elif line.startswith('v'):
                cols = line.split()
                x = float(cols[1])
                y = float(cols[2])
                z = float(cols[3])
                self.add_point(x,y,z)
     #           if abs(x) > 1 or abs(y) > 1 or abs(z)>1:
     #               is_in_unit_cube = False
            if line.startswith('f'):
                cols = line.split()
                face0 = cols[1].split('/')
                face1 = cols[2].split('/')
                face2 = cols[3].split('/')
                self.add_triangle(int(face0[0])-1, int(face1[0])-1, int(face2[0])-1)
        obj_file.close()
     #   if not is_in_unit_cube:
     #       self.fit_in_scale(0,1)

    def write_obj(self, obj_filepath):
        print('Saving 2D to', obj_filepath)
        obj_file = open(obj_filepath, 'w+')
        obj_file.write('#Vertices\n')
        for p in self.points:
            obj_file.write('v %f %f %f\n' % (p[0], p[1], p[2]))
        obj_file.write('#Texture\n')
        obj_file.write('#Faces\n')
        for t in self.triangles:
            obj_file.write('f %d %d %d\n' % (t[0]+1, t[1]+1, t[2]+1))
        obj_file.close()

    def write_obj_3d(self, obj_filepath):
        normals = []
        print('Saving 3D to', obj_filepath)
        obj_file = open(obj_filepath, 'w+')
        points_texture = copy.deepcopy(self.points)
        self.scale([1.0,1.0,1.0])

        obj_file.write('#Vertices\n')
        for p in self.points:
            obj_file.write('v %f %f %f\n' % (p[0], p[1], p[2]))

        obj_file.write('#Texture\n')
        for p in points_texture:
            obj_file.write('vt %f %f\n' % (p[0], p[1]))

##        obj_file.write('#Normals\n')
##        for i,t in enumerate(self.triangles):
##            normals.append(self.face_normal(t))
##
##        for i,p in enumerate(self.points):
##            normal_vertex = None
##            for j,t in enumerate(self.triangles):
##                if (j in t[0]) or (j in t[1]) or (j in t[2]):
##                    normal_vertex += face_normal(t)
##
##
##            obj_file.write('vn %f %f %f\n' % (normal[0], normal[1], normal[2]))

        obj_file.write('#Faces\n')
        for i,t in enumerate(self.triangles):
            obj_file.write('f %d/%d %d/%d %d/%d\n' % (t[0]+1,t[0]+1, t[1]+1,t[1]+1, t[2]+1,t[2]+1))
        obj_file.close()

    def min_x(self):
        min_x = sys.maxint
        for p in self.points:
            min_x = min(min_x, p.x)
        return min_x

    def max_x(self):
        max_x = -sys.maxint - 1
        for p in self.points:
            max_x = max(max_x, p.x)
        return max_x

    def min_y(self):
        min_y = sys.maxint
        for p in self.points:
            min_y = min(min_y, p.y)
        return min_y

    def max_y(self):
        max_y = -sys.maxint - 1
        for p in self.points:
            max_y = max(max_y, p.y)
        return max_y

    def min_z(self):
        min_z = sys.maxint
        for p in self.points:
            min_z = min(min_z, p.z)
        return min_z

    def max_z(self):
        max_z = -sys.maxint - 1
        for p in self.points:
            max_z = max(max_z, p.z)
        return max_z

    def from_sdk_face(self, face_dict, img_width, img_height):
        self.points = []
        left_offset = face_dict['bb'][0] / float(img_width)
        top_offset = face_dict['bb'][1] / float(img_height)
        for (x,y,z,) in face_dict['shape']:
            self.add_point(left_offset+(x*face_dict['bb'][2]/float(img_width)), top_offset+(y*face_dict['bb'][3]/float(img_height)), z)

    def to_sdk_face(self,
            img_width, img_height,
            border_left = 0.1,
            border_right = 0.1,
            border_top = 0.1,
            border_bottom = 0.15):

        min_x = self.min_x()
        max_x = self.max_x()

        sh_w = max_x - min_x
        min_x = min_x - border_left * sh_w
        max_x = max_x + border_right * sh_w
        sh_w = max_x - min_x

        min_y = self.min_y()
        max_y = self.max_y()

        sh_h = max_y - min_y
        min_y = min_y - border_top * sh_h # y is inversed in soufce shape
        max_y = max_y + border_bottom * sh_h
        sh_h = max_y - min_y

        bb_left = min_x * img_width
        bb_top = min_y * img_height
        bb_width = sh_w * img_width
        bb_height = sh_h * img_height
        bb = (bb_left,bb_top,bb_width,bb_height)

        points = []
        for p in self.points:
            points.append(((p.x - min_x)/sh_w, (p.y - min_y)/sh_h, p.z,))
        res = {'bb':bb, 'shape':points}
        return res

    def npoints(self):
        return len(self.points)

    def set_depth(self, depthmap_surface):
        pic_width, pic_height = depthmap_surface.get_size()
        print (pic_width, pic_height)
        for i, p in enumerate(self.points):
            p.z = float(depthmap_surface.get_at((int(p[0] * pic_width),int(p[1] * pic_height)))[0]) / 255.0
            #p.z = float(depthmap_surface.get_at([int(p[0] * depthmap_surface.get_width()),int(p[1] * depthmap_surface.get_height())])[0] / 255.0)
            self.points[i] = Point(p.x,p.y,p.z)

    def subdivide(self):
        """Subdivide each triangle into four triangles, pushing verts to the depthmap"""
        new_triangles = copy.deepcopy(self.triangles)
        nb_triangles = len(self.triangles)
        for faceIndex in xrange(nb_triangles):

            # Create three new verts at the midpoints of each edge:
            triangle = self.triangles[faceIndex]
            a = self.points[triangle[0]]
            b = self.points[triangle[1]]
            c = self.points[triangle[2]]

            a1 = (a + b) / 2
            b1 = (b + c) / 2
            c1 = (a + c) / 2

            self.points.append(a1)
            self.points.append(b1)
            self.points.append(c1)

            # Split the current triangle into four smaller triangles:
            i = len(self.points) - 3
            j = i + 1
            k = i + 2
            new_triangles.append((i, j, k))
            new_triangles.append((triangle[0], i, k))
            new_triangles.append((i, triangle[1], j))
            new_triangles[faceIndex] = (k, j, triangle[2])
        self.triangles = new_triangles

    def read(self, filepath):
        assert(filepath.endswith('asf') or filepath.endswith('obj') or filepath.endswith('pts'))
        if filepath.endswith('asf'):
            self.read_asf(filepath)
        if filepath.endswith('obj'):
            self.read_obj(filepath)
        if filepath.endswith('pts'):
            self.read_pts(filepath)


    def npoints(self):
        return len(self.points)

def main():

    # 170 points
    #shape1 = Shape()
    #shape1.read_obj()

    # 68 points
    shape2 = Shape()
    shape2.read_pts(r"G:\facialdb\dataset\original\db_ibug\ibug\image_003_1.pts")

    # Overlapping points by component
    innerface_indexes = [1,2,3,4,5]
    eye_indexes = [1,2,3,4,5]
    shape1_points = shape2.get_points( indexes )


    pass

if __name__ == '__main__':
    main()
