import cv2
import numpy as np
import sqlite3
import os
import sys
import collections
import struct
import io
import statistics as st
import glob
import time
import tqdm
from matplotlib import pyplot as plt
from pythontools.geometry import rolling_shutter as rs
from pythontools.utils import ransac as ransac
from pythontools.geometry import utils as ut
from pythontools.geometry import rotation as rot

# the following functions are taken from COLMAP python scripts, Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2
    
def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

# END of COLMAP scripts

# here we define the workspace
# the folder containing the exported COLMAP result
datadir = '/home/alblc/cloud/Rolling-Shutter/real_experiments/seq01/gs/'
# COLMAP database file, such that the absolute path is datadir+datafile
datafile = 'gs.db'
# colmap exported 3D points
pointsfile = 'points3D.bin'
# do you want to also extract the original camera poses?
extract_poses = 1
imagesfile = 'images.bin'
# the directory which contains the images we want to register to the COLMAP model
ar_images_dir = '/home/alblc/cloud/Rolling-Shutter/real_experiments/seq01/gs/'
# output directory
out_dir = 'ar_output/'
try:  
    os.makedirs(out_dir)
except OSError:  
    print ("Creation of the directory %s failed - perhaps it already exists... " % out_dir)
else:  
    print ("Successfully created the directory %s " % out_dir)
# calibration file for the camera that took the images to be registered
# the standard output of opencv calibration pipeline containing mtx as the calibration matrix
calib_file = 'calibration_output.npz'
# open the database
print("Opening the database...")
conn = sqlite3.connect(datadir+datafile)
c = conn.cursor()
# extract keypoints and descriptors
print('Extracting keypoints and descriptors...')
res = c.execute('SELECT * FROM descriptors')
descriptors = []
ndesc = 0
for row in res.fetchall():
    descriptors.append(np.frombuffer(row[3], dtype=np.uint8))
    ndesc += int(descriptors[-1].shape[0]/128)
print("Found %d images and %d keypoints total, average %d keypoints per image" % (len(descriptors),ndesc,int(ndesc/len(descriptors))))
if extract_poses:
    P = []
    print("Reading camera poses...")
    images = read_images_binary(datadir + imagesfile)
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape(3,1)
        P.append(np.hstack((R,t)))

print("Reading 3D points...")
points3d = read_points3d_binary(datadir + pointsfile)
print("Processing the input data...")
idxs = []
for pt in points3d.values():
    idxs.append((pt.id,len(pt.image_ids)))
sortedPts = sorted(idxs,key=lambda idx: idx[1], reverse=True)
#using the sorted array we can also select N most visible points, but now we select all
selPts = sortedPts[0:-1]
# find which SIFT feature corresponds to them
xyz = []
kpts = []
kptcoords = []
idxmap = []
#keep track of the maximum number of occurences of one 3D point, for the NN search settings
maxcnt = 0
for pt in selPts:
    pt3D = points3d[pt[0]]
    for i in range(len(pt3D.image_ids)):
        imId = pt3D.image_ids[i]
        featId = pt3D.point2D_idxs[i]
        kpt = descriptors[imId-1][(featId)*128:(featId)*128+128]
        xyz.append(pt3D.xyz)
        kpts.append(kpt)
        idxmap.append(pt[0])
        if i > maxcnt:
            maxcnt = i
xyznp = np.array(xyz)
print("maximum number of occurences of a single 3D point is %d" % maxcnt)
calib = np.load(calib_file)
    
files_rs = sorted(glob.glob(ar_images_dir + '*.jpg'))
frid = 0
skipped = 0
corr2D = []
corr3D = []
print("Creating 3D-2D matches...")
for file in tqdm.tqdm(files_rs):
    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d_SIFT.create(0,3,0.03)
    kptsimg,desimg = sift.detectAndCompute(img,None)
    kpcoord = []
    for kp in kptsimg:
            kpcoord.append((kp.pt[0], kp.pt[1]))
    npcoord = np.array(kpcoord)
    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # Match descriptors.
    desScene = np.array(kpts,dtype = np.float32)
    matches = bf.knnMatch(desimg, desScene, k=maxcnt+1)
    matchesRes = []
    ratioThr = 0.75
    kpMatched = []
    matched3Dpts = []
    for m in matches:
        firstidx = idxmap[m[0].trainIdx]
        for i in range(len(m)):
            if(idxmap[m[i].trainIdx] != firstidx):
                break
        if(m[0].distance/m[i].distance<ratioThr):
            kpMatched.append(m[0].queryIdx)
            matched3Dpts.append(m[0].trainIdx)
    idxs = np.array(kpMatched)
    idxs3D = np.array(matched3Dpts)
    imgpts = []
    imgptsrs = []
    if idxs3D.shape[0]>0:
        matchedcoords = npcoord[idxs]
        corr2D.append(npcoord[idxs])
        corr3D.append(xyznp[idxs3D,:])
    else:
        skipped += 1
if extract_poses:
    np.savez(out_dir + "output", corr2D=corr2D,corr3D=corr3D,P=P)
else:
    np.savez(out_dir + "output", corr2D=corr2D,corr3D=corr3D)
print("Done! Saved to %s/output.npz. No correspondences for %d images" % (out_dir,skipped))
