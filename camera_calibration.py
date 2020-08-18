# Calibration for 2 cameras in a stereo setup. Based on https://albertarmea.com/post/opencv-stereo-camera/ and opencv tutorials.
# Arguments are directory for left calibration images, directory for right calibration images, and output file

import sys
import numpy
import cv2
import os
import glob

SIZE = (9, 6)
ALPHA = 0.5
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OBJECT_POINT_ZERO = numpy.zeros((SIZE[0] * SIZE[1], 3), numpy.float32)
OBJECT_POINT_ZERO[:, :2] = numpy.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)
OBJECT_POINT_ZERO = OBJECT_POINT_ZERO #* 5
leftdir = sys.argv[1]
rightdir = sys.argv[2]
output = sys.argv[3]

def calibrate(dir):
    f = os.path.join(dir, "cache.npz")
    if os.path.exists(f):
        print(f)
        cache = numpy.load(f)
        return (cache["files"], cache["objpoints"], cache["imgpoints"], cache["size"].tolist())
    
    images = glob.glob(dir + "/*.jpg")
    objpoints = []
    imgpoints = []
    files=[]
    size = None
    for image in sorted(images):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        
        check, corners = cv2.findChessboardCorners(gray, SIZE)
        if check:
            objpoints.append(OBJECT_POINT_ZERO)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
            imgpoints.append(corners)
            files.append(os.path.basename(image))
        cv2.drawChessboardCorners(img, SIZE, corners, check)
        print(check)
        print(image)
        cv2.imshow(dir, img)
        cv2.waitKey(1)
    numpy.savez_compressed(f, files=files, objpoints=objpoints, imgpoints=imgpoints, size=size)
    return files, objpoints, imgpoints, size
(lfiles, lobjpoints, limgpoints, lsize) = calibrate(leftdir)
(rfiles, robjpoints, rimgpoints, rsize) = calibrate(rightdir)
files = list(set(lfiles) & set(rfiles))
print(files)
def matchpoints(files, allfiles, objpoints, imgpoints):
    fileset = set(files)
    newobjpoints = []
    newimgpoints = []

    for i, f in enumerate(allfiles):
        if f in fileset:
            newobjpoints.append(objpoints[i])
            newimgpoints.append(imgpoints[i])

    return newobjpoints, newimgpoints
lobjpoints, limgpoints = matchpoints(files, lfiles, lobjpoints, limgpoints)
robjpoints, rimgpoints = matchpoints(files, rfiles, robjpoints, rimgpoints)
objpoints = lobjpoints
size = tuple(lsize)
objpoints = objpoints
_, lcameramtx, ldist, _, _ = cv2.calibrateCamera(objpoints, limgpoints, size, None, None)
_, rcameramtx, rdist, _, _ = cv2.calibrateCamera(objpoints, rimgpoints, size, None, None)
(_, _, _, _, _, rotationmtx, translationv, E, F) = cv2.stereoCalibrate(objpoints, limgpoints, rimgpoints, lcameramtx, ldist, rcameramtx, rdist, size, None, None, None, None, cv2.CALIB_FIX_INTRINSIC, TERM_CRITERIA)
print('E')
print(E)
print('F')
print(F)
print('T')
print(translationv)
print('R')
print(rotationmtx)
print('l')
print(lcameramtx)
print('r')
print(rcameramtx)
(lrectification, rrectification, lprojection, rprojection, Q, lroi, rroi) = cv2.stereoRectify(lcameramtx, ldist, rcameramtx, rdist, size, rotationmtx, translationv, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, ALPHA)

lmapx, lmapy = cv2.initUndistortRectifyMap(lcameramtx, ldist, lrectification,lprojection, size, cv2.CV_32FC1)
rmapx, rmapy = cv2.initUndistortRectifyMap(rcameramtx, rdist, rrectification,rprojection, size, cv2.CV_32FC1)

numpy.savez_compressed(output, size=size, lmapx=lmapx, lmapy=lmapy, lroi=lroi, rmapx=rmapx, rmapy=rmapy, rroi=rroi, Q=Q, F=F, E=E, lprojection=lprojection, rprojection=rprojection, rdist=rdist, ldist=ldist, rcameramtx=rcameramtx, lcameramtx=lcameramtx, lrectification=lrectification, rrectification=rrectification)
cv2.destroyAllWindows()

