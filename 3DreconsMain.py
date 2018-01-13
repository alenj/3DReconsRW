import cv2
import numpy as np
from matplotlib import path
from matplotlib import pyplot as plt
import scipy
import math
from scipy import linalg, matrix

## functions


def compute_P_from_fundamental(F):
    """  Computes the second camera matrix (assuming P1 = [I 0])    from a fundamental matrix. """
    e = compute_epipole(F.T) # left epipole
    Te = np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
    P= np.vstack((np.dot(Te,F.T).T,e)).T
    return P


def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from
    least squares solution. """
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in
    x1,x2 (3*n homog. coordinates). """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    X = [triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X).T


def compute_epipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]




def project(P, X):
    x = np.dot(P, X)
    for i in range(3):
        x[i] /= x[2]
    return x




def DD2XYZ(lat, lon, alt):
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

    lat = np.radians(lat)
    lon = np.radians(lon)

    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF     = (1.0-f)**2
    C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
    S      = C * FF

    x = (rad * C + alt)*cosLat * np.cos(lon)
    y = (rad * C + alt)*cosLat * np.sin(lon)
    z = (rad * S + alt)*sinLat

    return np.array([x,y,z])


def rotation(a,b):
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)
    I = np.identity(3)
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    k = np.matrix(vXStr)
    return I + k + np.matmul(k,k) * ((1 -c)/(s**2))

def angleV(v1, v2):

    c_theta = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.arccos(np.clip(c_theta, -1.0, 1.0))


## Main
img1 = cv2.imread('9_28.16431133534867_-97.0113386341318_2011-05_90.jpg',0)  #queryimage # left image
img2 = cv2.imread('10_28.16431128887001_-97.01144035367199_2011-05_90.jpg',0) #trainimage # right image

sift = cv2.xfeatures2d.SIFT_create()


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)




pts1 = np.array(pts1)
pts2 = np.array(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]



xmin1 = 1110
ymin1 = 891
xmax1 = 1518
ymax1 = 1135


p1 = path.Path([(xmin1,ymin1), (xmin1, ymax1), (xmax1, ymax1), (xmax1, ymin1)])
flags1 = p1.contains_points(pts1)
ind1 = [i for i, x in enumerate(flags1) if x]
hpts1 = pts1[ind1]
hpts2 = pts2[ind1]

print(pts1.shape)
print(pts2.shape)

print(hpts1.shape)
print(hpts2.shape)


##################

# compute camera matrices
P1 = np.array([[1.0000000000,0.0000000000,0.0000000000,0.0000000000],[0.0000000000,1.0000000000,0.0000000000,0.0000000000],[0.0000000000,0.0000000000,1.0000000000,0.00000000]])
P2 = compute_P_from_fundamental(F)


x1 = np.vstack((hpts1.T,np.ones(hpts1.shape[0])))
x2 = np.vstack((hpts2.T,np.ones(hpts2.shape[0])))

# X = cv2.triangulatePoints(P1[:3],P2[:3],x1[:2],x2[:2])

def null(A, eps=1e-12):
    u, s, vh = np.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.asarray(null_space)

def triangulatePoints( P1, P2, x1, x2 ):
    X = cv2.triangulatePoints(P1[:3],P2[:3],x1[:2],x2[:2])
    return X

X = triangulatePoints(P1,P2,x1,x2)
X3D = X.T[:, :3]/np.repeat(X.T[:, 3], 3).reshape(-1, 3)


print(X.T)
print(X3D)

x1p = project(P1,X).T
x2p = project(P2,X).T
#
# M = np.dot(x2p,(np.linalg.pinv(x1p)))
# print(M)

## Getting the center of camera

ro = P2[:,:3]
tr = P2[:,3]

c2= null(ro)


# plt.figure()
# plt.imshow(img1)
# plt.gray()
# plt.plot(x1p[:,0],x1p[:,1],'o')
# plt.plot(hpts1[:,0],hpts1[:,1],'r.')
# plt.axis('off')
# plt.figure()
# plt.imshow(img2)
# plt.gray()
# plt.plot(x2p[:,0],x2p[:,1],'o')
# plt.plot(hpts2[:,0],hpts2[:,1],'r.')
# plt.axis('off')
# plt.show()
#
# print(np.abs(np.sum(x1p)))
# print(np.abs(np.sum(x2p)))
# print(np.abs(np.sum(x1)))
# print(np.abs(np.sum(x2)))


###############
g1 =np.array((28.16431133534867,-97.0113386341318,1))
g2 =np.array((28.16431128887001,-97.01144035367199,1))

lat1=28.164311335348670
lon1=-97.01133863413180
alt1=1.0000000000000000

lat2=28.164311288870010
lon2=-97.01144035367199
alt2=1.0000000000000000

gps1 = DD2XYZ(lat1,lon1,alt1)
gps2 = DD2XYZ(lat2,lon2,alt2)
print(gps1)


c=c2.flatten()
g = gps2-gps1
# T = rotation(c,g)

s = np.norm
print(T)
XP = np.dot(X3D,T)

# print(X3D)
print(XP)



v1 = np.subtract(gps2, gps1)
v2 = np.subtract(XP[1,:], gps1)
# v2p = np.subtract(XP, gps1)

print(v1.T)
print(v2.T)
# print(v2p.T)
theta = np.degrees(angleV(v1.T, v2.T))
# thetaP= angleV(v1.T, v2p.T)
print(theta)
# print(thetaP)