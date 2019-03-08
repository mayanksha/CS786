#!/usr/bin/env python
 
import sys
import numpy as np
import numpy.matlib 
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score

def build_filters():
    filters = []
    ksize = 10 
    sigma = 5.0
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 5*kern.sum()
        filters.append(kern)
    return filters
def detect_object(val):
    if val == 1:
        return("Triangle")
    elif val == 2:
        return("Square")
    else:
        return("Some other shape")

def intersection(line1, line2):
    rho1, theta1 = line1[0], line1[1]
    rho2, theta2 = line2[0], line2[1]

    if (abs(theta1 - theta2) < (np.pi/6)):
        return [None, None]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    filters = build_filters()
    res1 = process(img, filters)

    gabor_filter_img = res1
    gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    ret,thresh2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,thresh1 = cv2.threshold(thresh2,0,255,cv2.THRESH_BINARY)
    # ret,thresh1 = cv2.threshold(thresh1,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # thresh1 = cv2.erode(thresh1, None)
    # cv2.imshow('thresh1', thresh1)
    # if cv2.waitKey(0) & 0xff == 27:
        # cv2.destroyAllWindows()

    rho, theta, thresh = 7, np.pi/180, 175 
    lines = cv2.HoughLines(thresh1, rho, theta, thresh)

    for line in lines:
        for r, theta in line: 
            a = np.cos(theta) 
            b = np.sin(theta) 
            x0 = a*r 
            y0 = b*r 
            x1 = int(x0 + 1000*(-b)) 
            y1 = int(y0 + 1000*(a)) 
            x2 = int(x0 - 1000*(-b)) 
            y2 = int(y0 - 1000*(a)) 
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0),2) 

    X = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    x = intersection(line1, line2)
                    if x[0] != None:
                        X.append(x)

    for i in X:
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0))

    lined_image = img
    sse = {}
    for k in range(2, 7):
        kmeans = cluster.KMeans(n_clusters=k, max_iter=1000).fit(X)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    curve = list(sse.values()) 
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    max_index = np.argmax(distToLine)
    return max_index, gabor_filter_img, lined_image, sse

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape" , help="1 for Square, 0 for triangle", default="0")
    args = parser.parse_args()

    print(args)
    if args.shape == "1":
        img_fn = './square.png'
    else:
        img_fn = './triangle.png'

    img = cv2.imread(img_fn)
    if img is None:
        print 'Failed to load image file:', img_fn
        sys.exit(1)

    max_index, gabor_filter_img, lined_image, sse = predict(img)

    cv2.imshow('Gabor Filtered Image', gabor_filter_img)
    cv2.imwrite('Gabor_square.png', gabor_filter_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    cv2.imshow('Intersection Points of Gabor Filter Image', lined_image)
    cv2.imwrite('hough_square.png', lined_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

    print(detect_object(max_index))

