#!/usr/bin/env python
 
import numpy as np
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
        kern /= 2.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

if __name__ == '__main__':
    import sys
    try:
        img_fn = sys.argv[1]
    except:
        img_fn = './sqr.png'

    img = cv2.imread(img_fn)
    if img is None:
        print 'Failed to load image file:', img_fn
        sys.exit(1)

    filters = build_filters()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

    res1 = process(bin_img, filters)

    rho, theta, thresh = 2, np.pi/180, 200 
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)

    cv2.imshow('result', res1)
    cv2.waitKey(0)

    dilated = cv2.dilate(res1, None)
    dilated = cv2.dilate(res1, None)
    eroded  = cv2.erode(res1, None)
    eroded  = cv2.erode(res1, None)
    dst = cv2.cornerHarris(eroded,2,3,0.04)
    img[dst>0.01*dst.max()] = [0,255,0]

    b = np.matrix(dst>0.01*dst.max())
    points = (zip(*np.where(b == True)))
    points = (zip(*np.where(b == True)))
    print(points)

    temp = np.zeros(img.shape)

    for point in points:
        cv2.circle(temp, point, 1, (0,0,255))

    cv2.imshow('corner points',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    X = np.matrix([[a[0], a[1]] for a in points])

    sil_avgs = [] 
    for n_clusters in range(3, 6, 1):
        kmeans = cluster.KMeans(n_clusters, random_state=0).fit(X)
        cluster_labels = kmeans.fit_predict(X)
        # plt.scatter([:,0], [:,1])

        sil_avg = silhouette_score(X, cluster_labels)
        sil_avgs.append(sil_avg) 
        print(sil_avg)
        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)
    sil_avgs = np.array(sil_avgs)
    max_index = np.argmax(sil_avgs)
    if max_index == 0:
        print("Triangle Found")
    elif max_index == 1:
        print("Square Found")
    else:
        print("Some other shape")


