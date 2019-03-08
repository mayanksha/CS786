#!/usr/bin/env python
import sys
import math 
import numpy as np
import cv2
from sklearn import cluster

red, green, blue = (0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255)
RANDOMIZE = False
size = 20 

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("-n" ,"--num_objects", help="The number of objects to generate", default="10")
parser.add_argument("--conjunction", help="Supply this flag to simulate conjunction search", action="store_true")
parser.add_argument("--feature", help="0 for color different, 1 for shape different", )
args = parser.parse_args()

def generateImage(x, y, n_channels):
    img_height, img_width = x, y 
    image = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    return image

# pt1 = (np.random.randint(size / 2, size), np.random.randint(0, size / 2))
# pt2 = (np.random.randint(0, size / 2), np.random.randint(size, size * 1.5))
# pt3 = (np.random.randint(size, size * 1.5), np.random.randint(size, size * 1.5))
def generateTriangle(color):
    pt1 = (int(0.75 * size), int(0.5 * size))
    pt2 = (int(0.5 * size), int(1.5 * size))
    pt3 = (int(1.5 * size), int(1.5 * size))
    image = generateImage(size * 1.5, size * 1.5, 4) 

    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.fillConvexPoly(image, triangle_cnt, color)
    return image

def generateSquare(color):
    image = generateImage(size, size, 4) 
    pt1 = (0, 0)
    pt2 = (0, size)
    pt3 = (size, size)
    pt4 = (size, 0)
    square_cnt = np.array( [pt1, pt2, pt3, pt4] )
    cv2.fillConvexPoly(image, square_cnt, color)
    return image

def placeImage(l_img, s_img, offsets):
    x_offset, y_offset = offsets[0], offsets[1]
    x1, x2 = x_offset, x_offset + s_img.shape[0]
    y1, y2 = y_offset, y_offset + s_img.shape[1]

    alpha_s = 1 + s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])
    return l_img

if __name__ == '__main__':
    print("")
    if '--conjunction' in sys.argv and '--feature' in sys.argv:
        print("Both feature and conjunction not allowed at same time!")
        sys.exit(-1)
    if '--conjunction' not in sys.argv and '--feature' not in sys.argv:
        print("Atleast one of --conjunction or --feature flag is needed")
        print("Do python q2.py --help for help.")
        sys.exit(-1)
 
    num_objects = int(args.num_objects)
    grid_size = 1 + num_objects / 2 
    pixels = (size * 3) * grid_size

    image = generateImage(pixels, pixels, 4) 
    square_pos =  np.random.randint(0, num_objects)

    x, y = [], [] 
    for i in range(size, pixels - 2 * size, 3*size):
        for j in range(size, pixels - 2 * size, 3*size):
            row = np.random.randint(i + 0.5 * size, i + 3 * size)
            col = np.random.randint(j + 0.5 * size, j + 3 * size)
            x.append(row)
            y.append(col)

    all_locations = []
    for i in range(min(len(x), len(y))):
        all_locations.append([x[i], y[i]])
    chosen_locations = []
    for i in range(0, num_objects):
        while True:
            index = np.random.randint(0, len(all_locations))
            x, y = all_locations[index][0], all_locations[index][1]
            del(all_locations[index])
            if y <= pixels and y >= size and x <= pixels and x >= size:
                chosen_locations.append([x, y])
                break
  
    # print(chosen_locations)

    if args.feature != None: 
        print("-------- FEATURE SEARCH --------")
        for i in range(0, num_objects - 1):
            triangle = generateTriangle(blue)
            tri_cent = (chosen_locations[i][0], chosen_locations[i][1])
            image = placeImage(image, triangle, tri_cent)
        if args.feature == "0":
            triangle = generateTriangle(red)
            tri_cent = (chosen_locations[-1][0], chosen_locations[-1][1])
            image = placeImage(image, triangle, tri_cent)
        else:
            square = generateSquare(blue)
            sqr_cent = (chosen_locations[-1][0], chosen_locations[-1][1])
            image = placeImage(image, square, sqr_cent)
    elif args.conjunction != None: 
        print("-------- CONJUNCTION SEARCH --------")
        # So that triangles and squares occur in nearly equal number
        num_triangles = np.random.randint(0.25 * num_objects, 0.75 * num_objects) 
        num_squares = num_objects - num_triangles - 1
        print("Number of Triangles (Randomly chosen) = ", num_triangles)
        print("Number of Squares (Randomly chosen) = ", num_squares + 1)
        for i in range(0, num_triangles):
            triangle = generateTriangle(blue)
            tri_cent = (chosen_locations[i][0], chosen_locations[i][1])
            image = placeImage(image, triangle, tri_cent)
        for i in range(0, num_squares):
            square = generateSquare(red)
            sqr_cent = (chosen_locations[i + num_triangles][0], chosen_locations[i + num_triangles][1])
            image = placeImage(image, square, sqr_cent)
        square = generateSquare(blue)
        sqr_cent = (chosen_locations[-1][0], chosen_locations[-1][1])
        image = placeImage(image, square, sqr_cent)

    img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    cv2.imshow("image.png", image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    cv2.imwrite("./image.png", img)
