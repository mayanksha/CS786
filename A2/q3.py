#!/usr/bin/env python
import q2
import q1
import sys
import os 
import math 
import numpy as np
import cv2
from sklearn import cluster
import time 

benchmark = False 
print(sys.argv)
if '--benchmark' in sys.argv:
    benchmark = True 

red, green, blue = (0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255)
RANDOMIZE = False
size = q2.size 
# Returns -1 for blue, 0 for green and 1 for red
def identify_color(image):
    x_mid, y_mid = (image.shape[0])/2, (image.shape[0])/2
    mid_pixel_color = image[y_mid, x_mid]
    return (np.argmax(mid_pixel_color) - 1)

if __name__ == '__main__':
    Search_Type = {}
    Search_Type['Conjunction'] = [] 
    Search_Type['Feature'] = [] 
    if benchmark == False:
        img_fn = "./q3_input_image.png"
        csv_file = "./q3_input_locations.csv"

        try:
            image = cv2.imread(img_fn)
        except:
            print("Input image missing. Generate that image using q2.py firstly. Exiting.")
            sys.exit(-1) 

        points = '' 
        try:
            points = np.genfromtxt(csv_file, delimiter=',')
        except:
            print("Input csv missing. Generate that csv using q2.py firstly. Exiting.")
            sys.exit(-1) 

        shapes = {}
        shapes["Blue Squares"] = 0
        shapes["Red Squares"] = 0
        shapes["Blue Triangles"] = 0
        shapes["Red Triangles"] = 0

        t0 = time.time()
        for pt in points:
            time.sleep(0.03)
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(image, (x, y) , 2, green)
            crop_img = image[y: int(y + 1.6 * size), x : int(x + 1.6 * size)]
            # cv2.imshow(str(x), crop_img)
            color = identify_color(crop_img)
            if color == -1:
                color = "Blue"
            elif color == 0:
                color = "Green"
            else:
                color = "Red"

            max_index, gabor_filter_img, lined_image, sse = q1.predict(crop_img)
            geometry = q1.detect_object(max_index)

            shapes[color + " " + geometry + "s"] += 1
            # cv2.imshow('Gabor Filtered Image', gabor_filter_img)
            # if cv2.waitKey(0) & 0xff == 27:
                # cv2.destroyAllWindows()

            # cv2.imshow('Intersection Points of Gabor Filter Image', lined_image)
            # if cv2.waitKey(0) & 0xff == 27:
                # cv2.destroyAllWindows()

            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()

        total_obj = 0
        for i in shapes.keys():
            total_obj += shapes[i]
        print(shapes)
        if shapes['Blue Squares'] == 1 and shapes['Red Squares'] == 0:
            search_type = "Feature"
        elif shapes['Blue Squares'] == 0 and shapes['Red Triangles'] == 1:
            search_type = "Feature"
        elif shapes['Blue Squares'] == 1 and shapes['Red Squares'] != 0:
            search_type = "Conjunction"
            time.sleep(0.05 * total_obj)

        print("The search type used = %s" % (search_type))

        t1 = time.time()
        total_time = t1 - t0

        print("The total time elapsed in search = %f seconds" % (total_time))
        image = q2.image_resize(image, height=800)
        cv2.imshow(img_fn, image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    ## Code to Benchmark
    else:
        num_obj = [10, 20, 30, 40, 50]
        for i in 2 * len(num_obj):
            if i >= 5:
                os.system("python ./q2.py -n %d --conjunction --hide_image" % (num_obj[i % 5]))
            else:
                os.system("python ./q2.py -n %d --feature %d" % (num_obj[i % 5], 0))
            img_fn = "./q3_input_image.png"
            csv_file = "./q3_input_locations.csv"

            try:
                image = cv2.imread(img_fn)
            except:
                print("Input image missing. Generate that image using q2.py firstly. Exiting.")
                sys.exit(-1) 

            points = '' 
            try:
                points = np.genfromtxt(csv_file, delimiter=',')
            except:
                print("Input csv missing. Generate that csv using q2.py firstly. Exiting.")
                sys.exit(-1) 

            shapes = {}
            shapes["Blue Squares"] = 0
            shapes["Red Squares"] = 0
            shapes["Blue Triangles"] = 0
            shapes["Red Triangles"] = 0

            t0 = time.time()
            for pt in points:
                time.sleep(0.03)
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(image, (x, y) , 2, green)
                crop_img = image[y: int(y + 1.6 * size), x : int(x + 1.6 * size)]
                # cv2.imshow(str(x), crop_img)
                color = identify_color(crop_img)
                if color == -1:
                    color = "Blue"
                elif color == 0:
                    color = "Green"
                else:
                    color = "Red"

                max_index, gabor_filter_img, lined_image, sse = q1.predict(crop_img)
                geometry = q1.detect_object(max_index)

                shapes[color + " " + geometry + "s"] += 1
                # cv2.imshow('Gabor Filtered Image', gabor_filter_img)
                # if cv2.waitKey(0) & 0xff == 27:
                    # cv2.destroyAllWindows()

                # cv2.imshow('Intersection Points of Gabor Filter Image', lined_image)
                # if cv2.waitKey(0) & 0xff == 27:
                    # cv2.destroyAllWindows()

                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

            total_obj = 0
            for i in shapes.keys():
                total_obj += shapes[i]
            print(shapes)
            if shapes['Blue Squares'] == 1 and shapes['Red Squares'] == 0:
                search_type = "Feature"
            elif shapes['Blue Squares'] == 0 and shapes['Red Triangles'] == 1:
                search_type = "Feature"
            elif shapes['Blue Squares'] == 1 and shapes['Red Squares'] != 0:
                search_type = "Conjunction"
                time.sleep(0.05 * total_obj)

            print("The search type used = %s" % (search_type))

            t1 = time.time()
            total_time = t1 - t0
            Search_Type[search_type].append(total_time)
            print("The total time elapsed in search = %f seconds" % (total_time))
            image = q2.image_resize(image, height=800)
            cv2.imshow(img_fn, image)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()
        print(Search_Type)
