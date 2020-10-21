import cv2
import tensorflow as tf
import numpy as np
import sys
import time


sys.path.append("..")

def midpoint(pt1,pt2):
    return int((pt1+pt2)/2)

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# The TensorRT inference graph file downloaded from Colab or your local machine.
pb_fname = "inference_graph/frozen_inference_graphchildandfork.pb"
#pb_fname = tf.GraphDef()
trt_graph = get_frozen_graph(pb_fname)

input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def non_max_suppression(boxes, probs=None, nms_threshold=0.9):
    """Non-max suppression

    Arguments:
        boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
    Keyword arguments
        probs {np.array} -- Probabilities associated with each box. (default: {None})
        nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

    Returns:
        list -- A list of selected box indexes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick

def detect(img):
    image = img
    final_points=[0,0]
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None, ...]
    })
    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    trt_graph = get_frozen_graph(pb_fname)
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None]
    })
    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])


    boxes_pixels = []
    for i in range(num_detections):
        # scale box to image coordinates
        box = boxes[i] * np.array([image.shape[0],
                                image.shape[1], image.shape[0], image.shape[1]])
        box = np.round(box).astype(int)
        boxes_pixels.append(box)
    boxes_pixels = np.array(boxes_pixels)

    # Remove overlapping boxes with non-max suppression, return picked indexes.
    pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.9)


    min_threshold = 0.0
    child_flag=1
    fork_flag=1
    child_final_points=[0,0]
    fork_final_points=[0,0]
    for i in pick:
        if scores[i] > min_threshold:
            box = boxes_pixels[i]
            box = np.round(box).astype(int)
            #order ymin ,xmin ,ymax and xmax
            if(classes[i] == 2 and fork_flag == 1):
                print("fork:",scores[i])
                image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
                fork_final_points=[0,0]
                fork_final_points[1]=midpoint(box[0],box[2])
                fork_final_points[0]=midpoint(box[1],box[3]) 
                fork_flag=0
            if(classes[i] == 1 and child_flag == 1):
                print("Child:",scores[i])
                image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                child_final_points=[0,0]
                child_final_points[1]=midpoint(box[0],box[2])
                child_final_points[0]=midpoint(box[1],box[3]) 
                child_flag=0
            if(child_flag == 0 and fork_flag == 0):
                break
        else:
            continue
    cv2.namedWindow("Detection",cv2.WINDOW_NORMAL)
    cv2.imshow("Detection",image)
    time.sleep(0)
    return child_final_points ,fork_final_points

