import numpy as np
import pandas as pd
import os
from load_data import *
import copy


def skeleton_format( skeleton_format = "coco"):
    if skeleton_format == "coco" or skeleton_format == "COCO":
        keypoints_order = { 1: "nose", 2: "left_eye", 3: "right_eye", 4: "left_ear",
                 5: "right_ear", 6: "left_shoulder", 7: "right_shoulder",
                 8: "left_elbow", 9: "right_elbow", 10: "left_wrist",
                 11: "right_wrist", 12: "left_hip", 13: "right_hip",
                 14: "left_knee", 15: "right_knee", 16: "left_ankle",
                 17: "right_ankle" }
        connections = ((16,14), (14,12), (17,15), (15,13), (12,13), (6,12),
                        (7,13), (6,7), (6,8), (7,9), (8,10), (9,11), (2,3),
                        (1,2), (1,3), (2,4), (3,5), (4,6), (5,7))

    elif skeleton_format == "mpii" or skeleton_format == "MPII":
        keypoints_order = { 0: "r ankle", 1: "r knee", 2: "r hip", 3: "l hip",
                            4: "l knee", 5: "l ankle", 6: "pelvis", 7: "thorax",
                            8: "upper neck", 9: "head top", 10: "r wrist",
                            11: "r elbow", 12: "r shoulder", 13: "l shoulder",
                            14: "l elbow", 15: "l wrist" }
        connections = ((0,1), (1,2) , (2,6), (6,3), (3,4), (4,5), (6,7), (7,8),
                      (8,9), (10,11), (11,12), (12,7), (7,13), (13,14), (14,15))
    else:
        print("ERROR: estimation_skeleton_format (3rd arg) not recognised")

    return keypoints_order, connections;

def evaluate_pcp( gt, estimation_results, estimation_skeleton_format = "coco"):
    '''
    inputs: gt - loaded groundtruth data
            estimation_results - path to file with all results or to directory
                                 which constains such files
            estimation_skeleton_format - can be "coco" or "mpii"
    '''
    (keypoints_order, connections) = skeleton_format(estimation_skeleton_format)

    # TODO:get all images in dataset
    images_by_dist = get_image_groups_by_dist(gt)

    score_by_dist = dict.fromkeys(list(images_by_dist.keys()))

    for dist, images in images_by_dist.items():

        results = {k: 0 for k in connections}
        total = {k: 0 for k in connections}
        score = dict.fromkeys(connections)

        for img in images:
            #gt_keypoints = {i+1:j for i,j in enumerate(list(gt.loc[ i , :])[1:18])}
            gt_keypoints = dict(zip([int(i) for i in gt.columns[0:17]],
                                list(gt.loc[ img , :])[0:17]))
            est_keypoints = dict(zip(
                            [int(i) for i in estimation_results.columns[0:17]],
                            list(estimation_results.loc[ img , :])[0:17]))
            #est_keypoints = list(estimation_results.loc[ i , :])[0:17]

            for c in connections:
                if not(gt_keypoints[c[0]] is None or gt_keypoints[c[1]] is None):
                    if not(gt_keypoints[c[0]][2] or gt_keypoints[c[1]][2]):
                        total[c]+=1
                        delta_x = gt_keypoints[c[0]][0] - gt_keypoints[c[1]][0]
                        delta_y = gt_keypoints[c[0]][1] - gt_keypoints[c[1]][1]
                        gt_dist = np.sqrt(delta_x**2 + delta_y**2)
                    else:
                        continue
                else:
                    continue
                if not(est_keypoints[c[0]] is None or est_keypoints[c[1]] is None):
                    delta_x = est_keypoints[c[0]][0] - gt_keypoints[c[0]][0]
                    delta_y = est_keypoints[c[0]][1] - gt_keypoints[c[0]][1]
                    est_dist = np.sqrt(delta_x**2 + delta_y**2)
                    if est_dist > (gt_dist/2):
                        continue
                    delta_x = est_keypoints[c[1]][0] - gt_keypoints[c[1]][0]
                    delta_y = est_keypoints[c[1]][1] - gt_keypoints[c[1]][1]
                    est_dist = np.sqrt(delta_x**2 + delta_y**2)
                    if est_dist <= (gt_dist/2):
                        results[c] += 1
            for c in connections:
                if total[c] != 0:
                    score[c] = results[c]/total[c]
                else:
                    score[c] = None
        score_by_dist[dist] = score

    return score_by_dist

def evaluate_pcpm( gt, estimation_results, estimation_skeleton_format = "coco"):
    '''
    inputs: gt - loaded groundtruth data
            estimation_results - path to file with all results or to directory
                                 which constains such files
            estimation_skeleton_format - can be "coco" or "mpii"
    '''
    (keypoints_order, connections) = skeleton_format(estimation_skeleton_format)

    # TODO:get all images in dataset
    images_by_dist = get_image_groups_by_dist(gt)

    score_by_dist = dict.fromkeys(list(images_by_dist.keys()))
    for dist, images in images_by_dist.items():

        results = {k: 0 for k in connections}
        total = {k: 0 for k in connections}
        score = dict.fromkeys(connections)
        thres = {k: 0 for k in connections}

        for img in images:
            # TODO: get keypoints
            gt_keypoints = dict(zip([int(a) for a in gt.columns[0:17]],
                                list(gt.loc[ img , :])[0:17]))

            for c in connections:
                if not(gt_keypoints[c[0]] is None or gt_keypoints[c[1]] is None):
                    if not(gt_keypoints[c[0]][2] or gt_keypoints[c[0]][2]):
                        total[c]+=1
                        delta_x = gt_keypoints[c[0]][0] - gt_keypoints[c[1]][0]
                        delta_y = gt_keypoints[c[0]][1] - gt_keypoints[c[1]][1]
                        thres[c] += np.sqrt(delta_x**2 + delta_y**2)
        for c in connections:
            if total[c] != 0:
                thres[c] = (thres[c]/total[c])/2
            else:
                thres[c] = 0

        for img in images:
            est_keypoints = dict(zip(
                            [int(a) for a in estimation_results.columns[0:17]],
                            list(estimation_results.loc[ img , :])[0:17]))

            for c in connections:
                if not(gt_keypoints[c[0]] is None or est_keypoints[c[0]] is \
                   None or gt_keypoints[c[1]] is None or est_keypoints[c[1]]):
                    delta_x = est_keypoints[c[0]][0] - gt_keypoints[c[0]][0]
                    delta_y = est_keypoints[c[0]][1] - gt_keypoints[c[0]][1]
                    est_dist = np.sqrt(delta_x**2 + delta_y**2)
                    if est_dist > thres[c]:
                        continue
                    delta_x = est_keypoints[c[1]][0] - gt_keypoints[c[1]][0]
                    delta_y = est_keypoints[c[1]][1] - gt_keypoints[c[1]][1]
                    est_dist = np.sqrt(delta_x**2 + delta_y**2)
                    if est_dist <= thres[c]:
                        results[c] += 1
        for c in connections:
            if total[c] != 0:
                score[c] = results[c]/total[c]
            else:
                score[c] = None
        score_by_dist[dist] = score

    return score_by_dist

def evaluate_pck( gt, estimation_results, estimation_skeleton_format = "coco",
   threshold_type = "bbox"):
    '''
    inputs: gt - loaded groundtruth data
            estimation_results - path to file with all results or to directory
                                 which constains such files
            estimation_skeleton_format - can be "coco" or "mpii"
            threshold_type - can be "bbox", for 50% of bounding box height, or
                            "h", for 50% of head segment length
    '''
    keypoints_order, connections = skeleton_format(estimation_skeleton_format)

    # TODO:get all images in dataset
    images_by_dist = get_image_groups_by_dist(gt)

    score_by_dist = dict.fromkeys(list(images_by_dist.keys()))

    for dist, images in images_by_dist.items():
        total_keypoints = 0
        detected_keypoints = 0

        for img in images:
            #get keypoints
            gt_keypoints = dict(zip([int(i) for i in gt.columns[0:17]],
                                list(gt.loc[ img , :])[0:17]))
            est_keypoints = dict(zip(
                            [int(i) for i in estimation_results.columns[0:17]],
                            list(estimation_results.loc[ img , :])[0:17]))

            # TODO: calculate target height (divide by 2 here)
            if threshold_type == "bbox":
                l = dict(zip([int(i) for i in gt.columns],list(gt.loc[ img , :])))
                #top = max([i[1] for i in l.values() if i != None])
                top = max([i[1] for i in l.values()])
                #bottom = min([i[1] for i in l.values() if i != None])
                bottom = min([i[1] for i in l.values()])
                #print("top: " + str(top) + " bottom: " + str(bottom))
                thres = (top-bottom)/2
            elif threshold_type == "h":
                l = list(gt.loc[img,[20,21]])
                thres_x = l[0][0]-l[1][0]
                thres_y = l[0][1]-l[1][1]
                thres = (thres_x**2+thres_y**2)/2
            else:
                print("ERROR: threshold_type not recognized")

            for k in keypoints_order.keys():
                if gt_keypoints[k] is not None:
                    total_keypoints += 1
                    if est_keypoints[k] is not None:
                        delta_x = gt_keypoints[k][0] - est_keypoints[k][0]
                        delta_y = gt_keypoints[k][1] - est_keypoints[k][1]
                        est_dist = np.sqrt(delta_x**2 + delta_y**2)

                        if est_dist <= thres:
                            detected_keypoints += 1
        if total_keypoints != 0:
            score = detected_keypoints/total_keypoints
        else:
            score = None
        score_by_dist[dist]=score
    return score_by_dist

def evaluate_eucl_dist( gt, estimation_results, estimation_skeleton_format = "coco"):
    '''
    inputs: gt - loaded groundtruth data
            estimation_results - path to file with all results or to directory
                                 which constains such files
            estimation_skeleton_format - can be "coco" or "mpii"
    '''
    keypoints_order, connections = skeleton_format(estimation_skeleton_format)

    # TODO:get all images in dataset
    images_by_dist = get_image_groups_by_dist(gt)

    score_by_dist = dict.fromkeys(list(images_by_dist.keys()))

    for dist, images in images_by_dist.items():
        score = 0

        for img in images:
            gt_keypoints = dict(zip([int(i) for i in gt.columns[0:17]],
                                list(gt.loc[ img , :])[0:17]))
            est_keypoints = dict(zip(
                            [int(i) for i in estimation_results.columns[0:17]],
                            list(estimation_results.loc[ img , :])[0:17]))

            total_k_in_image = 0
            euc_dist = 0

            for k in keypoints_order.keys():
                if not(gt_keypoints[k] is None or est_keypoints[k] is None):
                        total_k_in_image += 1
                        delta_x = gt_keypoints[k][0] - est_keypoints[k][0]
                        delta_y = gt_keypoints[k][1] - est_keypoints[k][1]
                        euc_dist += np.sqrt(delta_x**2 + delta_y**2)
            if total_k_in_image != 0:
                score += euc_dist/total_k_in_image
            else:
                score += 0
        score_by_dist[dist] = score/len(images)

    return score_by_dist
