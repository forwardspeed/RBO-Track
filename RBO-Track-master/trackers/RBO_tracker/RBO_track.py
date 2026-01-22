"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
import copy
from .association import *
from collections import deque       # [hgx0418] deque for reid feature

np.random.seed(0)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    if x[2] * x[3]>0:
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
    else:
        w = 1
        h = 1
    
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        score = x[4]
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

def fine_det(bbox_t, bbox_d, bbox_s):

    if bbox_t[2] == -1:
        return bbox_d
    
    v_1 = (bbox_d[0] + bbox_d[2])/2 - (bbox_s[0] + bbox_s[2])/2
    v_2 = (bbox_d[1] + bbox_d[3])/2 - (bbox_s[1] + bbox_s[3])/2
    
    old_w = bbox_t[2] - bbox_t[0]
    old_h = bbox_t[3] - bbox_t[1]
    
    z_fine = [-1, -1, -1, -1, bbox_d[4]]
    
    if v_1 < 0:
        z_fine[0] = bbox_d[0]
        z_fine[2] = z_fine[0] + old_w
    else:
        z_fine[2] = bbox_d[2]
        z_fine[0] = z_fine[2] - old_w

    z_fine[1] = bbox_d[1]
    z_fine[3] = z_fine[1] + old_h
    # if v_2 > 0:
    #     z_fine[1] = bbox_d[1]
    #     z_fine[3] = z_fine[1] + old_h
    # else:
    #     z_fine[3] = bbox_d[3]
    #     z_fine[1] = z_fine[3] - old_h
    return z_fine

def iou_two_boxes(box1, box2):

    box1, box2 = np.asarray(box1), np.asarray(box2)

    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, feat, uncertainty=1000., orig=False, args=None):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        # if not orig and not args.kalman_GPR:
        if not orig:
            from .kalmanfilter import KalmanFilter as KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= uncertainty  # give high uncertainty to the unobservable initial velocities
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        # self.observations = dict()
        # self.history_observations = []
        self.confidence = bbox[-1]
        self.args = args
        self.oc_time = 0
        self.accident = 0
        # momentum of embedding update
        self.alpha = self.args.alpha
        # add the following values and functions
        self.feat = feat
        # self.update_features(feat)

    # ReID. for update embeddings during tracking
    def update_features(self, feat, score=-1, siou=0):
        feat /= np.linalg.norm(feat)
        if self.feat is None:
            self.feat = feat
        elif score > 0 :
            if score > self.confidence:
                score = score * (1-siou)
                self.feat = ((2) / (2 + score)) * self.feat + (score/(2 + score)) * feat
            else:
                self.feat = ((5) / (5 + score)) * self.feat + (score/(5 + score)) * feat
        else:
            self.feat = self.alpha * self.feat + (1 - self.alpha) * feat
        self.feat /= np.linalg.norm(self.feat)

    # def update_features(self, feat, score=-1, siou=0):
    #     feat /= np.linalg.norm(feat)
    #     if self.feat is None:
    #         self.feat = feat
    #     else:
    #         if self.args.adapfs:
    #             assert score > 0
    #             pre_w = self.alpha * (self.confidence / (self.confidence + score))
    #             cur_w = (1 - self.alpha) * (score / (self.confidence + score))
    #             sum_w = pre_w + cur_w
    #             pre_w = pre_w / sum_w
    #             cur_w = cur_w / sum_w
    #             self.feat = pre_w * self.feat + cur_w * feat
    #         else:
    #             self.feat = self.alpha * self.feat + (1 - self.alpha) * feat
    #     self.feat /= np.linalg.norm(self.feat)
        
        
    def camera_update(self, warp_matrix):
        """
        update 'self.mean' of current tracklet with ecc results.
        Parameters
        ----------
        warp_matrix: warp matrix computed by ECC.
        """
        bbox = convert_x_to_bbox(self.kf.x)
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[0][2]
        y2 = bbox[0][3]
        # x1, y1, x2, y2 = convert_x_to_bbox(self.kf.x)[0]
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        # w, h = x2_ - x1_, y2_ - y1_
        # cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = convert_bbox_to_z([x1_, y1_, x2_, y2_])

    def update(self, bbox, id_feature=None, siou = 0, update_feature=None):
        """
        Updates the state vector.
        """
        if bbox is not None:
            # self.last_observation = bbox
            # self.time_since_update = 0

            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            
            # add interface for update feature or not
            if update_feature:
                self.update_features(id_feature, bbox[-1], siou)
            self.confidence = bbox[-1]
        else:
            self.kf.update(bbox)
        self.last_observation = convert_x_to_bbox(self.kf.x_post)[0]
            
    def update_keepv(self, bbox, state, fine_v=0, id_feature=None, siou = 0, update_feature=None):
        """
        Updates the state vector.
        """
        if bbox is not None:
            if state is not None:
                if len(self.kf.history_obs) > 5 and self.kf.history_obs[-1] is not None and self.kf.history_obs[-6] is not None and fine_v:
                    v4 = (self.kf.history_obs[-1][0] - self.kf.history_obs[-6][0])/5
                    v5 = (self.kf.history_obs[-1][1] - self.kf.history_obs[-6][1])/5
                else:
                    v4 = self.kf.x[4]
                    v5 = self.kf.x[5]
                z = convert_bbox_to_z(bbox)
                z_state = convert_bbox_to_z(state)
                gap4 = z[0] - z_state[0]
                gap5 = z[1] - z_state[1]
                r = np.sqrt((v4 ** 2 + v5 ** 2) / (gap4 ** 2 + gap5 ** 2))
                z[0] = z_state[0] + gap4 * r
                z[1] = z_state[1] + gap5 * r
                # print("keep_z",convert_x_to_bbox(z))
            else:
                z = convert_bbox_to_z(bbox)
            self.kf.update(z)
            if update_feature:
                self.update_features(id_feature, bbox[-1], siou)
            self.confidence = bbox[-1] 
            self.hits += 1
        else:
            self.kf.update(bbox)
        # self.time_since_update = 0
        self.hit_streak += 1
        self.last_observation = convert_x_to_bbox(self.kf.x_post)[0]

    def predict_n(self, n):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        pre_n = self.kf.x.copy()
        pre_n[0] += n * pre_n[4]
        pre_n[1] += n * pre_n[5]
        
        return convert_x_to_bbox(pre_n)
    
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        return convert_x_to_bbox(self.kf.x)

    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""


class RBO_Track(object):
    def __init__(self, args, init_thresh, determined_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3,emb_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.emb_threshold = emb_threshold
        self.d_trackers = []
        self.oc_trackers = []
        self.und_trackers = []
        self.frame_count = 0
        self.init_thresh = init_thresh
        self.determined_thresh = determined_thresh
        self.args = args
        KalmanBoxTracker.count = 0

    # ECC for CMC
    def camera_update(self, trackers, warp_matrix):
        for tracker in trackers:
            tracker.camera_update(warp_matrix)
    
    def update(self, output_results, img_info, img_size, id_feature=None, warp_matrix=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        if self.args.ECC:
            # camera update for all stracks
            if warp_matrix is not None:
                self.camera_update(self.d_trackers, warp_matrix)
                self.camera_update(self.oc_trackers, warp_matrix)
                self.camera_update(self.und_trackers, warp_matrix)
        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_bottom = scores > self.args.low_thresh
        inds_init = scores < self.init_thresh
        inds_cant_init = np.logical_and(inds_init, inds_bottom)  # self.det_init > score > 0.1
        dets_cant_init = dets[inds_cant_init]

        inds_high = scores < self.determined_thresh
        inds_ud = scores >= self.init_thresh
        inds_ud_init = np.logical_and(inds_high, inds_ud)  # self.det_determined > score > self.det_init
        dets_ud_init = dets[inds_ud_init]

        dets_d_init = dets[scores >= self.determined_thresh]
        id_d_feature = id_feature[scores >= self.determined_thresh]
        id_ud_feature = id_feature[inds_ud_init]
        id_cant_feature = id_feature[inds_cant_init]

        to_del1 = []
        d_trks = np.zeros((len(self.d_trackers), 5))
        d_pre = np.zeros((len(self.d_trackers), 4))
        for t, trk in enumerate(d_trks):
            pos = self.d_trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], self.d_trackers[t].confidence]
            prepos = self.d_trackers[t].predict_n(3)
            d_pre[t] = [prepos[0][0], prepos[0][1], prepos[0][2], prepos[0][3]]
            if np.any(np.isnan(pos)):
                to_del1.append(t)
        d_trks = np.ma.compress_rows(np.ma.masked_invalid(d_trks))
        for t in reversed(to_del1):
            self.d_trackers.pop(t)
        d_pre = np.delete(d_pre, to_del1, axis=0)

        to_del2 = []
        oc_time = np.zeros(len(self.oc_trackers))                           
        oc_trks = np.zeros((len(self.oc_trackers), 5))
        oc_pre = np.zeros((len(self.oc_trackers), 4))
        for t, trk in enumerate(oc_trks):
            oc_time[t] = self.oc_trackers[t].oc_time
            pos = self.oc_trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], self.oc_trackers[t].confidence]
            prepos = self.oc_trackers[t].predict_n(3)
            oc_pre[t] = [prepos[0][0], prepos[0][1], prepos[0][2], prepos[0][3]]
            if np.any(np.isnan(pos)):
                to_del2.append(t)
        oc_trks = np.ma.compress_rows(np.ma.masked_invalid(oc_trks))
        for t in reversed(to_del2):
            self.oc_trackers.pop(t)
        oc_pre = np.delete(oc_pre, to_del2, axis=0)
        oc_time = np.delete(oc_time, to_del2, axis=0)

        to_del3 = []
        und_trks = np.zeros((len(self.und_trackers), 5))
        und_pre = np.zeros((len(self.und_trackers), 4))
        for t, trk in enumerate(und_trks):
            pos = self.und_trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], self.und_trackers[t].confidence]
            prepos = self.und_trackers[t].predict_n(3)
            und_pre[t] = [prepos[0][0], prepos[0][1], prepos[0][2], prepos[0][3]]
            if np.any(np.isnan(pos)):
                to_del3.append(t)
        und_trks = np.ma.compress_rows(np.ma.masked_invalid(und_trks))
        for t in reversed(to_del3):
            self.und_trackers.pop(t)
        und_pre = np.delete(und_pre, to_del3, axis=0)

        surface_dets = dets[inds_ud]
        d_to_oc = []
        oc_to_d = []
        ud_to_d = []
        oc_to_del = []
        ud_to_del = []

        lenth1 = len(d_trks)
        lenth2 = len(oc_trks)
        lenthd = len(dets_d_init)
        lenthud = len(dets_ud_init)
        dets_init = np.concatenate([x if x.size else np.empty((0, 5)) for x in (dets_d_init, dets_ud_init)], axis=0)
        all_dets = np.concatenate([x if x.size else np.empty((0, 5)) for x in (dets_d_init, dets_ud_init, dets_cant_init)], axis=0)
        dets_feature = np.concatenate([x if x.size else np.empty((0, 2048)) for x in (id_d_feature, id_ud_feature, id_cant_feature)], axis=0)
        
        
        all_iou = iou_batch(all_dets, dets_init)
        all_dets_iou = iou_batch_box1(all_dets, dets_init)
        np.fill_diagonal(all_iou, 0)
        np.fill_diagonal(all_dets_iou, 0)
        
        
        d_max_iou = np.zeros(lenthd+lenthud)
        for n in range(lenthd+lenthud):
            siou = all_dets_iou[n]
            idx = np.where(siou.squeeze() > 0.5)[0]
            maxiou = 0
            for ind in idx:
                if siou[ind] > maxiou and all_dets[ind][3] > all_dets[n][3]:
                    maxiou = siou[ind]
            d_max_iou[n] = maxiou

        all_trks = np.concatenate([x if x.size else np.empty((0, 5)) for x in (d_trks, oc_trks, und_trks)], axis=0)
        all_trks_oc = np.zeros(len(all_trks))
        all_trks_oc[lenth1 : lenth1 + lenth2] = (oc_time > 1).astype(int)
        all_pret = np.concatenate([x if x.size else np.empty((0, 4)) for x in (d_pre, oc_pre, und_pre)], axis=0)

        d_track_features = np.asarray([track.feat for track in self.d_trackers], dtype=float)
        oc_track_features = np.asarray([track.feat for track in self.oc_trackers], dtype=float)
        und_track_features = np.asarray([track.feat for track in self.und_trackers], dtype=float)
        all_features = np.concatenate([x if x.size else np.empty((0, 2048)) for x in (d_track_features, oc_track_features, und_track_features)], axis=0)
        
        pre_iou = iou_batch_box1(all_pret, all_pret)
        np.fill_diagonal(pre_iou, 0)
        # oced_set = set()
        to_oced_set = np.zeros(len(all_trks))
        row, col = np.where(pre_iou > 0.7)
        for r, c in zip(row, col):
            if all_pret[r][3] < all_pret[c][3]:
                to_oced_set[r] = 0.5

        """
            First round of association with dets_init and trks, use features
        """
        emb_dists = embedding_distance(dets_feature, all_features)
        
        matched, unmatched_inds, unmatched_inds_all_trks = associate_detections_to_trackers_reid(all_dets, all_trks, emb_dists, all_trks_oc + to_oced_set, self.iou_threshold, self.emb_threshold)
        """
            Second round of associaton with unmatched_d_trks and dets_ud_init
        """
        if len(unmatched_inds) > 0 and len(unmatched_inds_all_trks) > 0:
            unmatched_mid_dets = all_dets[unmatched_inds] if len(unmatched_inds) > 0 else np.empty((0, all_dets.shape[1]), dtype=dets_init.dtype)
            unmatched_all_trks = all_trks[unmatched_inds_all_trks] if len(unmatched_inds_all_trks) > 0 else np.empty((0, all_trks.shape[1]), dtype=all_trks.dtype)
            unmatched_all_trks_oc = all_trks_oc[unmatched_inds_all_trks]
            unmatched_to_oced_set = to_oced_set[unmatched_inds_all_trks]
            unmatched_emb_dists = emb_dists[np.ix_(unmatched_inds, unmatched_inds_all_trks)]
            
            rematched, last_inds, last_inds_trks = associate_detections_to_trackers(unmatched_mid_dets, unmatched_all_trks, unmatched_emb_dists, unmatched_all_trks_oc+unmatched_to_oced_set, self.iou_threshold-0.2)
            second_matches = []
            for m in rematched:
                real0 = unmatched_inds[m[0]]
                real1 = unmatched_inds_all_trks[m[1]]
                if real0 < lenthd and emb_dists[real0, real1] < 0.8:
                    last_inds_trks = np.concatenate([last_inds_trks, [m[1]]])
                    continue
                second_matches.append([real0, real1])

            for i in range(len(last_inds_trks)):
                last_inds_trks[i] = unmatched_inds_all_trks[last_inds_trks[i]]
            for i in range(len(last_inds)):
                last_inds[i] = unmatched_inds[last_inds[i]]
            if len(second_matches) > 0:
                second_matches = np.array(second_matches, dtype=int)
                matched = np.vstack([matched, second_matches]) if matched.size else second_matches
        else:
            last_inds = unmatched_inds
            last_inds_trks = unmatched_inds_all_trks

        for m in matched:
            if m[0] < lenthd:
                if m[1] < lenth1:
                    r = (all_dets[m[0]][3] - all_dets[m[0]][1])/(all_trks[m[1]][3] - all_trks[m[1]][1])
                    aiou = all_iou[m[0]]
                    siou = all_dets_iou[m[0]]
                    idx = np.where(siou.squeeze() > 0.65)[0]
                    if to_oced_set[m[1]] > 0 and (r < 0.9 or r > 1.1) and self.d_trackers[m[1]].hits > 2:
                        s_max = 1
                        ind_max = m[0]
                        for ind in idx:
                            s_2 = (all_dets[ind][2] - all_dets[ind][0]) * (all_dets[ind][3] - all_dets[ind][1])
                            if s_2 > s_max and all_dets[ind][3] > all_dets[m[0]][3]:
                                s_max = s_2
                                ind_max = ind
                        fdet = fine_det(self.d_trackers[m[1]].get_state()[0], dets_init[m[0], :], all_dets[ind_max, :])
                        self.d_trackers[m[1]].update_keepv(fdet, self.d_trackers[m[1]].last_observation, fine_v = 1, id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    else:
                        self.d_trackers[m[1]].update(dets_init[m[0], :], id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    # self.d_trackers[m[1]].update(dets_init[m[0], :], id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    self.d_trackers[m[1]].accident = 0
                    self.d_trackers[m[1]].oc_time = 0
                elif m[1] < lenth1 + lenth2:
                    realind = m[1] - lenth1
                    self.oc_trackers[realind].update(dets_init[m[0], :], id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    self.oc_trackers[realind].oc_time = 0
                    self.oc_trackers[realind].accident = 0
                    oc_to_d.append(realind)
                else:
                    realind = m[1] - lenth1 - lenth2
                    self.und_trackers[realind].update(dets_init[m[0], :], id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    ud_to_d.append(realind)
                    self.und_trackers[realind].oc_time = 0
            elif m[0] < lenthd + lenthud:
                if m[1] < lenth1:
                    r = (all_dets[m[0]][3] - all_dets[m[0]][1])/(all_trks[m[1]][3] - all_trks[m[1]][1])
                    siou = all_dets_iou[m[0]]
                    idx = np.where(siou.squeeze() > 0.65)[0]
                    if to_oced_set[m[1]] > 0 and (r < 0.9 or r > 1.1) and self.d_trackers[m[1]].hits > 2:
                        s_max = 1
                        ind_max = m[0]
                        for ind in idx:
                            aiou = all_iou[m[0]]
                            s_2 = (all_dets[ind][2] - all_dets[ind][0]) * (all_dets[ind][3] - all_dets[ind][1])
                            if s_2 > s_max and all_dets[ind][3] > all_dets[m[0]][3]:
                                s_max = s_2
                                ind_max = ind
                        fdet = fine_det(self.d_trackers[m[1]].get_state()[0], dets_init[m[0], :], all_dets[ind_max, :])
                        self.d_trackers[m[1]].update_keepv(fdet, self.d_trackers[m[1]].last_observation, fine_v = 1)
                    else:
                        self.d_trackers[m[1]].update(dets_init[m[0], :])
                    # self.d_trackers[m[1]].update(dets_init[m[0], :])
                    self.d_trackers[m[1]].accident = 0
                    self.d_trackers[m[1]].oc_time = 0
                elif m[1] < lenth1 + lenth2:
                    realind = m[1] - lenth1
                    self.oc_trackers[realind].update(dets_init[m[0], :])
                    self.oc_trackers[realind].oc_time = 0
                    self.oc_trackers[realind].accident = 0
                    oc_to_d.append(realind)
                else:
                    realind = m[1] - lenth1 - lenth2
                    self.und_trackers[realind].update(dets_init[m[0], :], id_feature = dets_feature[m[0], :], siou = d_max_iou[m[0]], update_feature=True)
                    self.und_trackers[realind].oc_time = 0
            else:
                real_det_ind = m[0] - lenthd - lenthud
                if m[1] < lenth1:
                    realind = m[1]
                    r = (all_dets[m[0]][3] - all_dets[m[0]][1])/(all_trks[m[1]][3] - all_trks[m[1]][1])
                    aiou = all_iou[m[0]]
                    siou = all_dets_iou[m[0]]
                    idx = np.where(siou.squeeze() > 0.65)[0]
                    if to_oced_set[m[1]] == 0 and (r < 0.9 or r > 1.1) and self.d_trackers[realind].hits > 2:
                        s_max = 1
                        ind_max = m[0]
                        for ind in idx:
                            s_2 = (all_dets[ind][2] - all_dets[ind][0]) * (all_dets[ind][3] - all_dets[ind][1])
                            if s_2 > s_max and all_dets[ind][3] > all_dets[m[0]][3]:
                                s_max = s_2
                                ind_max = ind
                        fdet = fine_det(self.d_trackers[realind].get_state()[0], dets_cant_init[real_det_ind, :], all_dets[ind_max, :])
                        self.d_trackers[realind].update_keepv(fdet, self.d_trackers[realind].last_observation, fine_v = 1)
                    else:
                        self.d_trackers[realind].update(dets_cant_init[real_det_ind, :])
                    # self.d_trackers[realind].update(dets_cant_init[real_det_ind, :])
                    self.d_trackers[realind].accident = 0
                    self.d_trackers[m[1]].oc_time = 0
                elif m[1] < lenth1 + lenth2:
                    realind = m[1] - lenth1
                    aiou = all_iou[m[0]]
                    siou = all_dets_iou[m[0]]
                    idx = np.where(siou.squeeze() > 0.65)[0]
                    s_max = 1
                    ind_max = -1
                    for ind in idx:
                        s_2 = (all_dets[ind][2] - all_dets[ind][0]) * (all_dets[ind][3] - all_dets[ind][1])
                        if s_2 > s_max and all_dets[ind][3] > dets_cant_init[real_det_ind, :][3]:
                            s_max = s_2
                            ind_max = ind
                    if ind_max > -1  and self.oc_trackers[realind].hits > 2:
                        fdet = fine_det(self.oc_trackers[realind].get_state()[0], dets_cant_init[real_det_ind, :], all_dets[ind_max, :])
                        self.oc_trackers[realind].update_keepv(fdet, self.oc_trackers[realind].last_observation)
                    else:
                        self.oc_trackers[realind].update(dets_cant_init[real_det_ind, :])
                    # self.oc_trackers[realind].update(dets_cant_init[real_det_ind, :])
                    self.oc_trackers[realind].accident = 0
                    self.oc_trackers[realind].oc_time = 0
                else:
                    realind = m[1] - lenth1 - lenth2
                    self.und_trackers[realind].update(dets_cant_init[real_det_ind, :])
                    self.und_trackers[realind].hits -= 1
                    self.und_trackers[realind].oc_time = 0


        if len(last_inds) > 0:
            for i in last_inds:
                if all_dets[i, :][4] > self.determined_thresh - 0.1:
                    if i>=lenthd and i<lenthd+lenthud:
                        trk = KalmanBoxTracker(dets_init[i, :], dets_feature[i, :], uncertainty=10000, args=self.args)
                        self.und_trackers.append(trk)
                    elif i<lenthd:
                        trk = KalmanBoxTracker(dets_init[i, :], dets_feature[i, :], uncertainty=1000,args=self.args)
                        self.d_trackers.append(trk)

        for u in last_inds_trks:
            cond1 = all_trks[u][3] <= (all_trks[u][3] - all_trks[u][1]) / 4
            cond2 = all_trks[u][2] <= (all_trks[u][2] - all_trks[u][0]) / 4
            cond3 = abs(float(img_h) - all_trks[u][1]) <= (all_trks[u][3] - all_trks[u][1]) / 4
            cond4 = abs(float(img_w) - all_trks[u][0]) <= (all_trks[u][2] - all_trks[u][0]) / 4
            if len(dets_init) > 0:
                iou = iou_batch_box1([all_trks[u]], dets_init)
                surface_inds = np.argmax(iou)
                surface_det = dets_init[surface_inds, :]
                surface_iou = iou[0, surface_inds]
            else:
                surface_iou = 0
                surface_det = [0, 0, 0, 0, 0]
            realind = u
            if realind < lenth1:
                if cond1 or cond2 or cond3 or cond4:
                    self.d_trackers[realind].oc_time = 90
                    d_to_oc.append(realind)
                elif surface_iou > 0.1:
                    self.d_trackers[realind].oc_time += 1
                    self.d_trackers[realind].update(None)
                    d_to_oc.append(realind)
                else:
                    self.d_trackers[realind].oc_time += 1
                    self.d_trackers[realind].accident += 1
                    self.d_trackers[realind].update(None)
                    if self.d_trackers[realind].accident > 1:
                        d_to_oc.append(realind)
            elif realind < lenth1 + lenth2:
                realind -= lenth1
                if cond1 or cond2 or cond3 or cond4:
                    self.oc_trackers[realind].oc_time = 90
                else:
                    self.oc_trackers[realind].update(None)
                    self.oc_trackers[realind].oc_time += 1
                if self.oc_trackers[realind].oc_time >= 90:
                    oc_to_del.append(realind)
            else:
                realind -= lenth1 + lenth2
                if cond1 or cond2 or cond3 or cond4:
                    self.und_trackers[realind].oc_time = 40
                else:
                    self.und_trackers[realind].update(None)
                    self.und_trackers[realind].oc_time += 1
                if self.und_trackers[realind].oc_time >= 35:
                    ud_to_del.append(realind)

        # 在三种轨迹都更新完成之后，进行轨迹间的转换
        '''
        d_to_oc = []
        oc_to_d = []
        ud_to_d = []
        oc_to_del = []
        ud_to_del = []
        '''
        d_to_oc.sort(reverse=True)
        for n in d_to_oc:
            self.oc_trackers.append(self.d_trackers[n])
            self.d_trackers.pop(n)
        for n in oc_to_d:
            self.d_trackers.append(self.oc_trackers[n])
        for n in ud_to_d:
            self.d_trackers.append(self.und_trackers[n])
        oc_to_del += oc_to_d
        ud_to_del += ud_to_d
        oc_to_del.sort(reverse=True)
        for n in oc_to_del:
            self.oc_trackers.pop(n)
        ud_to_del.sort(reverse=True)
        for n in ud_to_del:
            self.und_trackers.pop(n)
            
        ret = []
        if len(self.d_trackers) > 0:
            for trk in self.d_trackers:
                if trk.oc_time < 1:
                    # +1 as MOT benchmark requires positive
                    d = trk.get_state()[0][:4]
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if len(self.oc_trackers) > 0:
            for trk in self.oc_trackers:
                if trk.oc_time < 1:
                    # +1 as MOT benchmark requires positive
                    d = trk.get_state()[0][:4]
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if len(self.und_trackers) > 0:
            for trk in self.und_trackers:
                if (trk.hits >= self.min_hits or self.frame_count <= self.min_hits) and trk.oc_time < 1:
                    # +1 as MOT benchmark requires positive
                    d = trk.get_state()[0][:4]
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
