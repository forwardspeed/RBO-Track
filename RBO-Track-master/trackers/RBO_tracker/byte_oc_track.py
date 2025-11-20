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

    def __init__(self, bbox, feat=None, uncertainty=1000., orig=False, args=None):
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
        self.time_since_update = 0
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
        self.start_frame = 0
        self.last_frame = 0
        self.lost_time = 0
        self.tracked = 0
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
                score *= (1-siou)
                self.feat = ((2) / (2 + score)) * self.feat + (score/(2 + score)) * feat
            else:
                self.feat = ((5) / (5 + score)) * self.feat + (score/(5 + score)) * feat
        else:
            self.feat = self.alpha * self.feat + (1 - self.alpha) * feat
        self.feat /= np.linalg.norm(self.feat)

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
            self.time_since_update = 0

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
        else:
            self.kf.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_observation = convert_x_to_bbox(self.kf.x_post)[0]

    def predict_n(self, n):
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
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append()
        return convert_x_to_bbox(self.kf.x)

    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Byte_oc_Track(object):
    def __init__(self, args, init_thresh, determined_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3,emb_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracked_trackers = []
        self.lost_trackers = []
        self.removed_trackers = []
        self.frame_count = 0
        
        self.max_time_lost = args.track_buffer
        self.init_thresh = init_thresh
        self.args = args
        KalmanBoxTracker.count = 0

    # ECC for CMC
    def camera_update(self, trackers, warp_matrix):
        for tracker in trackers:
            tracker.camera_update(warp_matrix)
    
    def update_b(self, output_results, img_info, img_size, id_feature=None, warp_matrix=None):
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
                self.camera_update(self.tracked_trackers, warp_matrix)
                self.camera_update(self.lost_trackers, warp_matrix)
                self.camera_update(self.removed_trackers, warp_matrix)
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
        dets_all = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        
        remain_inds = scores > self.init_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.init_thresh

        
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets_all[inds_second]
        dets = dets_all[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        dets_all = np.concatenate((dets, dets_second), axis=0)
        # id_d_feature = id_feature[scores >= self.determined_thresh]
        # id_ud_feature = id_feature[inds_ud_init]

        to_del1 = []
        tracked_trks = np.zeros((len(self.tracked_trackers), 5))
        tracked_pre = np.zeros((len(self.tracked_trackers), 4))
        for t, trk in enumerate(tracked_trks):
            pos = self.tracked_trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], self.tracked_trackers[t].confidence]
            prepos = self.tracked_trackers[t].predict_n(3)
            tracked_pre[t] = [prepos[0][0], prepos[0][1], prepos[0][2], prepos[0][3]]
            if np.any(np.isnan(pos)):
                to_del1.append(t)
        tracked_trks = np.ma.compress_rows(np.ma.masked_invalid(tracked_trks))
        for t in reversed(to_del1):
            self.tracked_trackers.pop(t)
        # tracked_pre = np.delete(tracked_pre, to_del1, axis=0)

        to_del2 = []                          
        lost_trks = np.zeros((len(self.lost_trackers), 5))
        lost_pre = np.zeros((len(self.lost_trackers), 4))
        for t, trk in enumerate(lost_trks):
            pos = self.lost_trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], self.lost_trackers[t].confidence]
            prepos = self.lost_trackers[t].predict_n(3)
            lost_pre[t] = [prepos[0][0], prepos[0][1], prepos[0][2], prepos[0][3]]
            if np.any(np.isnan(pos)):
                to_del2.append(t)
        lost_trks = np.ma.compress_rows(np.ma.masked_invalid(lost_trks))
        for t in reversed(to_del2):
            self.lost_trackers.pop(t)
        lost_pre = np.delete(lost_pre, to_del2, axis=0)


        all_dets_ioa = iou_batch_box1(dets_all, dets_all)
        np.fill_diagonal(all_dets_ioa, 0)    
        
        
        all_pret = np.concatenate([x if x.size else np.empty((0, 4)) for x in (tracked_pre, lost_pre)], axis=0)
        pre_iou = iou_batch_box1(all_pret, all_pret)
        np.fill_diagonal(pre_iou, 0)
        to_oced_set = np.zeros(len(all_pret))
        row, col = np.where(pre_iou > 0.7)
        for r, c in zip(row, col):
            if all_pret[r][3] < all_pret[c][3]:
                to_oced_set[r] = 0.5
        
        
        tracked_to_lost = []
        lost_to_tracked = []
        tracked_to_del = []
        lost_to_del = []
        
        unconfirmed = []
        tracked_inds = []
        for i, t in enumerate(self.tracked_trackers):
            if t.tracked == 1:
                tracked_inds.append(i)
            else:
                unconfirmed.append(i)
        len_d = len(dets)
        len_f = len(tracked_inds)
        first_tracks = np.concatenate((tracked_trks[tracked_inds], lost_trks), axis=0)
        matched, un_dets_inds, un_trks_inds = associate_detections_to_trackers_ori(dets, first_tracks, 1, 1, iou_threshold = 0.9)
        
        for m in matched:
            if m[1]<len_f:
                t_ind = tracked_inds[m[1]]
                # r = (dets[m[0]][3] - dets[m[0]][1])/(first_tracks[m[1]][3] - first_tracks[m[1]][1])
                # if dets[m[0]][4]>0.8 and to_oced_set[t_ind] > 0 and (r < 0.9 or r > 1.1) and self.tracked_trackers[t_ind].tracked == 1:
                #     # a_iou = all_iou[m[0]]
                #     a_ioa = all_dets_ioa[m[0]]
                #     idx = np.where(a_ioa.squeeze() > 0.7)[0]
                #     s_max = 1
                #     ind_max = m[0]
                #     for ind in idx:
                #         s = (dets_all[ind][2] - dets_all[ind][0]) * (dets_all[ind][3] - dets_all[ind][1])
                #         if s > s_max and dets_all[ind][3] > dets[m[0]][3]:
                #             s_max = s
                #             ind_max = ind
                #     fdet = fine_det(self.tracked_trackers[t_ind].get_state()[0], dets[m[0], :], dets_all[ind_max, :])
                #     self.tracked_trackers[t_ind].update_keepv(fdet, self.tracked_trackers[t_ind].last_observation, fine_v = 1)
                # else:
                #     self.tracked_trackers[t_ind].update(dets[m[0], :])
                self.tracked_trackers[t_ind].update(dets[m[0], :])
                self.tracked_trackers[t_ind].tracked = 1
                self.tracked_trackers[t_ind].last_frame = self.frame_count
                self.tracked_trackers[t_ind].lost_time = 0
            else:
                realt = m[1]-len_f
                self.lost_trackers[realt].update(dets[m[0], :])
                self.lost_trackers[realt].tracked = 1
                self.lost_trackers[realt].last_frame = self.frame_count
                self.lost_trackers[realt].lost_time = 0
                lost_to_tracked.append(realt)
                
        re_tracks = first_tracks[un_trks_inds]
        rematched, un_dets2_inds, un_trks2_inds = associate_detections_to_trackers_ori(dets_second, re_tracks, 1, 0, iou_threshold = 0.5)
        
        for m in rematched:
            real1 = un_trks_inds[m[1]]
            if real1<len_f:
                t_ind = tracked_inds[real1]
                self.tracked_trackers[t_ind].update(dets_second[m[0], :])
                self.tracked_trackers[t_ind].tracked = 1
                self.tracked_trackers[t_ind].last_frame = self.frame_count
                self.tracked_trackers[t_ind].lost_time = 0
            else:
                realt = real1-len_f
                self.lost_trackers[realt].update(dets_second[m[0], :])
                self.lost_trackers[realt].tracked = 1
                self.lost_trackers[realt].last_frame = self.frame_count
                self.lost_trackers[realt].lost_time = 0
                lost_to_tracked.append(realt)
                
        for t in un_trks2_inds:
            real1 = un_trks_inds[t]
            # real1 = t
            if real1<len_f:
                t_ind = tracked_inds[real1]
                self.tracked_trackers[t_ind].update(None)
                self.tracked_trackers[t_ind].tracked = 0
                self.tracked_trackers[t_ind].lost_time += 1
                tracked_to_lost.append(t_ind)
            else:
                realt = real1-len_f
                self.lost_trackers[realt].update(None)
                self.lost_trackers[realt].lost_time += 1
        
        uc_dets = dets[un_dets_inds]
        uc_tracks = tracked_trks[unconfirmed]
        ucmatched, un_ucdets_inds, un_uctracks_inds = associate_detections_to_trackers_ori(uc_dets, uc_tracks, 1, 1, 0.7)
        
        for m in ucmatched:
            t_ind = unconfirmed[m[1]]
            self.tracked_trackers[t_ind].update(uc_dets[m[0], :])
            self.tracked_trackers[t_ind].tracked = 1
            self.tracked_trackers[t_ind].last_frame = self.frame_count
            self.tracked_trackers[t_ind].lost_time = 0
        for t in un_uctracks_inds:
            t_ind = unconfirmed[t]
            self.tracked_trackers[t_ind].tracked = -1
            tracked_to_del.append(t_ind)
            
        for d in un_ucdets_inds:
            if uc_dets[d][4]>=self.init_thresh+0.1:
                
                trk = KalmanBoxTracker(uc_dets[d], uncertainty=100,args=self.args)
                trk.start_frame = self.frame_count
                trk.last_frame = self.frame_count
                if self.frame_count==1:
                    trk.tracked = 1
                self.tracked_trackers.append(trk)
        
        for i, t in enumerate(self.lost_trackers):
            if t.lost_time > self.max_time_lost:
                lost_to_del.append(i)
        
        # 在三种轨迹都更新完成之后，进行轨迹间的转换
        '''
        tracked_to_lost = []
        lost_to_tracked = []
        tracked_to_del = []
        lost_to_del = []
        '''
        for n in tracked_to_lost:
            self.lost_trackers.append(self.tracked_trackers[n])
        for n in lost_to_tracked:
            self.tracked_trackers.append(self.lost_trackers[n])
        tracked_to_del += tracked_to_lost
        lost_to_del += lost_to_tracked
        
        tracked_to_del.sort(reverse=True)
        for n in tracked_to_del:
            self.tracked_trackers.pop(n)
        lost_to_del.sort(reverse=True)
        for n in lost_to_del:
            self.lost_trackers.pop(n)

            
            
        tracked_trks = np.zeros((len(self.tracked_trackers), 4))
        for t, trk in enumerate(tracked_trks):
            pos = self.tracked_trackers[t].get_state()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3]]
        
        lost_trks = np.zeros((len(self.lost_trackers), 4))
        for t, trk in enumerate(lost_trks):
            pos = self.lost_trackers[t].get_state()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3]]
        
        final_iou = 1 - iou_batch(tracked_trks, lost_trks)
        pairs = np.where(final_iou<0.15)
        
        rem_t = set()
        rem_l = set()
        for r, c in zip(*pairs):
            if (self.tracked_trackers[r].last_frame - self.tracked_trackers[r].start_frame) > (self.lost_trackers[c].last_frame - self.lost_trackers[c].start_frame):
                rem_l.add(c)
            else:
                rem_t.add(r)

        rem_t = list(rem_t)
        rem_l = list(rem_l)
        rem_t.sort(reverse=True)
        for n in rem_t:
            self.tracked_trackers.pop(n)
        rem_l.sort(reverse=True)
        for n in rem_l:
            self.lost_trackers.pop(n)
        
        ret = []
        if len(self.tracked_trackers) > 0:
            for trk in self.tracked_trackers:
                if trk.tracked == 1:
                    d = trk.get_state()[0][:4]
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))