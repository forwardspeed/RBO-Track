import os
import numpy as np
from sklearn.cluster import KMeans

def intersection_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersections = w * h
    return intersections

def box_area(bbox):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return area

def iou_batch_box1(bboxes1, bboxes2):

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]))                                          
    return(o)

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def h_batch(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = bboxes1[..., 1]
    yy12 = bboxes1[..., 3]

    yy21 = bboxes2[..., 1]
    yy22 = bboxes2[..., 3]
    o1 = (yy12 - yy11) / (yy22 - yy21)
    o2 = (yy22 - yy21) / (yy12 - yy11)
    o = np.minimum(o1, o2)
    return (o)

def cal_score_dif_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    score2 = bboxes2[..., 4]
    score1 = bboxes1[..., 4]

    return (abs(score2 - score1))

def cal_score_dif_batch_two_score(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    score2 = bboxes2[..., 5]
    score1 = bboxes1[..., 4]

    return (abs(score2 - score1))

def fhiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o1 = (yy12 - yy11) / (yy22 - yy21)
    o2 = (yy22 - yy21) / (yy12 - yy11)
    o = np.minimum(o1, o2)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def siou_batch(bboxes1, bboxes2):

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    o1 = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]))/((bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]))
    o2 = ((bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]))/((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]))
    o = np.minimum(o1, o2)
    return (o)


def hmiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou

def giou_batch_true(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou

def ct_dist(bboxes1, bboxes2):

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    return ct_dist

def ct_r(bboxes1, bboxes2):

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    w1 = (bboxes1[..., 2] - bboxes1[..., 0])
    w2 = (bboxes2[..., 2] - bboxes2[..., 0])
    
    ct_r = w1 + w2
    # ct_r = np.repeat(w1[:, np.newaxis], len(bboxes2), axis=1)*2
    # ct_r *= trks_in_oc
    return ct_r

def score_batch(bboxes1, bboxes2):
    conf1 = bboxes1[:, 4]
    c = np.repeat(conf1[:, np.newaxis], len(bboxes2), axis=1)
    return c

def v_batch(bboxes1, bboxes2, pre_v):

    # 1. 中心点
    c1 = (bboxes1[:, :2] + bboxes1[:, 2:4]) / 2.0          # (N, 2)
    c2 = (bboxes2[:, :2] + bboxes2[:, 2:4]) / 2.0          # (M, 2)

    # 2. 位移方向 (N, M, 2)
    delta = c1[:, None, :] - c2[None, :, :]                # (N, M, 2)
    # 3. 轨迹速度方向 (M, 2) -> (1, M, 2)
    v = pre_v[None, :, :]                                   # (1, M, 2)

    # 4. 单位化
    delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8
    delta_u = delta / delta_norm
    delta_mask = delta_norm < 10
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    v_mask = (v_norm < 10).squeeze(-1)
    v_u = v / v_norm
    # v_u[v_mask] = 0

    # 5. 方向余弦
    cos_theta = np.sum(delta_u * v_u, axis=-1)             # (N, M)
    # speed_mask = delta_norm.squeeze(-1) < v_norm.squeeze(-1)  # (N, M)
    speed_mask = (delta_norm < 10).squeeze(-1)
    cos_theta[speed_mask] = 0.0
    # sign = cos_theta > 0
    # sign = 2 * sign - 1
    # cos_theta = cos_theta * cos_theta * sign
    return 1 - cos_theta

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def set_threshold(dists, ori_threshold, min_anchor, max_anchor):
    # # More Sampling with linear assignment (Do not use)
    # indices = linear_assignment(dists)
    # dists = dists[indices[0], indices[1]]

    # Prepare
    threshold = ori_threshold
    dists_1d = dists.reshape(-1, 1)
    dists_1d = dists_1d[dists_1d < max_anchor]
    dists_1d = dists_1d[min_anchor < dists_1d]

    if len(dists_1d) > 0:
        # Prepare
        dists_1d = list(dists_1d) + [min_anchor, max_anchor]
        dists_1d = np.array(dists_1d).reshape(-1, 1)

        # Select Clustering
        model = KMeans(n_clusters=2, init=np.array([[min_anchor], [max_anchor]]), n_init=1, random_state=10000)

        # Fit
        result = model.fit_predict(dists_1d)

        # Rare exception (Only occurs with Gaussian mixture clustering)
        if np.sum(result == 0) == 0 or np.sum(result == 1) == 0:
            return ori_threshold

        # Set threshold
        threshold = min(np.max(dists_1d[result == 0]), np.max(dists_1d[result == 1])) + 1e-5

    return threshold
    
def associate_detections_to_trackers(detections, trackers, emb_cost, all_trks_oc,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers), dtype=int))
    all_trks_oc = all_trks_oc.reshape(1, -1)
    score_matrix = score_batch(detections, trackers)
    emb_cost = np.where(score_matrix < 0.5, emb_cost*2, emb_cost)
    
    iou_matrix = 1 - hmiou(detections, trackers)
    siou_matrix = siou_batch(detections, trackers)
    
    trks_oc = ((all_trks_oc < 1.5) & (all_trks_oc > 0.5)).astype(np.int32)
    trks_in_oc = (all_trks_oc > 0.5).astype(np.int32)
    
    ct_matrix = ct_dist(detections, trackers)
    r = ct_r(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        fix_emb_cost = emb_cost - 0.2 * trks_oc
        fix_iou_matrix = iou_matrix + 0.2 * trks_in_oc
        a1 = (iou_matrix <= 0.5) & (siou_matrix > 0.7)
        a6 = (ct_matrix < 4*r) & (siou_matrix * trks_in_oc > 0.7) & (fix_emb_cost < 0.2)
        a = (a1 | a6).astype(np.int32)
        
        # a = (iou_matrix <= 0.5) & (siou_matrix > 0.7)
        cost = fix_iou_matrix - a
        # cost = iou_matrix - a
        
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices = linear_assignment(cost)
            matched_indices = iterative_assignment(cost)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(a[m[0], m[1]] == 0):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_trackers, dtype=np.int32)

def associate_detections_to_trackers_reid(detections, trackers, emb_cost, all_trks_oc, iou_threshold = 0.3, emb_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers), dtype=int))
    score_matrix = score_batch(detections, trackers)
    iou_matrix = 1 - hmiou(detections, trackers)*score_matrix
    
    emb_cost = np.where(score_matrix < 0.5, emb_cost*2, emb_cost)
    
    all_trks_oc = all_trks_oc.reshape(1, -1)
    trks_oc = ((all_trks_oc < 1.5) & (all_trks_oc > 0.5)).astype(np.int32)
    trks_in_oc = (all_trks_oc > 0.5).astype(np.int32)
    
    ct_matrix = ct_dist(detections, trackers)
    r = ct_r(detections, trackers)
    siou_matrix = siou_batch(detections, trackers)
    
    fix_iou_matrix = iou_matrix - 0.2 * trks_oc
    mix_cost = iou_matrix/2 + (1 - score_matrix) * emb_cost*4 - siou_matrix/2 - score_matrix/2


    a_ori = (iou_matrix < 0.9) & (score_matrix > 0.65)
    a1 = (fix_iou_matrix < 0.6) & (emb_cost < 0.8)
    a2 = (ct_matrix < 2*r) & (siou_matrix * trks_in_oc > 0.7) & (emb_cost < 0.55)
    a3 = (ct_matrix < 3*r) & (siou_matrix * trks_in_oc > 0.7) & (emb_cost < 0.35)
    a4 = (ct_matrix < 4*r) & (siou_matrix * trks_in_oc > 0.7) & (emb_cost < 0.3)
    a5 = (ct_matrix < 5*r) & (siou_matrix * trks_in_oc > 0.7) & (emb_cost < 0.2)
    a = (a_ori | a4 | a5).astype(np.int32)
    cost = mix_cost - a
    if min(iou_matrix.shape) > 0:
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices = linear_assignment(cost)
            matched_indices = iterative_assignment(cost)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(a[m[0], m[1]] == 0):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_trackers, dtype=np.int32)

def associate_detections_to_trackers_ori(detections, trackers, iou_type, fuse, iou_threshold = 0.3, scend_det = 0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers), dtype=int))
    if iou_type == 0:
        iou_matrix = 1 - iou_batch(detections, trackers)
    else:
        iou_matrix = 1 - fhiou(detections, trackers)
    if fuse == 1:
        score_matrix = score_batch(detections, trackers)
        iou_matrix = 1 -((1-iou_matrix) * score_matrix)
    if scend_det > 0:
        iou_matrix[scend_det:len(detections)] *= 2
            
    # max_iou = iou_matrix.max()
    # min_iou = iou_matrix.min()
    # iou_threshold = set_threshold(iou_matrix, iou_threshold, min_iou, max_iou)
    a = (iou_matrix < iou_threshold)
    if min(iou_matrix.shape) > 0:
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices = linear_assignment(iou_matrix)
            matched_indices = iterative_assignment(iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(a[m[0], m[1]] == 0):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_trackers, dtype=np.int32)

def iterative_assignment(cost):
    # Initialization
    matches = []

    # Match
    while True:
        # Match tracks with detections
        matches_ = associate(cost)
        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break

        # Append
        matches += matches_

        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 10.
            cost[:, d] = 10.

    return np.array(matches)

def associate(cost):
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < 10:
                matches.append([tdx, ddx])

    return matches

from scipy.spatial.distance import cdist
def embedding_distance(tracks_feat, detections_feat, metric='cosine'):
    """
    :param tracks: list[KalmanBoxTracker]
    :param detections: list[KalmanBoxTracker]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks_feat), len(detections_feat)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    # det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)    # [detection_num, emd_dim]
    # #for i, track in enumerate(tracks):
    #     #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)    # [track_num, emd_dim]
    cost_matrix = np.maximum(0.0, cdist(tracks_feat, detections_feat, metric))  # Nomalized features, metric: cosine, [track_num, detection_num]
    return cost_matrix

    