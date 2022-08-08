import json
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from EvalResult import Node2DEvalResult, DetectionResult, TrackResult


class FeatureTypes(Enum):
    # 对应的是Node结构里的features的key
    # node->{featureType: []}
    Box2D = "box2D"


class Node(object):
    '''
    Define the structure of one node.
    A node could represent either a detection, a tracklet, or a trajectory.
    '''

    def __init__(self, tid, timestamp, features):
        self._tid = tid  # track-id
        self.__timestamps = np.array([], dtype=np.int)
        self.features = features
        # if timestamp is not None and features is not None:
        self.__timestamps = np.append(self.__timestamps, timestamp).astype(np.int)

    @property
    def timestamps(self):
        return self.__timestamps

    @property
    def feature_types(self):
        return self.features.keys()

    @property
    def size(self):
        count = 0
        count += len(self.timestamps)
        return count

    @property
    def isempty(self):
        return len(self.timestamps.size > 0) == 0

    @property
    def tid(self):
        return self._tid

    @property
    def mv_start(self):
        '''
        :return: the earliest time that any camera starts working
        '''
        ret = None
        camera_start = self.start
        if ret is None:
            ret = camera_start
        elif camera_start < ret:
            ret = camera_start
        return ret

    @property
    def mv_end(self):
        '''
        :return: the latest time that any camera ends
        '''
        ret = None
        camera_end = self.end
        if ret is None:
            ret = camera_end
        elif camera_end > ret:
            ret = camera_end
        return ret

    @property
    def start(self):
        if self.size == 0:
            return None
        return self.__timestamps[0]

    @property
    def end(self):
        if self.size == 0:
            return None
        return self.__timestamps[-1]

    def extend(self, other):
        '''
        extend self with the other. should be faster than merge
        '''
        ts = [t for i, t in enumerate(other.timestamps) if t > self.end]
        for key in self.feature_types:
            self.features[key] += [other.features[key][i] for i, t in enumerate(other.timestamps) if t > self.end]
        self.__timestamps = np.append(self.__timestamps, ts).astype(np.int)


def is_bbox_in_another_bbox(bbox, another_bbox):
    xmin, ymin, w, h = bbox
    another_xmin, another_ymin, another_w, another_h = another_bbox
    return xmin >= another_xmin and ymin >= another_ymin and (xmin + w <= another_xmin + another_w) and (
            ymin + h <= another_ymin + another_h)


def slit_pid_by_time_gap(node_dict, split_time_gap):
    new_tid = 0
    nodes = []

    for tid in sorted(node_dict.keys()):
        tnodes = node_dict[tid]
        tnodes = sorted(tnodes, key=lambda x: x.mv_start)
        tnodes_ts = [tnode.mv_start for tnode in tnodes]

        last, slow, fast = 0, 0, 1

        while slow < len(tnodes_ts) and fast < len(tnodes_ts):
            if tnodes_ts[fast] - tnodes_ts[slow] > split_time_gap:
                sub_tnodes = tnodes[last: fast]
                node = sub_tnodes[0]
                node._tid = new_tid
                new_tid += 1
                for i in range(1, len(sub_tnodes)):
                    node.extend(sub_tnodes[i])
                nodes.append(node)
                last = fast

            slow += 1
            fast += 1

        sub_tnodes = tnodes[last: fast]

        if len(sub_tnodes) > 0:
            node = sub_tnodes[0]
            node._tid = new_tid
            for i in range(1, len(sub_tnodes)):
                node.extend(sub_tnodes[i])

            nodes.append(node)
            new_tid += 1

    return nodes


# nodes后处理，很难debug.改成用前处理
def filter_nodes_by_mask(nodes: list, masked_image_file: str, image_size) -> list:
    if not Path(masked_image_file).exists:
        raise FileNotFoundError(masked_image_file)
    nodes_new = []
    masked_image_list = [masked_image_file]
    ignore_masks = [load_mask_from_image_file(str(image_path)) for image_path in masked_image_list]
    for node in nodes:
        timestamps = np.array([], dtype=np.int)
        features = {FeatureTypes.Box2D: []}
        tid = node.tid
        for index, box in enumerate(node.features[FeatureTypes.Box2D]):
            if not check_box_in_multi_ignore_region_or_not(ignore_masks, box, image_size):
                features[FeatureTypes.Box2D].append(box)
                timestamps = np.append(timestamps, node.timestamps[index]).astype(int)
        if timestamps:
            node_new = Node(tid=tid, timestamp=timestamps, features=features)
            nodes_new.append(node_new)
    nodes_new = sorted(nodes_new, key=lambda x: x.mv_start)
    return nodes_new


# nodes前处理
def filter_travis_by_mask(travis_data: dict, masked_image_file: str, image_size: Tuple) -> dict:
    if not Path(masked_image_file).exists:
        raise FileNotFoundError(masked_image_file)
    ret = defaultdict(list)
    masked_image_list = [masked_image_file]
    ignore_masks = [load_mask_from_image_file(str(image_path)) for image_path in masked_image_list]
    for frame, bboxes_info in travis_data.items():
        useful_bbox_info = []
        for bbox_info in bboxes_info:
            if not check_box_in_multi_ignore_region_or_not(ignore_masks, bbox_info[0], image_size):
                useful_bbox_info.append(bbox_info)
        ret.update({frame: useful_bbox_info})
    return ret


class AverageMetric:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, k=1):
        self.sum += value * k
        self.count += k

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class Nodes(object):
    @staticmethod
    def load(filename,
             imshape=None,
             drop_image_border=False,
             image_border_ratio=0.8,
             is_split_pid=False,
             split_time_gap=10,
             mask_file=None):
        # only support json and support standard final.reduced.json style

        if drop_image_border:
            image_width, image_height = imshape
            inner_image_x, inner_image_y = image_width * (1 - image_border_ratio) / 2, image_height * (
                    1 - image_border_ratio) / 2
            inner_image_width, inner_image_height = image_width * image_border_ratio, image_height * image_border_ratio
            innder_bbox = [inner_image_x, inner_image_y, inner_image_width, inner_image_height]

        nodes = []
        node_dict = defaultdict(list)
        with open(filename, 'r') as f:
            data = json.loads(f.read())
        if mask_file:
            data = filter_travis_by_mask(data, mask_file, image_size=imshape)

        frames = sorted([int(frame) for frame in data.keys()])

        avg_det_num = AverageMetric()

        for frame_index in sorted(data.keys(), key=lambda x: int(x)):
            avg_det_num.update(len(data[frame_index]))
            for trackdet in data[frame_index]:
                box, tid = trackdet[:2]
                if drop_image_border and not is_bbox_in_another_bbox(box, innder_bbox):
                    continue
                features = {
                    FeatureTypes.Box2D: [[int(_) for _ in box]]
                }
                node = Node(tid, int(frame_index), features)
                node_dict[tid].append(node)

        logger.info(f"There are avg {avg_det_num.avg} number of detections for {filename}.")

        if is_split_pid:
            nodes = slit_pid_by_time_gap(node_dict, split_time_gap)
        else:
            for tid in sorted(node_dict.keys()):
                tnodes = node_dict[tid]
                tnodes = sorted(tnodes, key=lambda x: x.mv_start)
                # group all
                node = tnodes[0]
                node._tid = tid
                for i in range(1, len(tnodes)):
                    node.extend(tnodes[i])
                nodes.append(node)

        logger.info(f"There are {len(nodes)} number of nodes after grouping by tid.")
        # sort them
        nodes = sorted(nodes, key=lambda x: x.mv_start)
        return nodes, frames


class Eval2DNodes(Nodes):
    def __init__(self, rpath, lpath):
        self.rpath = rpath
        self.lpath = lpath
        self.results = {}

    @property
    def rfiles(self):
        rfiles = list(Path(self.rpath).rglob(f"*.json"))
        if not rfiles:
            raise FileNotFoundError(f"not found any json file in {self.rpath}")
        return rfiles

    @property
    def lfiles(self):
        lfiles = list(Path(self.lpath).rglob(f"*.json"))
        if not lfiles:
            raise FileNotFoundError(f"not found any json file in {self.lpath}")
        return lfiles

    @property
    def rnames(self):
        file_mapping = {}
        for file in self.rfiles:
            file_mapping[file.name.split(".")[0]] = file
        return file_mapping

    @property
    def lnames(self):
        file_mapping = {}
        for file in self.lfiles:
            file_mapping[file.name.split(".")[0]] = file
        return file_mapping

    def filter_files(self, match_all=True):
        lost_file = set()
        match_files = {}
        rnames = self.rnames
        lnames = self.lnames
        for rname, rfile in rnames.items():
            if rname in lnames:
                match_files[rfile] = lnames[rname]
            else:
                lost_file.add(rfile)
                logger.error(f"{rfile} not found the gt file")
        if match_all and lost_file:
            raise FileNotFoundError(f"There are {len(lost_file)} nums files not found gt file, please check")
        return match_files

    def __call__(self, min_track_length=10, min_split_length=10, overlap_threshold=0.4,
                 imshape=(2560, 1440), filter_nodes=True, match_all=True, max_worker=35, drop_image_border=False,
                 image_border_ratio=0.8, is_split_pid=False, split_time_gap=10, mask_path=None,
                 dynamic_ntu_window=False):
        evaluation = partial(self._evaluate,
                             min_track_length=min_track_length,
                             min_split_length=min_split_length,
                             overlap_threshold=overlap_threshold,
                             filter_nodes=filter_nodes,
                             imshape=imshape, drop_image_border=drop_image_border,
                             image_border_ratio=image_border_ratio,
                             is_split_pid=is_split_pid,
                             split_time_gap=split_time_gap,
                             mask_path=mask_path,
                             dynamic_ntu_window=dynamic_ntu_window)
        # 通过算法结果文件找gt文件
        match_files = self.filter_files(match_all=match_all)
        with ProcessPoolExecutor(max_workers=max_worker) as p:
            futures = [p.submit(evaluation, rfile=rfile, lfile=lfile) for rfile, lfile in match_files.items()]
        rets = list(map(lambda ft: ft.result(), futures))
        return rets

    @staticmethod
    def _evaluate(rfile, lfile, min_track_length, min_split_length, overlap_threshold, filter_nodes, imshape,
                  drop_image_border, image_border_ratio, is_split_pid, mask_path, split_time_gap,
                  dynamic_ntu_window=False):
        fname = rfile.name.split(".")[0]
        if mask_path:
            mask_file = Path(mask_path).joinpath(f"{fname}.jpg")
        else:
            mask_file = None
        rnodes, rframes = Eval2DNodes.load(rfile, imshape, drop_image_border=drop_image_border,
                                           is_split_pid=is_split_pid,
                                           image_border_ratio=image_border_ratio, mask_file=mask_file,
                                           split_time_gap=split_time_gap)
        lnodes, lframes = Eval2DNodes.load(lfile, imshape, drop_image_border=drop_image_border,
                                           image_border_ratio=image_border_ratio,
                                           is_split_pid=is_split_pid, mask_file=mask_file,
                                           split_time_gap=split_time_gap)
        # return 默认是None
        if not rfile.exists() or not lfile.exists():
            logger.error(f"{rfile} or {lfile} not found, please check the file path; skip")
            exit(1)
        logger.info(f"ready to eval: {rfile.name} , {lfile.name}")
        det_result = Eval2DNodes._evaluate_precision_recall(fname=fname, rnodes=rnodes, lnodes=lnodes,
                                                            ovth=overlap_threshold,
                                                            min_track_length=min_track_length,
                                                            rframes=rframes, lframes=lframes)

        if dynamic_ntu_window and det_result.fps != 0:
            min_split_length = int(round(min_track_length * (det_result.fps / 12)) + 1)

        trk_result = Eval2DNodes._evaluate_purity(rnodes=rnodes, lnodes=lnodes, fname=fname,
                                                  minov=overlap_threshold,
                                                  imshape=imshape,
                                                  min_track_length=min_track_length,
                                                  min_split_length=min_split_length,
                                                  filter_nodes=filter_nodes)
        n2er = Node2DEvalResult(fname=fname, det_result=det_result, trk_result=trk_result)
        logger.info(n2er)
        return n2er

    @staticmethod
    def _evaluate_precision_recall(fname, rnodes, lnodes, ovth=0.4, min_track_length=0, rframes=(), lframes=()):
        """
        The function name is inaccurate
        This function counts the number of bounding-boxes for (1) true positive (2) false positive, and (3) all GT bboxes
        :param rnodes: right-tracks, or the detected tracks to evaluate
        :param lnodes: left-tracks, or the ground-truth tracks
        :param ovth:   overlapping-threshold
        :param min_track_length: the threshold for the min-track-length to be considered
        :return: a dictionary containing the number of bboxes
                   for (1) true-positive (2) false-positive, and (3) total GT boxes
        """
        det_result = DetectionResult()
        gt_boxes = {}  # to collect all bounding boxes in all gt-tracks, and store them by time
        track_boxes = {}  # to collect all bboxes in all detected tracks, and store them by time

        if len(rframes) > 0 and len(lframes) > 0:
            gt_frame_num = max(lframes) - min(lframes)
            pt_valid_frame_num = len(rframes)
            fps = pt_valid_frame_num / gt_frame_num * 12
            det_result.fps = fps
        else:
            det_result.fps = 0.0

        # collect all bounding boxes in all gt-tracks, and store them by in a dict indexed by time
        for node in lnodes:  # for each GT track
            for i in range(node.size):  # for each frame
                frame = node.timestamps[i]  # frame-number (or time)
                box = np.array(node.features[FeatureTypes.Box2D][i])  # the bounding box
                gt_boxes.setdefault(frame, [])
                gt_boxes[frame].append(box.tolist())

        for node in rnodes:  # for each detected track
            if node.size < min_track_length:  # ignore very short fragments
                continue
            for i in range(node.size):  # for each frame
                frame = node.timestamps[i]  # frame-number (or time)
                box = np.array(node.features[FeatureTypes.Box2D][i])
                track_boxes.setdefault(frame, [])
                track_boxes[frame].append(box.tolist())

        # a dictionary containing the number of bboxed for (1) true-positive (2) false-positive, and (3) total GT boxes
        frames = set(gt_boxes.keys()) | set(
            track_boxes.keys())  # all frame-numbers appearing in either GT or detected tracks

        for frame in sorted(frames):  # for each frame that contains either GT bboxes or detected bboxes
            gt_boxes.setdefault(frame, [])  # dict returns [] if a key is missing
            track_boxes.setdefault(frame, [])  # dict returns [] if a key is missing

            label_boxes = gt_boxes[frame]
            det_result.gt += len(label_boxes)  # the total number of bbxoes in GT tracks
            det_result.nd += len(track_boxes[frame])

            t_boxes = track_boxes[frame]  # the detected bbox in the current frame
            if t_boxes and label_boxes:  # if the current frame contains both GT and detected bboxes
                for tbox in t_boxes:  # for each detected bbox
                    if not label_boxes:  # if no GT boxes. Redundant logic. Should be removed.
                        det_result.fp += 1  # count false positives
                        continue
                    ov = iou(np.array(tbox), np.array(label_boxes).reshape(
                        (-1, 4)))  # compute the iou between the detected-bbox with all GT-bboxes
                    idx = np.argmax(ov)  # find the GT bbox the overlap best with the detected-bbox
                    if ov[idx] > ovth:  # if the IOU is good enough
                        det_result.tp += 1  # count the number of true positive
                        del label_boxes[idx]  # a GT-bbox can only pair with one detected-bbox
                    else:  # if the detected bbox can't match any GT-bbox
                        det_result.fp += 1  # count the number of false-positive
            elif t_boxes:  # if the current frame contains only detected bboxes but no GT bboxes
                det_result.fp += len(t_boxes)  # count the number of false positives
        logger.info(f"{fname}-det_result: {det_result}")
        #  DetectionResult(gt=35660, nd=16414, fp=2021, tp=14393)
        return det_result

    @staticmethod
    def _evaluate_purity(fname, rnodes, lnodes, minov=0.4, imshape=(2560, 1440),
                         window_size=100, min_track_length=0, min_split_length=10, filter_nodes=True):
        """
        :param rnodes:                  detected tracks
        :param lnodes:                  Ground-truth tracks
        :param minov:                   the threshold (min-acceptable) for IOU
        :param imshape:                 the image size
        :param min_track_length:        the min track-length to be considered
        :return:
        """
        lsearch = TrackSearch(lnodes)
        # long frame intervals that contains no GT bboxes
        missing_frames = identify_missing_label_frames(lnodes, imshape)
        trk_result = TrackResult()
        trk_result.gt = len(lnodes)
        events = []

        # calculate the number of continus GT-tracks

        if min_track_length > 0:
            rnodes = [node for node in rnodes if node.size >= min_track_length]
            logger.info("{} number of tracks after filter out short tracks {}".format(len(rnodes), min_track_length))

        matches_mapping = {}

        for node in rnodes:
            midx = lsearch(node.mv_start, node.mv_end)
            candidates = [lnodes[i] for i in midx]
            ptr = [0 for _ in midx]
            for i in range(len(candidates)):
                ptr[i] = bisect_left(candidates[i].timestamps, node.timestamps[0])

            ov = np.zeros((len(candidates), node.size), dtype=np.float32)
            lengths = np.zeros((len(candidates),), dtype=np.int32)
            for i, t in enumerate(node.timestamps):
                box = np.array(node.features[FeatureTypes.Box2D][i])
                boxes = np.zeros((len(candidates), 4))
                for j in range(len(candidates)):
                    data = candidates[j]
                    while ptr[j] < data.size and data.timestamps[ptr[j]] < t:
                        ptr[j] += 1
                    if ptr[j] >= data.size:
                        continue
                    if data.timestamps[ptr[j]] == t:
                        boxes[j] = data.features[FeatureTypes.Box2D][ptr[j]]
                        lengths[j] += 1
                ov[:, i] = iou(box, boxes)
            valid = (ov > minov)

            if filter_nodes:
                for i in range(valid.shape[0]):
                    valid[i] = validate_both_sides(valid[i], window_size)

            matches = -np.ones((ov.shape[1],), dtype=np.int32)
            anyupdate = True
            midx = []
            match_str = ""

            while anyupdate and np.prod(ov.shape):
                anyupdate = False
                votes = ov.sum(axis=1) / np.maximum(lengths, 1.0)

                best_idx = votes.argmax()
                for i in range(matches.size):
                    if valid[best_idx, i] and matches[i] < 0:
                        matches[i] = best_idx
                        ov[:, i] = 0.0
                        anyupdate = True
                if anyupdate:
                    midx.append(candidates[best_idx].tid)
                    match_str += "{} ".format(candidates[best_idx].tid)
                    matches_mapping.setdefault(node.tid, [])
                    matches_mapping[node.tid].append(candidates[best_idx].tid)

                    assert candidates[best_idx].tid in missing_frames, \
                        "Need to mark out frames with missing labels to avoid evaluation noise"

                    for mseg in missing_frames[candidates[best_idx].tid]:
                        idx = (node.timestamps >= mseg[0]) & (
                                node.timestamps <= mseg[1])
                        valid[:, idx] = False
                        ov[:, idx] = 0
            if len(midx) > 1:
                # logger.info("Impure: Track {} matches with multiple GT {}".format(node.tid, match_str))
                ts = node.timestamps[matches >= 0]
                matches = matches[matches >= 0]
                matches = [candidates[m].tid for m in matches]

                lastm = -1  # matches[0]
                counter = 0
                num_splits = 0
                # do not care about too short switching
                if filter_nodes:
                    min_consecutive_frames = max(min_split_length, int(float(node.size) * 0.05))
                else:
                    min_consecutive_frames = 0

                if filter_nodes:
                    for i, t in enumerate(ts):
                        if lastm != matches[i]:
                            counter += 1
                            if counter > min_consecutive_frames:
                                if isinstance(lastm, int):
                                    if lastm > 0:
                                        logger.info(
                                            "vname {} Track {}: impurity from {} to {} at {}".format(fname, node.tid,
                                                                                                     lastm,
                                                                                                     matches[i], t))
                                        trk_result.splits += 1
                                        events.append(
                                            (node.tid, lastm, matches[i], ts[i - counter], ts[i - counter + 1]))
                                        num_splits += 1
                                elif isinstance(lastm, str):
                                    if lastm != "unknown":
                                        logger.info(
                                            "vname {} Track {}: impurity from {} to {} at {}".format(fname, node.tid,
                                                                                                     lastm,
                                                                                                     matches[i], t))
                                        trk_result.splits += 1
                                        events.append(
                                            (node.tid, lastm, matches[i], ts[i - counter], ts[i - counter + 1]))
                                        num_splits += 1

                                counter = 0
                                lastm = matches[i]
                        else:
                            counter = 0

                    nodes_dict = {node.tid: node for node in rnodes}

                    for event in events:
                        tid, _, _, switch_time, error_match_time = event
                        node = nodes_dict[tid]

                        switch_box = np.array(node.features[FeatureTypes.Box2D][
                                                  np.where(node.timestamps - switch_time)[0][0]
                                              ])
                        error_match_box = np.array(node.features[FeatureTypes.Box2D][
                                                       np.where(node.timestamps - error_match_time)[0][0]
                                                   ])
                        ov = iou(switch_box, np.array([error_match_box]))[0]

                        if error_match_time - switch_time < 5 and ov < 0.05:
                            trk_result.jump_case += 1
                else:
                    for i, t in enumerate(ts):
                        if lastm != matches[i]:
                            counter += 1
                            if counter > min_consecutive_frames:
                                # if lastm > 0:
                                logger.info(
                                    "Track{}: impurity from {} to {} at {}".format(node.tid, lastm, matches[i], t))
                                trk_result.splits += 1
                                events.append((node.tid, lastm, matches[i], t - counter))
                                num_splits += 1
                                counter = 0
                                lastm = matches[i]
                        else:
                            counter = 0
                if num_splits > 0:
                    trk_result.ntu += 1
            trk_result.nt += 1
            trk_result.pt += len(midx) > 0
        logger.info("There are {} number of tracks with {} positives.".format(trk_result.nt, trk_result.pt))
        logger.info(f"{fname}-trk_result:{trk_result}")
        # logger.debug(f"{fname}-trk_events:{events}")
        # logger.debug(f"{fname}-trk_matches_mapping:{matches_mapping}")
        return trk_result


def iou(bbox, candidates, eps=0.00001):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:4]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:4].prod(axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection + eps)


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to length 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def to_xyah(bbox_tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    bbox_tlwh = np.array(bbox_tlwh, dtype=np.float64)
    ret = bbox_tlwh.copy()[0:4]
    ret[:2] += ret[2:4] / 2
    ret[2] /= ret[3]
    return ret


def identify_missing_label_frames(nodes, imshape):
    """
    :param nodes: all labeled nodes (or tracks)
    :param imshape: the size of video frame
    :return: a list of list of 2-list (pair)
             each pair for a time interval
             a list of pairs for each track
             list of lists of pairs for all tracks
    """
    missing_frames = {}
    imbound = [0, 0, imshape[0], imshape[1]]

    def middle(box, margin=50):
        """ to determine if a box is in the middle of imbound.
            middle is defined w.r.t. the margin
        """
        if box[0] > margin and box[1] > margin and \
                box[0] + box[2] < imbound[2] - margin and \
                box[1] + box[3] < imbound[3] - margin:
            return True
        return False

    for node in nodes:  # calculate a list of frame-intervals for each track
        mframes = []
        prev_t = -1
        prev_box = None
        tol = 10  # tolerance

        for i in range(node.size):  # length of
            t = node.timestamps[i]  # the time point
            box = node.features[FeatureTypes.Box2D][i]  # the bbox at time t
            if prev_t > 0 and t - prev_t > tol and (middle(box) or middle(prev_box)):
                # only if disappeared in the middle of the image or
                # reappear in the middle of the image
                # then, unfortunately we need to ignore this case!
                mframes.append([prev_t + 1, t - 1])

            prev_t = t
            prev_box = box
        missing_frames[node.tid] = mframes  # missing frames are calculated for each track
    return missing_frames


def validate_both_sides(vlist, window_size):
    # windowed or operation
    windowed_validity = np.cumsum(np.array(vlist, dtype=int))
    if len(vlist) > window_size:
        windowed_validity[window_size:] -= windowed_validity[:-window_size]
    windowed_validity = windowed_validity.astype(np.bool)
    # check first n - window_size + 1 number of elements
    ret = np.logical_and(
        windowed_validity[:-(window_size - 1)],
        windowed_validity[(window_size - 1):])
    # check remaining at the right end
    remaining = min(window_size - 1, len(vlist) - ret.size)
    ret = np.concatenate((ret,
                          [windowed_validity[i + ret.size] and windowed_validity[-1] for i in range(remaining)]))
    assert ret.size == len(vlist)
    return ret


class TrackSearch(object):
    def __init__(self, nodes):
        self._nodes = nodes
        self._starts = np.array([node.mv_start for node in nodes])
        self._ends = np.array([node.mv_end for node in nodes])

    def __call__(self, left, right):
        """
        Return the indices of all tracks that overlap with (left, right) exclusively
        需要等于，否则在单独只有一帧的情况轨迹评估会出现bug
        """
        return np.where((self._starts <= right) & (self._ends >= left))[0]


# 不要直接调用这个函数，调用下面的 check_box_in_multi_ignore_region_or_not 函数
def check_box_in_ignore_region_or_not(ignore_mask: np.array, box: List[int], image_size: Tuple[int, int],
                                      iou_threshold: float = 0.8):
    x, y, w, h = box
    W, H = image_size
    image = np.ones((H, W), dtype=np.uint8)
    image[y: y + h, x: x + w] = 0
    is_box_in_ignore = np.logical_and(image == 0, ignore_mask == 0)
    if np.sum(is_box_in_ignore) < (w * h * iou_threshold):
        return False
    else:
        return True


def check_box_in_multi_ignore_region_or_not(ignore_masks: List[np.array], car_box: List[int],
                                            image_size: Tuple[int, int], iou_threshold: float = 0.8):
    car_box = [int(_) for _ in car_box]
    for ignore_mask in ignore_masks:
        if len(ignore_mask.shape) == 3:
            ignore_mask = cv2.cvtColor(ignore_mask, cv2.COLOR_RGB2GRAY)
            ignore_mask = cv2.threshold(ignore_mask, 200, 1, type=cv2.THRESH_BINARY)[1]
            ignore_mask = 1 - ignore_mask

        if check_box_in_ignore_region_or_not(ignore_mask, car_box, image_size, iou_threshold):
            return True
    return False


def load_mask_from_image_file(image_path: str):
    masked_image = cv2.imread(image_path)
    return masked_image
