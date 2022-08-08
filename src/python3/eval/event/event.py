import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


class Event:
    def __init__(self, face_crop_json, tsf_path, event_json_path, gap=10000,
                 event_type="ENTER", limit=0, debug=False):
        with open(face_crop_json, 'r') as f:
            bm_list = json.load(f)
        self.gt_timestemps = self._parse_bm(bm_list, tsf_path)
        self.pred_timestemps = self._parse_pred(event_json_path, event_type,
                                                limit)
        self.debug = debug
        self.tp, self.fn, self.fp, self.recall, self.precision = self.eval(
            self.gt_timestemps, self.pred_timestemps,
            gap, debug=False)

    @staticmethod
    def _parse_bm(bm_list, tsf_path):
        event_timestemps = []
        for index, event in enumerate(bm_list):
            filename = event["filename"]
            tsf_name = '_'.join(filename.split("_")[5:7]) + ".mp4.cut.mp4.tsf"
            cur_frame = event["cur_frame"]
            with open(f"{os.path.join(tsf_path, tsf_name)}", 'r') as f:
                timestemp = ''.join(
                    f.readlines()[cur_frame - 1:cur_frame:1]).strip()
                event_timestemps.append(int(timestemp))
        return event_timestemps

    def __str__(self):
        return f"recall: {self.recall}\nprecision: {self.precision}"

    @staticmethod
    def _parse_event_json(json_path):
        json_files = gglob(os.path.join(json_path, "*json"))
        pred_list = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                pred_list.append(json.load(f))
        return pred_list

    @staticmethod
    def _parse_pred(json_path, event_type, limit):
        pred_list = SingViewEvent._parse_event_json(json_path)
        event_timestemps = []
        for events in pred_list:
            for event in events['events']:
                if event['event_type'] == event_type and int(
                        event['start_time']) > limit:
                    timestemp = event['start_time']
                    event_timestemps.append(int(timestemp))
        return event_timestemps

    # ERROR： 列表里面套列表[[]]，通过二维数组的方式改变列表内数值，会出现改变全部数值的情况。
    """
    通过方形消费矩阵进行最大匹配下求最小消费总和组合。
    Parameters
    ----------
    gt_list: GT以事件发生事件时间戳。
    pred_list: 程序跑出来的事件时间戳。
    gap: 当前默认为10,单位为s。
    Returns
    -------
    还未想好返回什么。
    """

    @staticmethod
    def eval(gt_list, pred_list, gap=10, debug=False):
        fp = []
        fn = []
        tp = []
        len_max = max(len(gt_list), len(pred_list))
        cost_matrix = [[sys.maxsize] * len_max] * len_max
        gt_list.extend([sys.maxsize] * (len_max - len(gt_list)))
        pred_list.extend([sys.maxsize] * (len_max - len(pred_list)))
        cost_matrix = np.asarray(cost_matrix)
        for row_index, pred in enumerate(cost_matrix):
            for col_index, gt in enumerate(pred):
                cost = abs(gt_list[col_index] - pred_list[row_index])
                cost_matrix[row_index][
                    col_index] = cost if cost <= gap else sys.maxsize
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
        # 提取每个行索引的最优指派列索引所在的元素，形成数组]
        for index, cost in enumerate(cost_matrix[row_ind, col_ind]):
            if cost <= gap:
                tp.append(index)
            if cost > gap:
                if gt_list[col_ind[index]] != sys.maxsize:
                    fn.append((col_ind[index], gt_list[col_ind[index]]))
                if pred_list[row_ind[index]] != sys.maxsize:
                    fp.append((row_ind[index], pred_list[row_ind[index]]))
        precision = len(tp) / (len(tp) + len(fp))
        recall = len(tp) / (len(tp) + len(fn))
        return tp, fn, fp, recall, precision
