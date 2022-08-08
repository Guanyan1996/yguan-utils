from dataclasses import dataclass, field, asdict


@dataclass
class DetectionResult:
    gt: int = field(default_factory=int)
    nd: int = field(default_factory=int)
    fp: int = field(default_factory=int)
    tp: int = field(default_factory=int)
    fps: int = field(default_factory=int)

    def __add__(self, other):
        assert isinstance(other, DetectionResult)
        this = asdict(self)
        other = asdict(other)
        for key in this.keys():
            if isinstance(this[key], int):
                this[key] += other[key]
        return DetectionResult(**this)


@dataclass
class TrackResult:
    gt: int = field(default_factory=int)
    nt: int = field(default_factory=int)
    pt: int = field(default_factory=int)
    ntu: int = field(default_factory=int)
    splits: int = field(default_factory=int)
    jump_case: int = field(default_factory=int)

    def __add__(self, other):
        assert isinstance(other, TrackResult)
        this = asdict(self)
        other = asdict(other)
        for key in this.keys():
            if isinstance(this[key], int):
                this[key] += other[key]
        return TrackResult(**this)


@dataclass
class Node2DEvalResult:
    fname: field
    det_result: field(default_factory=DetectionResult)
    trk_result: field(default_factory=TrackResult)

    def __add__(self, other):
        # 会把引用类转成dict进来。
        assert isinstance(other, Node2DEvalResult)
        det_result = self.det_result + other.det_result
        trk_result = self.trk_result + other.trk_result
        return Node2DEvalResult(fname="COUNT", det_result=det_result, trk_result=trk_result)

    @property
    def result(self):
        base = {}
        base['fname'] = self.fname
        base['det_gt'] = self.det_result.gt
        base['det_p'] = self.det_p
        base['det_r'] = self.det_r
        base['t_gt'] = self.trk_result.gt
        base['nt_div_gt'] = self.nt_div_gt
        base['pt_div_gt'] = self.pt_div_gt
        base['ntu_div_gt'] = self.ntu_div_gt
        base['splits_div_ntu'] = self.splits_div_ntu
        base['jump_case'] = self.trk_result.jump_case
        base['fps'] = round(self.det_result.fps, 2)
        return base

    @property
    def det_p(self) -> float:
        return round(float(self.det_result.tp) / (self.det_result.tp + self.det_result.fp + 1e-10), 3)

    @property
    def det_r(self) -> float:
        return round(float(self.det_result.tp) / (self.det_result.gt + 1e-10), 3)

    @property
    def nt_div_gt(self) -> float:
        return round(float(self.trk_result.nt) / (self.trk_result.gt + 1e-10), 3)

    @property
    def pt_div_gt(self) -> float:
        return round(float(self.trk_result.pt) / (self.trk_result.gt + 1e-10), 3)

    @property
    def ntu_div_gt(self) -> float:
        return round(float(self.trk_result.ntu) / (self.trk_result.gt + 1e-10), 3)

    @property
    def splits_div_ntu(self) -> float:
        return round(float(self.trk_result.splits) / (self.trk_result.ntu + 1e-10), 3)
