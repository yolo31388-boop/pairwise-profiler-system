"""Profiler系统 - 含5个bug"""
from dataclasses import dataclass, field
import time

@dataclass
class FrameSample:
    frame_time: float
    cpu_time: float
    gpu_time: float

class ProfilerSystem:
    def __init__(self):
        self.frame_samples: list[FrameSample] = []
        self.call_stacks: list = []
        self.memory_samples: list = []

    def sample_frame(self, frame_time: float, cpu_time: float, gpu_time: float) -> None:
        # bug1: 平均帧率，大窗口
        self.frame_samples.append(FrameSample(frame_time, cpu_time, gpu_time))

    def get_fps(self) -> float:
        # bug1: 平均帧率
        if not self.frame_samples:
            return 0
        avg_time = sum(s.frame_time for s in self.frame_samples) / len(self.frame_samples)
        return 1000 / avg_time if avg_time > 0 else 0

    def record_callstack(self, func_name: str, duration: float, line: int = 0) -> None:
        # bug2: 不记录行号，不采样
        self.call_stacks.append({"func": func_name, "duration": duration})

    def track_memory(self, size: int, callstack: str = "") -> None:
        # bug3: 只跟踪堆内存
        self.memory_samples.append({"size": size})

    def add_marker(self, name: str, color: str = "white", data: dict = None) -> None:
        # bug4: 不支持嵌套
        self.call_stacks.append({"marker": name, "color": color})

    def generate_report(self) -> dict:
        # bug5: 只显示平均
        if not self.frame_samples:
            return {}
        avg = sum(s.frame_time for s in self.frame_samples) / len(self.frame_samples)
        return {"avg_frame_time": avg}
