"""Profiler系统

修复点:
1. 帧率: 瞬时帧率(最近一帧), 小滑动窗口统计, 区分CPU/GPU时间
2. 调用栈: 记录行号, 按采样率采样, 按调用路径聚合热点
3. 内存: 区分heap/gpu/cache等类型, 记录分配调用栈, 支持快照对比
4. 标记: 支持嵌套栈, 颜色/分类/自定义数据
5. 报告: avg/p95/p99/max, 火焰图, JSON导出
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import math
import random
import time


@dataclass
class FrameSample:
    frame_time: float
    cpu_time: float
    gpu_time: float


def _percentile(sorted_values: list[float], pct: float) -> float:
    """nearest-rank百分位, pct取0-100. 偶发尖峰不会被插值稀释."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return float(sorted_values[rank - 1])


class ProfilerSystem:
    # 小窗口: 1秒(60帧), 旧实现是10秒(600帧), 卡顿结束后才反映出来
    DEFAULT_WINDOW = 60

    def __init__(self, window_size: int = DEFAULT_WINDOW, sample_rate: float = 1.0):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.frame_samples: deque[FrameSample] = deque(maxlen=window_size)
        self.call_stacks: list[dict] = []
        self.memory_samples: list[dict] = []
        self.markers: list[dict] = []
        self._marker_stack: list[dict] = []
        # 调用路径 -> 聚合统计
        self.callstack_aggregates: dict[tuple, dict] = {}
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # 帧率统计: 瞬时帧率 + 小窗口 + CPU/GPU 分离
    # ------------------------------------------------------------------
    def sample_frame(self, frame_time: float, cpu_time: float, gpu_time: float) -> None:
        self.frame_samples.append(FrameSample(frame_time, cpu_time, gpu_time))

    def get_fps(self) -> float:
        """瞬时帧率: 只看最近一帧, 卡顿当帧立刻暴露."""
        if not self.frame_samples:
            return 0.0
        last = self.frame_samples[-1].frame_time
        return 1000.0 / last if last > 0 else 0.0

    def get_window_fps(self) -> float:
        """小窗口内平均帧率(窗口默认1秒)."""
        if not self.frame_samples:
            return 0.0
        avg_time = sum(s.frame_time for s in self.frame_samples) / len(self.frame_samples)
        return 1000.0 / avg_time if avg_time > 0 else 0.0

    def get_cpu_fps(self) -> float:
        if not self.frame_samples:
            return 0.0
        t = self.frame_samples[-1].cpu_time
        return 1000.0 / t if t > 0 else 0.0

    def get_gpu_fps(self) -> float:
        if not self.frame_samples:
            return 0.0
        t = self.frame_samples[-1].gpu_time
        return 1000.0 / t if t > 0 else 0.0

    def get_bottleneck(self) -> str:
        """返回当前帧瓶颈在 cpu / gpu / none."""
        if not self.frame_samples:
            return "none"
        s = self.frame_samples[-1]
        if s.cpu_time > s.gpu_time:
            return "cpu"
        if s.gpu_time > s.cpu_time:
            return "gpu"
        return "none"

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = max(0.0, min(1.0, rate))

    # ------------------------------------------------------------------
    # 调用栈: 行号 + 采样 + 聚合
    # ------------------------------------------------------------------
    def record_callstack(
        self,
        func_name: str,
        duration: float,
        line: int = 0,
        stack: tuple | list = (),
    ) -> None:
        # 采样: 按采样率丢弃部分调用, 避免Profiler本身拖慢游戏
        if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
            return
        path = tuple(stack)
        record = {
            "func": func_name,
            "duration": duration,
            "line": line,
            "stack": list(path),
            "timestamp": time.perf_counter(),
        }
        self.call_stacks.append(record)

        key = path + (f"{func_name}:{line}",)
        agg = self.callstack_aggregates.get(key)
        if agg is None:
            agg = {"func": func_name, "line": line, "stack": list(path),
                   "count": 0, "total_time": 0.0, "min_time": duration,
                   "max_time": duration}
            self.callstack_aggregates[key] = agg
        agg["count"] += 1
        agg["total_time"] += duration
        agg["min_time"] = min(agg["min_time"], duration)
        agg["max_time"] = max(agg["max_time"], duration)

    def get_aggregated_callstacks(self, top: int | None = None) -> list[dict]:
        """按总耗时降序返回聚合热点."""
        rows = list(self.callstack_aggregates.values())
        rows.sort(key=lambda r: r["total_time"], reverse=True)
        return rows[:top] if top else rows

    # ------------------------------------------------------------------
    # 内存: 多类型 + 分配调用栈 + 快照对比
    # ------------------------------------------------------------------
    def track_memory(self, size: int, mem_type: str = "heap", callstack: str = "") -> None:
        self.memory_samples.append({
            "size": size,
            "type": mem_type,
            "callstack": callstack,
            "timestamp": time.perf_counter(),
        })

    def get_memory_usage(self) -> dict[str, int]:
        """按类型汇总当前跟踪到的内存(含heap/gpu/cache等全部类型)."""
        usage: dict[str, int] = defaultdict(int)
        for sample in self.memory_samples:
            usage[sample["type"]] += sample["size"]
        return dict(usage)

    def get_total_memory(self) -> int:
        return sum(s["size"] for s in self.memory_samples)

    def take_memory_snapshot(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        by_callstack: dict[str, int] = defaultdict(int)
        for sample in self.memory_samples:
            by_type[sample["type"]] += sample["size"]
            key = f"[{sample['type']}] {sample['callstack'] or '<unknown>'}"
            by_callstack[key] += sample["size"]
        return {
            "total": sum(by_type.values()),
            "by_type": dict(by_type),
            "by_callstack": dict(by_callstack),
        }

    @staticmethod
    def diff_memory_snapshots(before: dict, after: dict) -> dict:
        """两个快照对比, 输出增长量(只保留增长项外的全部变化也可直接看)."""
        def _delta(b: dict, a: dict) -> dict:
            out: dict = {}
            for key in set(a) | set(b):
                delta = a.get(key, 0) - b.get(key, 0)
                if delta:
                    out[key] = delta
            return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

        return {
            "total_delta": after.get("total", 0) - before.get("total", 0),
            "by_type": _delta(before.get("by_type", {}), after.get("by_type", {})),
            "by_callstack": _delta(
                before.get("by_callstack", {}), after.get("by_callstack", {})
            ),
        }

    # ------------------------------------------------------------------
    # 性能标记: 嵌套 + 颜色/分类 + 自定义数据
    # ------------------------------------------------------------------
    def add_marker(
        self,
        name: str,
        color: str = "white",
        category: str = "",
        data: dict | None = None,
    ) -> dict:
        marker = {
            "marker": name,
            "name": name,
            "color": color,
            "category": category,
            "data": dict(data) if data else {},
            "depth": len(self._marker_stack),
            "start_time": time.perf_counter(),
            "end_time": None,
            "duration": None,
            "children": [],
        }
        if self._marker_stack:
            self._marker_stack[-1]["children"].append(marker)
        self._marker_stack.append(marker)
        self.markers.append(marker)
        # 兼容旧接口: 标记也写入 call_stacks
        self.call_stacks.append(marker)
        return marker

    def end_marker(self, name: str | None = None) -> dict | None:
        if not self._marker_stack:
            return None
        if name is not None and self._marker_stack[-1]["name"] != name:
            # 不允许跨层关闭: 找到对应标记, 自动关闭其内层
            while self._marker_stack and self._marker_stack[-1]["name"] != name:
                self._close(self._marker_stack[-1])
            if not self._marker_stack:
                return None
        return self._close(self._marker_stack[-1])

    def _close(self, marker: dict) -> dict:
        marker["end_time"] = time.perf_counter()
        marker["duration"] = marker["end_time"] - marker["start_time"]
        if self._marker_stack and self._marker_stack[-1] is marker:
            self._marker_stack.pop()
        return marker

    def get_marker_tree(self) -> list[dict]:
        """返回顶层标记(嵌套关系在 children 中)."""
        roots: list[dict] = []
        for marker in self.markers:
            if marker["depth"] == 0:
                roots.append(marker)
        return roots

    # ------------------------------------------------------------------
    # 报告: avg/p95/p99/max + CPU/GPU + 火焰图 + 导出
    # ------------------------------------------------------------------
    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(values)
        return {
            "avg": sum(values) / len(values),
            "min": ordered[0],
            "max": ordered[-1],
            "p95": _percentile(ordered, 95),
            "p99": _percentile(ordered, 99),
        }

    def generate_flamegraph(self) -> dict:
        """根据聚合调用路径生成火焰图树(value为总耗时)."""
        root = {"name": "root", "value": 0.0, "children": {}}

        def walk(node: dict) -> dict:
            return {"name": node["name"], "value": node["value"],
                    "children": [walk(c) for c in node["children"].values()]}

        for key, agg in self.callstack_aggregates.items():
            node = root
            for frame in key:
                child = node["children"].setdefault(
                    frame, {"name": frame, "value": 0.0, "children": {}}
                )
                child["value"] += agg["total_time"]
                node["value"] += agg["total_time"]
                node = child
        return walk(root)

    def generate_report(self) -> dict:
        if not self.frame_samples:
            return {}
        frames = list(self.frame_samples)
        frame_stats = self._stats([s.frame_time for s in frames])
        cpu_stats = self._stats([s.cpu_time for s in frames])
        gpu_stats = self._stats([s.gpu_time for s in frames])
        return {
            "frame_count": len(frames),
            "window_size": self.window_size,
            "fps": self.get_fps(),
            "window_fps": self.get_window_fps(),
            "cpu_fps": self.get_cpu_fps(),
            "gpu_fps": self.get_gpu_fps(),
            "bottleneck": self.get_bottleneck(),
            "avg_frame_time": frame_stats["avg"],
            "max_frame_time": frame_stats["max"],
            "min_frame_time": frame_stats["min"],
            "p95": frame_stats["p95"],
            "p99": frame_stats["p99"],
            "frame_time": frame_stats,
            "cpu_time": cpu_stats,
            "gpu_time": gpu_stats,
            "memory": {
                "total": self.get_total_memory(),
                "by_type": self.get_memory_usage(),
            },
            "callstacks": self.get_aggregated_callstacks(),
            "flamegraph": self.generate_flamegraph(),
            "markers": self.get_marker_tree(),
        }

    def export_report(self, path: str) -> str:
        """导出报告为JSON文件, 返回路径."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.generate_report(), fh, ensure_ascii=False, indent=2,
                      default=str)
        return path
