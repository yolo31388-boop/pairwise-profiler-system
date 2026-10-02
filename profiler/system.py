"""Profiler系统

修复点:
1. 帧率统计: 瞬时帧率 + 小滑动窗口, 区分 CPU/GPU 时间
2. 调用栈: 记录行号, 按采样率采样, 按调用路径聚合热点
3. 内存统计: 区分 heap/gpu/cache, 记录分配调用栈, 支持快照对比
4. 性能标记: 支持嵌套, 颜色与分类, 自定义数据
5. 性能报告: 平均/P99/最大值, 火焰图, 支持 JSON 导出
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import random
import time


WINDOW_FRAMES = 60          # 约1秒的小窗口(60fps), 不再用10秒大窗口
DEFAULT_SAMPLE_RATE = 1.0   # 调用栈采样率(0.0~1.0)
MEMORY_TYPES = ("heap", "gpu", "cache")


@dataclass
class FrameSample:
    frame_time: float
    cpu_time: float
    gpu_time: float
    timestamp: float = field(default_factory=time.time)


class ProfilerSystem:
    def __init__(self, window_size: int = WINDOW_FRAMES,
                 sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.window_size = window_size
        # fix1: 小滑动窗口而非无限增长的全量历史
        self.frame_samples: deque[FrameSample] = deque(maxlen=window_size)
        # fix2: 采样后的原始调用栈记录(含行号)
        self.call_stacks: list[dict] = []
        self.memory_samples: list[dict] = []
        self.markers: list[dict] = []
        # fix2: 调用栈聚合结果 key=(call_path, func, line)
        self.callstack_agg: dict[tuple, dict] = defaultdict(
            lambda: {"count": 0, "total_duration": 0.0, "max_duration": 0.0}
        )
        self.sample_rate = sample_rate
        self._rng = random.Random()
        # fix4: 嵌套标记栈
        self._marker_stack: list[dict] = []

    # ------------------------------------------------------------------
    # 1. 帧率统计
    # ------------------------------------------------------------------
    def sample_frame(self, frame_time: float, cpu_time: float,
                     gpu_time: float) -> None:
        self.frame_samples.append(
            FrameSample(frame_time, cpu_time, gpu_time)
        )

    def get_fps(self) -> float:
        """瞬时帧率: 基于最近一帧的帧时间, 卡顿立刻能看到。"""
        if not self.frame_samples:
            return 0.0
        latest = self.frame_samples[-1]
        return 1000.0 / latest.frame_time if latest.frame_time > 0 else 0.0

    def get_window_fps(self) -> float:
        """小窗口帧率(约1秒), 用于平滑展示, 不再是10秒平均。"""
        if not self.frame_samples:
            return 0.0
        avg_time = sum(s.frame_time for s in self.frame_samples) / len(
            self.frame_samples
        )
        return 1000.0 / avg_time if avg_time > 0 else 0.0

    def get_frame_breakdown(self) -> dict:
        """区分 CPU / GPU 时间, 判断瓶颈在哪一侧。"""
        if not self.frame_samples:
            return {"cpu": 0.0, "gpu": 0.0}
        return {
            "cpu": self.frame_samples[-1].cpu_time,
            "gpu": self.frame_samples[-1].gpu_time,
        }

    # ------------------------------------------------------------------
    # 2. 调用栈采样与聚合
    # ------------------------------------------------------------------
    def record_callstack(self, func_name: str, duration: float,
                         line: int = 0, path: str = "",
                         sample_rate: float | None = None) -> None:
        """记录一次函数调用。

        fix2: 记录行号; 按采样率采样以降低 profiler 自身开销;
        同时按 (path, func, line) 聚合, 便于定位热点。
        """
        rate = self.sample_rate if sample_rate is None else sample_rate
        if rate < 1.0 and self._rng.random() >= rate:
            return

        record = {
            "func": func_name,
            "line": line,
            "path": path,
            "duration": duration,
        }
        self.call_stacks.append(record)

        key = (path, func_name, line)
        agg = self.callstack_agg[key]
        agg["count"] += 1
        agg["total_duration"] += duration
        if duration > agg["max_duration"]:
            agg["max_duration"] = duration

    def get_hotspots(self, top_n: int | None = None) -> list[dict]:
        """聚合后的调用热点, 相同调用路径合并显示。"""
        rows = [
            {
                "func": func,
                "line": line,
                "path": path,
                "count": v["count"],
                "total_duration": v["total_duration"],
                "avg_duration": v["total_duration"] / v["count"],
                "max_duration": v["max_duration"],
            }
            for (path, func, line), v in self.callstack_agg.items()
        ]
        rows.sort(key=lambda r: r["total_duration"], reverse=True)
        return rows[:top_n] if top_n is not None else rows

    # ------------------------------------------------------------------
    # 3. 内存统计
    # ------------------------------------------------------------------
    def track_memory(self, size: int, mem_type: str = "heap",
                     callstack: str = "") -> None:
        """跟踪内存分配。

        fix3: mem_type 支持 heap/gpu/cache, 记录分配调用栈,
        便于和任务管理器对账并定位泄漏来源。
        """
        if mem_type not in MEMORY_TYPES:
            raise ValueError(
                f"unknown memory type: {mem_type!r}, expect {MEMORY_TYPES}"
            )
        self.memory_samples.append({
            "size": int(size),
            "type": mem_type,
            "callstack": callstack,
            "timestamp": time.time(),
        })

    def get_memory_by_type(self) -> dict[str, int]:
        totals = dict.fromkeys(MEMORY_TYPES, 0)
        for sample in self.memory_samples:
            totals[sample["type"]] += sample["size"]
        return totals

    def get_total_memory(self) -> int:
        return sum(s["size"] for s in self.memory_samples)

    def take_memory_snapshot(self) -> dict:
        """内存快照: 按类型汇总, 并保留每个分配点(callstack)的占用。"""
        by_type = self.get_memory_by_type()
        by_callstack: dict[str, int] = defaultdict(int)
        for sample in self.memory_samples:
            key = f"{sample['type']}:{sample['callstack'] or 'unknown'}"
            by_callstack[key] += sample["size"]
        return {
            "timestamp": time.time(),
            "total": sum(by_type.values()),
            "by_type": by_type,
            "by_callstack": dict(by_callstack),
        }

    @staticmethod
    def diff_memory_snapshots(before: dict, after: dict) -> dict:
        """两次快照对比, 输出各类内存和各调用点的增长量。"""
        types = sorted(set(before["by_type"]) | set(after["by_type"]))
        by_type = {
            t: after["by_type"].get(t, 0) - before["by_type"].get(t, 0)
            for t in types
        }
        keys = set(before["by_callstack"]) | set(after["by_callstack"])
        by_callstack = {
            k: after["by_callstack"].get(k, 0)
            - before["by_callstack"].get(k, 0)
            for k in keys
        }
        growing = {k: v for k, v in by_callstack.items() if v > 0}
        return {
            "total_delta": after["total"] - before["total"],
            "by_type": by_type,
            "by_callstack": by_callstack,
            "growing": growing,
        }

    # ------------------------------------------------------------------
    # 4. 性能标记(嵌套/颜色/分类/自定义数据)
    # ------------------------------------------------------------------
    def begin_marker(self, name: str, color: str = "white",
                     category: str = "generic", data: dict | None = None
                     ) -> dict:
        marker = {
            "marker": name,
            "name": name,
            "color": color,
            "category": category,
            "data": dict(data or {}),
            "depth": len(self._marker_stack),
            "parent": self._marker_stack[-1]["name"]
            if self._marker_stack else None,
            "start_time": time.time(),
            "end_time": None,
            "children": [],
        }
        if self._marker_stack:
            self._marker_stack[-1]["children"].append(marker)
        self._marker_stack.append(marker)
        return marker

    def end_marker(self) -> dict | None:
        if not self._marker_stack:
            return None
        marker = self._marker_stack.pop()
        marker["end_time"] = time.time()
        marker["duration"] = marker["end_time"] - marker["start_time"]
        if not self._marker_stack:
            self.markers.append(marker)
            self.call_stacks.append(marker)
        return marker

    def add_marker(self, name: str, color: str = "white",
                   category: str = "generic",
                   data: dict | None = None) -> dict:
        """记录一个(可能嵌套的)性能标记。

        fix4: 标记可通过 begin/end 嵌套, 带颜色与分类, 可携带自定义数据;
        内层标记不再覆盖外层。
        """
        marker = self.begin_marker(name, color, category, data)
        return self.end_marker()

    def get_markers_by_category(self, category: str) -> list[dict]:
        def _walk(node: dict) -> list[dict]:
            found = [node] if node["category"] == category else []
            for child in node["children"]:
                found.extend(_walk(child))
            return found

        result: list[dict] = []
        for root in self.markers:
            result.extend(_walk(root))
        return result

    # ------------------------------------------------------------------
    # 5. 性能报告(分布/火焰图/导出)
    # ------------------------------------------------------------------
    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        index = (len(sorted_values) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        frac = index - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac

    def _frame_stats(self, attr: str) -> dict:
        values = sorted(getattr(s, attr) for s in self.frame_samples)
        if not values:
            return {}
        return {
            "avg": sum(values) / len(values),
            "p50": self._percentile(values, 0.50),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
            "max": values[-1],
            "min": values[0],
        }

    def build_flame_graph(self) -> dict:
        """基于聚合调用栈构建火焰图(嵌套树, value 为总耗时)。"""
        root: dict = {"name": "root", "value": 0.0, "children": {}}
        for row in self.get_hotspots():
            label = f"{row['func']}:{row['line']}"
            parts = [p for p in row["path"].split("/") if p]
            node = root
            for part in parts + [label]:
                children = node["children"]
                if part not in children:
                    children[part] = {"name": part, "value": 0.0,
                                      "children": {}}
                node = children[part]
                node["value"] += row["total_duration"]
            root["value"] += row["total_duration"]
        return root

    def generate_report(self) -> dict:
        if not self.frame_samples:
            return {}
        frame_stats = self._frame_stats("frame_time")
        cpu_stats = self._frame_stats("cpu_time")
        gpu_stats = self._frame_stats("gpu_time")
        return {
            "frame_count": len(self.frame_samples),
            "instant_fps": self.get_fps(),
            "window_fps": self.get_window_fps(),
            "frame_time": frame_stats,
            "cpu_time": cpu_stats,
            "gpu_time": gpu_stats,
            # fix5: P99/最大值, 偶发卡顿可见
            "avg_frame_time": frame_stats["avg"],
            "p99": frame_stats["p99"],
            "max_frame_time": frame_stats["max"],
            "hotspots": self.get_hotspots(top_n=20),
            "memory": {
                "by_type": self.get_memory_by_type(),
                "total": self.get_total_memory(),
            },
            "markers": self.markers,
            "flame_graph": self.build_flame_graph(),
        }

    def export_report(self, filepath: str) -> str:
        """把报告导出为 JSON 文件, 支持在游戏外分析。"""
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.generate_report(), fh, ensure_ascii=False,
                      indent=2, default=str)
        return filepath
