"""Profiler系统 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.system import ProfilerSystem

class TestInstantFPS:
    def test_spike_detected(self):
        ps = ProfilerSystem()
        ps.sample_frame(16, 8, 8)
        ps.sample_frame(100, 50, 50)
        fps = ps.get_fps()
        assert fps < 60

class TestCallstackLine:
    def test_callstack_has_line_number(self):
        ps = ProfilerSystem()
        ps.record_callstack("func", 1.0, line=42)
        assert ps.call_stacks[-1].get("line") == 42

class TestGPUMemory:
    def test_gpu_memory_tracked(self):
        ps = ProfilerSystem()
        ps.track_memory(1024, "gpu")
        assert ps.memory_samples[-1].get("type") == "gpu"

class TestNestedMarker:
    def test_nested_markers_supported(self):
        ps = ProfilerSystem()
        ps.add_marker("outer", data={"nested": True})
        assert "data" in ps.call_stacks[-1]

class TestP99Report:
    def test_report_includes_p99(self):
        ps = ProfilerSystem()
        ps.sample_frame(16, 8, 8)
        ps.sample_frame(100, 50, 50)
        report = ps.generate_report()
        assert "p99" in report
