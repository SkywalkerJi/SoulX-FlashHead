"""RealtimeProcessor 单元测试 — 纯 numpy，无 GPU/torch 依赖。"""
import time

import numpy as np
import pytest

from flash_head.realtime.processor import RealtimeProcessor

SLICE_LEN = 24
SR = 16000
FPS = 25
OUT_SR = 48000
SLICE_16K = SLICE_LEN * SR // FPS          # 15360
SLICE_OUT = SLICE_LEN * OUT_SR // FPS      # 46080
PER_FRAME_OUT = OUT_SR // FPS              # 1920


class FakeGen:
    """记录收到的滑窗，返回带序号图案的帧。"""
    def __init__(self, h=4, w=4):
        self.windows = []
        self.h, self.w = h, w
        self.calls = 0

    def __call__(self, window):
        self.windows.append(window.copy())
        frames = np.full((SLICE_LEN, self.h, self.w, 3), self.calls % 256, dtype=np.uint8)
        self.calls += 1
        return frames


def make_proc(**kw):
    gen = FakeGen()
    proc = RealtimeProcessor(
        generate_chunk=gen, slice_len=SLICE_LEN, sample_rate=SR, tgt_fps=FPS,
        cached_audio_duration=8, output_sample_rate=OUT_SR, **kw,
    )
    return proc, gen


def test_less_than_one_slice_extracts_nothing():
    proc, _ = make_proc()
    proc.add_audio(np.ones(SLICE_16K - 1, np.float32), np.ones(SLICE_OUT - 3, np.float32))
    assert proc._try_extract_slice() is None


def test_exactly_one_slice_extracts_once():
    proc, _ = make_proc()
    a16 = np.arange(SLICE_16K, dtype=np.float32) / SLICE_16K
    aout = np.arange(SLICE_OUT, dtype=np.float32) / SLICE_OUT
    proc.add_audio(a16, aout)
    got = proc._try_extract_slice()
    assert got is not None
    s16, sout = got
    np.testing.assert_allclose(s16, a16)
    np.testing.assert_allclose(sout, aout)
    assert proc._try_extract_slice() is None  # 已消费完


def test_remainder_is_kept_for_next_slice():
    proc, _ = make_proc()
    one_and_half_16 = np.ones(SLICE_16K + SLICE_16K // 2, np.float32)
    one_and_half_out = np.ones(SLICE_OUT + SLICE_OUT // 2, np.float32)
    proc.add_audio(one_and_half_16, one_and_half_out)
    assert proc._try_extract_slice() is not None
    assert proc._try_extract_slice() is None
    proc.add_audio(np.ones(SLICE_16K // 2, np.float32), np.ones(SLICE_OUT // 2, np.float32))
    assert proc._try_extract_slice() is not None


def test_short_out_audio_padded_to_nominal():
    """麦克风抖动导致 48k 侧样本略少时，补零到名义长度，保证逐帧配对不塌。"""
    proc, _ = make_proc()
    proc.add_audio(np.ones(SLICE_16K, np.float32), np.ones(SLICE_OUT - 100, np.float32))
    got = proc._try_extract_slice()
    assert got is not None
    _, sout = got
    assert len(sout) == SLICE_OUT
    np.testing.assert_allclose(sout[-100:], 0.0)
