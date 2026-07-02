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


def _feed_one_slice(proc, value16=0.5, value_out=0.25):
    a16 = np.full(SLICE_16K, value16, np.float32)
    aout = np.full(SLICE_OUT, value_out, np.float32)
    proc.add_audio(a16, aout)
    got = proc._try_extract_slice()
    assert got is not None
    proc._process_slice(*got)


def test_process_slice_enqueues_paired_frames_and_audio():
    proc, gen = make_proc()
    _feed_one_slice(proc, value_out=0.25)
    pairs = []
    while True:
        p = proc.get_pair()
        if p is None:
            break
        pairs.append(p)
    assert len(pairs) == SLICE_LEN
    frame, seg = pairs[0]
    assert frame.shape == (4, 4, 3) and frame.dtype == np.uint8
    assert seg.shape == (PER_FRAME_OUT,)
    # 24 段音频拼回来应等于整片
    joined = np.concatenate([s for _, s in pairs])
    np.testing.assert_allclose(joined, np.full(SLICE_OUT, 0.25, np.float32))


def test_window_slides_with_input_slice():
    proc, gen = make_proc()
    marker = np.linspace(0.1, 0.9, SLICE_16K).astype(np.float32)
    proc.add_audio(marker, np.zeros(SLICE_OUT, np.float32))
    proc._process_slice(*proc._try_extract_slice())
    window = gen.windows[0]
    assert len(window) == SR * 8
    np.testing.assert_allclose(window[-SLICE_16K:], marker, atol=1e-6)
    np.testing.assert_allclose(window[:-SLICE_16K], 0.0)  # 初始滑窗为静音


def test_noise_gate_feeds_zeros_to_window_but_keeps_playback_audio():
    proc, gen = make_proc(noise_gate_rms=0.05)
    quiet16 = np.full(SLICE_16K, 0.01, np.float32)   # RMS=0.01 < 0.05
    quiet_out = np.full(SLICE_OUT, 0.01, np.float32)
    proc.add_audio(quiet16, quiet_out)
    proc._process_slice(*proc._try_extract_slice())
    np.testing.assert_allclose(gen.windows[0][-SLICE_16K:], 0.0)  # 管线吃到零
    _, seg = proc.get_pair()
    np.testing.assert_allclose(seg, 0.01)                          # 回放原声不动


def test_backlog_drops_oldest_chunk():
    proc, gen = make_proc(max_backlog_chunks=2)
    for _ in range(3):  # 入 3 chunk 不消费 → 第 3 次入队前应丢最旧 24 帧
        _feed_one_slice(proc)
    assert proc.stats["dropped_frames"] == SLICE_LEN
    assert proc._pairs.qsize() == 2 * SLICE_LEN
    frame, _ = proc.get_pair()
    assert frame[0, 0, 0] == 1  # chunk#0(值0)被丢，队头是 chunk#1


def test_worker_thread_end_to_end():
    proc, gen = make_proc()
    proc.start()
    try:
        proc.add_audio(np.ones(SLICE_16K, np.float32), np.ones(SLICE_OUT, np.float32))
        deadline = time.time() + 5
        pairs = []
        while len(pairs) < SLICE_LEN and time.time() < deadline:
            p = proc.get_pair()
            if p is not None:
                pairs.append(p)
            else:
                time.sleep(0.01)
        assert len(pairs) == SLICE_LEN
        assert proc.stats["chunks_done"] == 1
        assert proc.stats["last_chunk_ms"] >= 0.0
    finally:
        proc.stop()
    assert not proc._worker.is_alive()
