# 实时流式数字人面板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `gradio_app_realtime.py` 面板：浏览器麦克风流式输入 → FlashHead Lite 实时 chunk 推理 → WebRTC 音画同步下行（25fps），支持"自己看（静音）/给别人看（有声）"切换。

**Architecture:** 三层：① `flash_head/realtime/processor.py`——**不依赖 torch/gradio/fastrtc 的纯逻辑模块**（攒片、滑窗、配对队列、噪声门限、追赶丢帧），在无 GPU 开发机上 TDD；② `flash_head/realtime/binding.py`——把 processor 的 `generate_chunk` 回调绑定到现有推理管线（复用 `flash_head.inference`，管线零改动）；③ `gradio_app_realtime.py`——fastrtc `AsyncAudioVideoStreamHandler` + Gradio 面板，pacing 交给 fastrtc 库（视频按 fps=25、音频按 20ms 轮询）。

**Tech Stack:** gradio 5.50.0（已 pin，不动）、fastrtc（新增，仅实时面板用）、soxr（librosa 已带，用于重采样）、numpy、pytest（仅开发机测试）。

**设计文档:** `docs/superpowers/specs/2026-07-02-realtime-streaming-panel-design.md`

## Global Constraints

- 现有 `gradio_app.py` / `gradio_app_streaming.py` / `flash_head` 管线代码**一律不改**
- 新依赖只进 `requirements_realtime.txt`，不动 `requirements.txt`（`gradio==5.50.0` 保持）
- `flash_head/realtime/processor.py` **禁止 import torch / flash_head 其他模块 / gradio / fastrtc**（保证开发机可测）
- 严格实时目标仅 Lite 档；面板 Model Type 默认 `lite`，选 `pro` 显示警告
- 单会话：fastrtc `concurrency_limit=1`
- 帧色彩约定：processor 及 binding 输出 **RGB** uint8；仅在 handler 下行处转 BGR（是否需要转以 PoC 实测为准，见 Task 4）
- 关键数值（Lite）：slice_len=24、16k 每片 15360 样本（0.96s）、输出音频 48kHz、每帧配 1920 样本（40ms）、滑窗 8s=128000 样本
- 追赶策略：配对队列深度 ≥ 2 chunk（48 帧）时丢最旧 24 帧；噪声门限默认 0（关闭）
- **开发机无 GPU 无模型**：Task 1-2 全部可本地验证；Task 3-6 的 RunPod 步骤标注为 `【RunPod 检查点】`，执行到该处需在 pod 上操作（`bash /workspace/start.sh` 恢复环境）
- fastrtc 是 0.0.x（API 不稳定）：所有 fastrtc 接口签名**以 pod 上实际安装版本的源码为准**，计划中代码按 v0.0.34 调研结论编写，Task 4/5 含核对步骤

## File Structure

| 文件 | 职责 |
|---|---|
| `flash_head/realtime/__init__.py` | 空包标记 |
| `flash_head/realtime/processor.py` | 纯逻辑：攒片→滑窗→generate_chunk 回调→(帧,音频段)配对队列；工作线程；噪声门限；追赶丢帧；统计 |
| `flash_head/realtime/binding.py` | `make_generate_chunk(pipeline, infer_params)`：绑定 wav2vec 嵌入+管线生成（import torch，只在 RunPod 运行） |
| `tests/test_realtime_processor.py` | processor 单元测试（numpy-only，开发机运行） |
| `scripts/feed_wav_realtime.py` | 无浏览器集成验证：喂 wav 走真管线，测每 chunk 耗时/有效 FPS，存预览 mp4 |
| `scripts/poc_fastrtc_topology.py` | PoC：纯麦克风上行+音视频下行的最小 fastrtc 应用（决策门：原版 fastrtc vs gradio-webrtc fork） |
| `gradio_app_realtime.py` | 面板 + AvatarHandler + TURN 配置 + 静音开关 + 状态栏 |
| `requirements_realtime.txt` | fastrtc 等新增依赖 |
| `README.md` | 追加实时面板一节（简短） |

## Interfaces（跨任务契约，实现者必读）

```python
# Task 1+2 产出 —— flash_head/realtime/processor.py
class RealtimeProcessor:
    def __init__(
        self,
        generate_chunk,            # Callable[[np.ndarray], np.ndarray]
                                   #   入参: float32 滑窗音频, len = sample_rate*cached_audio_duration
                                   #   返回: uint8 RGB 帧数组 (slice_len, H, W, 3)
        slice_len: int = 24,
        sample_rate: int = 16000,
        tgt_fps: int = 25,
        cached_audio_duration: int = 8,
        output_sample_rate: int = 48000,
        noise_gate_rms: float = 0.0,
        max_backlog_chunks: int = 2,
    ): ...
    def start(self) -> None            # 启动工作线程
    def stop(self) -> None             # 停止并 join
    def add_audio(self, audio_16k: "np.float32[N]", audio_out: "np.float32[M]") -> None
    def get_pair(self):                # -> Optional[tuple[frame_rgb(H,W,3)uint8, audio_seg float32[1920]]]
    stats: dict  # {"last_chunk_ms": float, "queue_depth": int, "dropped_frames": int, "chunks_done": int}
    # 内部可测单元:
    def _try_extract_slice(self):      # -> Optional[tuple[slice16k, slice_out]]（攒够才返回，含配比消费+补零）
    def _process_slice(self, s16, sout) -> None  # 门限→滑窗→generate_chunk→配对入队(含追赶丢帧)

# Task 3 产出 —— flash_head/realtime/binding.py
def make_generate_chunk(pipeline, infer_params: dict):  # -> Callable，符合上面 generate_chunk 契约
```

---

### Task 1: RealtimeProcessor 攒片核心（TDD，开发机）

**Files:**
- Create: `flash_head/realtime/__init__.py`
- Create: `flash_head/realtime/processor.py`
- Test: `tests/test_realtime_processor.py`

**Interfaces:**
- Consumes: 无（纯 numpy）
- Produces: `RealtimeProcessor.__init__` / `add_audio` / `_try_extract_slice`（签名见上方契约表）

- [ ] **Step 1: 准备开发机测试环境**

```bash
cd /home/sj/Documents/github/SoulX-FlashHead
python3 -c "import numpy" || pip install --user numpy
python3 -c "import pytest" || pip install --user pytest
mkdir -p tests flash_head/realtime
touch flash_head/realtime/__init__.py tests/__init__.py
```

- [ ] **Step 2: 写失败测试（攒片边界）**

创建 `tests/test_realtime_processor.py`：

```python
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
```

- [ ] **Step 3: 运行确认失败**

```bash
python3 -m pytest tests/test_realtime_processor.py -v
```
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError: RealtimeProcessor`）

- [ ] **Step 4: 最小实现（构造器 + add_audio + _try_extract_slice）**

创建 `flash_head/realtime/processor.py`：

```python
"""实时流式处理器：攒片 → 滑窗 → chunk 推理回调 → (帧, 音频段) 配对队列。

设计约束：本文件不得 import torch / gradio / fastrtc / flash_head 其他模块，
保证在无 GPU 的开发机上可单元测试。GPU 侧逻辑经 generate_chunk 回调注入
（见 flash_head/realtime/binding.py）。
"""
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np


class RealtimeProcessor:
    def __init__(
        self,
        generate_chunk: Callable[[np.ndarray], np.ndarray],
        slice_len: int = 24,
        sample_rate: int = 16000,
        tgt_fps: int = 25,
        cached_audio_duration: int = 8,
        output_sample_rate: int = 48000,
        noise_gate_rms: float = 0.0,
        max_backlog_chunks: int = 2,
    ):
        self._generate_chunk = generate_chunk
        self.slice_len = slice_len
        self.sample_rate = sample_rate
        self.tgt_fps = tgt_fps
        self.output_sample_rate = output_sample_rate
        self.noise_gate_rms = noise_gate_rms
        self.max_backlog_chunks = max_backlog_chunks

        self.slice_samples_16k = slice_len * sample_rate // tgt_fps
        self.samples_per_frame_out = output_sample_rate // tgt_fps
        self.slice_samples_out = slice_len * self.samples_per_frame_out

        window_len = sample_rate * cached_audio_duration
        self._window = deque([0.0] * window_len, maxlen=window_len)

        self._pending_16k = np.zeros(0, dtype=np.float32)
        self._pending_out = np.zeros(0, dtype=np.float32)
        self._in_cond = threading.Condition()

        self._pairs: "queue.Queue[Tuple[np.ndarray, np.ndarray]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self.stats = {
            "last_chunk_ms": 0.0,
            "queue_depth": 0,
            "dropped_frames": 0,
            "chunks_done": 0,
        }

    # ---------- 输入侧 ----------
    def add_audio(self, audio_16k: np.ndarray, audio_out: np.ndarray) -> None:
        """追加一段麦克风音频。audio_16k 供管线推理；audio_out 为回放原声（output_sample_rate）。"""
        with self._in_cond:
            self._pending_16k = np.concatenate([self._pending_16k, audio_16k.astype(np.float32)])
            self._pending_out = np.concatenate([self._pending_out, audio_out.astype(np.float32)])
            self._in_cond.notify()

    def _try_extract_slice(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """攒够一个 chunk 的 16k 音频则弹出 (slice16k, slice_out)；不足返回 None。

        out 侧按名义长度消费；不足补零（麦克风重采样抖动），保证逐帧配对长度恒定。
        """
        with self._in_cond:
            if len(self._pending_16k) < self.slice_samples_16k:
                return None
            s16 = self._pending_16k[: self.slice_samples_16k]
            self._pending_16k = self._pending_16k[self.slice_samples_16k:]

            take = min(self.slice_samples_out, len(self._pending_out))
            sout = self._pending_out[:take]
            self._pending_out = self._pending_out[take:]
            if take < self.slice_samples_out:
                sout = np.concatenate(
                    [sout, np.zeros(self.slice_samples_out - take, dtype=np.float32)]
                )
            return s16, sout
```

- [ ] **Step 5: 运行确认通过**

```bash
python3 -m pytest tests/test_realtime_processor.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add flash_head/realtime/ tests/
git commit -m "feat(realtime): processor slicing core with paired audio consumption (TDD)"
```

---

### Task 2: RealtimeProcessor 推理配对 + 噪声门限 + 追赶 + 工作线程（TDD，开发机）

**Files:**
- Modify: `flash_head/realtime/processor.py`（追加方法）
- Test: `tests/test_realtime_processor.py`（追加用例）

**Interfaces:**
- Consumes: Task 1 的 `_try_extract_slice`
- Produces: `_process_slice(s16, sout)`、`get_pair()`、`start()`、`stop()`、`stats`（契约见顶部）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_realtime_processor.py` 末尾追加：

```python
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
```

（`import time` 已在 Task 1 的文件顶部 import 区，无需重复添加。）

- [ ] **Step 2: 运行确认失败**

```bash
python3 -m pytest tests/test_realtime_processor.py -v
```
Expected: 前 4 个 PASS，新增 5 个 FAIL（`AttributeError: _process_slice` 等）

- [ ] **Step 3: 实现 _process_slice / get_pair / start / stop**

在 `flash_head/realtime/processor.py` 的类内追加：

```python
    # ---------- 推理与配对 ----------
    def _process_slice(self, s16: np.ndarray, sout: np.ndarray) -> None:
        """一个 chunk：噪声门限 → 滑窗推进 → 推理 → 追赶丢帧 → 逐帧配对入队。"""
        if self.noise_gate_rms > 0.0:
            rms = float(np.sqrt(np.mean(np.square(s16))))
            if rms < self.noise_gate_rms:
                s16 = np.zeros_like(s16)

        self._window.extend(s16.tolist())
        window = np.array(self._window, dtype=np.float32)

        t0 = time.monotonic()
        frames = self._generate_chunk(window)  # (slice_len, H, W, 3) uint8 RGB
        self.stats["last_chunk_ms"] = (time.monotonic() - t0) * 1000.0
        self.stats["chunks_done"] += 1

        # 追赶：积压超过 max_backlog_chunks 个 chunk 时丢最旧一个 chunk
        if self._pairs.qsize() >= self.max_backlog_chunks * self.slice_len:
            for _ in range(self.slice_len):
                try:
                    self._pairs.get_nowait()
                    self.stats["dropped_frames"] += 1
                except queue.Empty:
                    break

        spf = self.samples_per_frame_out
        for i in range(frames.shape[0]):
            seg = sout[i * spf: (i + 1) * spf]
            if len(seg) < spf:
                seg = np.concatenate([seg, np.zeros(spf - len(seg), dtype=np.float32)])
            self._pairs.put((frames[i], seg))
        self.stats["queue_depth"] = self._pairs.qsize()

    # ---------- 输出侧 ----------
    def get_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            pair = self._pairs.get_nowait()
        except queue.Empty:
            self.stats["queue_depth"] = 0
            return None
        self.stats["queue_depth"] = self._pairs.qsize()
        return pair

    # ---------- 工作线程 ----------
    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="RealtimeProcessor",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._in_cond:
                if len(self._pending_16k) < self.slice_samples_16k:
                    self._in_cond.wait(timeout=0.1)
            got = self._try_extract_slice()
            if got is None:
                continue
            self._process_slice(*got)

    def stop(self) -> None:
        self._stop_event.set()
        with self._in_cond:
            self._in_cond.notify_all()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3)
```

- [ ] **Step 4: 运行确认全部通过**

```bash
python3 -m pytest tests/test_realtime_processor.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add flash_head/realtime/processor.py tests/test_realtime_processor.py
git commit -m "feat(realtime): paired queue, noise gate, backlog catch-up, worker thread (TDD)"
```

---

### Task 3: 管线绑定 + 无浏览器集成脚本（代码本地写，验证在 RunPod）

**Files:**
- Create: `flash_head/realtime/binding.py`
- Create: `scripts/feed_wav_realtime.py`

**Interfaces:**
- Consumes: `RealtimeProcessor`（Task 1-2）；`flash_head.inference` 的 `get_pipeline / get_base_data / get_infer_params / get_audio_embedding / run_pipeline`（现有代码，签名见 `flash_head/inference.py`）
- Produces: `make_generate_chunk(pipeline, infer_params) -> Callable[[np.ndarray], np.ndarray]`

- [ ] **Step 1: 写 binding**

创建 `flash_head/realtime/binding.py`：

```python
"""把 RealtimeProcessor 的 generate_chunk 回调绑定到 FlashHead 推理管线。

本文件 import torch，仅在 GPU 机器（RunPod）上运行。
复用 flash_head.inference 的现有函数，管线代码零改动。
"""
import numpy as np

from flash_head.inference import get_audio_embedding, run_pipeline


def make_generate_chunk(pipeline, infer_params: dict):
    """返回符合 RealtimeProcessor 契约的 generate_chunk 回调。

    入参 window_audio: float32, len = sample_rate * cached_audio_duration（8s 滑窗）
    返回: uint8 RGB (slice_len, H, W, 3)
    """
    tgt_fps = infer_params["tgt_fps"]
    cached = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    audio_end_idx = cached * tgt_fps            # 200
    audio_start_idx = audio_end_idx - frame_num  # 167 (lite)

    def generate_chunk(window_audio: np.ndarray) -> np.ndarray:
        emb = get_audio_embedding(pipeline, window_audio, audio_start_idx, audio_end_idx)
        frames = run_pipeline(pipeline, emb)          # torch float (T, H, W, 3), 0..255, RGB
        frames = frames[motion_frames_num:]
        return frames.cpu().numpy().astype(np.uint8)

    return generate_chunk
```

- [ ] **Step 2: 写集成脚本**

创建 `scripts/feed_wav_realtime.py`：

```python
"""无浏览器集成验证：喂 wav 文件走 RealtimeProcessor + 真实管线。

用途：隔离 WebRTC 变量，单独验证 processor 与管线的对接、每 chunk 耗时、
有效 FPS，并保存预览 mp4 供肉眼检查口型。

用法（RunPod）:
  python scripts/feed_wav_realtime.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite --cond_image examples/girl.png \
    --audio_path examples/podcast_sichuan_16k.wav --max_seconds 20
"""
import argparse
import os
import time

import imageio
import librosa
import numpy as np
from loguru import logger

from flash_head.inference import get_pipeline, get_base_data, get_infer_params
from flash_head.realtime.binding import make_generate_chunk
from flash_head.realtime.processor import RealtimeProcessor

OUT_SR = 48000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--wav2vec_dir", required=True)
    ap.add_argument("--model_type", default="lite", choices=["lite", "pro"])
    ap.add_argument("--cond_image", required=True)
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--feed_step", type=float, default=0.5, help="模拟麦克风每次送入的秒数")
    args = ap.parse_args()

    pipeline = get_pipeline(1, args.ckpt_dir, args.model_type, args.wav2vec_dir)
    get_base_data(pipeline, args.cond_image, 9999, use_face_crop=False)
    params = get_infer_params()

    proc = RealtimeProcessor(
        generate_chunk=make_generate_chunk(pipeline, params),
        slice_len=params["frame_num"] - params["motion_frames_num"],
        sample_rate=params["sample_rate"],
        tgt_fps=params["tgt_fps"],
        cached_audio_duration=params["cached_audio_duration"],
        output_sample_rate=OUT_SR,
    )
    proc.start()

    a16, _ = librosa.load(args.audio_path, sr=params["sample_rate"], mono=True)
    a16 = a16[: int(args.max_seconds * params["sample_rate"])].astype(np.float32)
    aout, _ = librosa.load(args.audio_path, sr=OUT_SR, mono=True)
    aout = aout[: int(args.max_seconds * OUT_SR)].astype(np.float32)

    step16 = int(args.feed_step * params["sample_rate"])
    step_out = int(args.feed_step * OUT_SR)
    n_steps = len(a16) // step16

    frames, times = [], []
    t_start = time.time()
    for i in range(n_steps):
        proc.add_audio(a16[i * step16:(i + 1) * step16], aout[i * step_out:(i + 1) * step_out])
        while True:
            p = proc.get_pair()
            if p is None:
                break
            frames.append(p[0])
        if proc.stats["chunks_done"] > len(times):
            times.append(proc.stats["last_chunk_ms"])
            logger.info(
                f"chunk-{len(times)-1}: {times[-1]:.0f}ms "
                f"({params['frame_num'] - params['motion_frames_num']}帧, "
                f"预算 {(params['frame_num'] - params['motion_frames_num']) / params['tgt_fps'] * 1000:.0f}ms)"
            )
    # 排空
    deadline = time.time() + 30
    while time.time() < deadline:
        p = proc.get_pair()
        if p is None:
            if proc.stats["chunks_done"] * (params["frame_num"] - params["motion_frames_num"]) <= len(frames):
                break
            time.sleep(0.05)
            continue
        frames.append(p[0])
    proc.stop()

    total = time.time() - t_start
    steady = times[2:] if len(times) > 3 else times  # 丢 compile 预热
    slice_len = params["frame_num"] - params["motion_frames_num"]
    logger.info(f"总帧数 {len(frames)}, 总耗时 {total:.1f}s, 稳态每chunk {np.mean(steady):.0f}ms "
                f"(实时预算 {slice_len / params['tgt_fps'] * 1000:.0f}ms), "
                f"有效FPS {slice_len / (np.mean(steady) / 1000):.1f}")

    os.makedirs("gradio_results", exist_ok=True)
    out = "gradio_results/feed_wav_preview.mp4"
    with imageio.get_writer(out, format="mp4", mode="I", fps=params["tgt_fps"], codec="h264") as w:
        for f in frames:
            w.append_data(f)
    logger.info(f"预览已存 {out}（无音轨，仅看口型节奏）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 本地静态检查（无 GPU，仅语法/导入结构）**

```bash
python3 -m py_compile flash_head/realtime/binding.py scripts/feed_wav_realtime.py && echo OK
```
Expected: OK（不实际 import torch）

- [ ] **Step 4: Commit**

```bash
git add flash_head/realtime/binding.py scripts/feed_wav_realtime.py
git commit -m "feat(realtime): pipeline binding and offline wav feed script"
```

- [ ] **Step 5:【RunPod 检查点】跑集成脚本**

在 pod 上（`bash /workspace/start.sh` 恢复环境后，git pull 本分支）：

```bash
cd /workspace/SoulX-FlashHead && source /workspace/venv/bin/activate
python scripts/feed_wav_realtime.py \
  --ckpt_dir models/SoulX-FlashHead-1_3B --wav2vec_dir models/wav2vec2-base-960h \
  --model_type lite --cond_image examples/girl.png \
  --audio_path examples/podcast_sichuan_16k.wav --max_seconds 20
```
Expected: 稳态每 chunk 耗时 < 960ms（实时预算内）；`gradio_results/feed_wav_preview.mp4` 口型正常。
若首 chunk 数十秒属 torch.compile 预热，正常。记录稳态数值供 Task 6 汇报。

---

### Task 4: PoC — fastrtc 纯麦克风上行拓扑（决策门）

**Files:**
- Create: `scripts/poc_fastrtc_topology.py`
- Create: `requirements_realtime.txt`

**Interfaces:**
- Consumes: 无（独立 PoC，不碰模型）
- Produces: 决策结论（原版 fastrtc / gradio-webrtc fork），写入脚本头部注释与 commit message；Task 5 据此选依赖与 import

**背景（调研已证实）：** 原版 fastrtc `modality="audio-video"` 的服务端视频下行轨仅在 `@pc.on("track")` 收到**客户端视频轨**时创建，默认 `track_constraints` 会同时申请摄像头+麦克风。本 PoC 验证能否用 `track_constraints={"video": False, ...}` 或等价手段实现"只弹麦克风权限，仍有视频下行"。若不行，退路是 OpenAvatarChat 维护的 fork（`gradio-webrtc`，avatar 场景专用）。

- [ ] **Step 1: 写 requirements_realtime.txt**

```
# 实时流式面板专用依赖（gradio_app_realtime.py / scripts/poc_fastrtc_topology.py）
# 主 requirements.txt 保持不变
fastrtc==0.0.34
```

- [ ] **Step 2: 写 PoC 脚本**

创建 `scripts/poc_fastrtc_topology.py`：

```python
"""PoC：验证 fastrtc 能否做「纯麦克风上行 + 音视频下行」拓扑。

通过标准（浏览器实测）:
  1. 点击连接时只弹麦克风权限（不弹摄像头）
  2. 页面能看到服务端生成的测试画面（帧计数 + 音量条），~25fps 平滑
  3. 能听到自己麦克风的回声（音频下行通了）
决策门:
  - 全部通过 → Task 5 用原版 fastrtc
  - 视频下行不出现 → 依次尝试: a) track_constraints 变体
    b) pip install gradio-webrtc（HumanAIGC-Engineering fork）改 import 重试
    把结论写进本文件头部注释并提交。

RunPod 运行:
  HF_TOKEN=hf_xxx python scripts/poc_fastrtc_topology.py
  浏览器开 https://<podID>-7860.proxy.runpod.net
"""
import asyncio
import os
import time

import gradio as gr
import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler, WebRTC

SR_OUT = 48000
FPS = 25


def get_rtc_config():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    from fastrtc import get_cloudflare_turn_credentials
    return get_cloudflare_turn_credentials(hf_token=token)


def build_webrtc_kwargs():
    """构造 WebRTC 组件参数。设计要求客户端与服务端 ICE 都要配 TURN：
    rtc_configuration=客户端浏览器；server_rtc_configuration=pod 侧 aiortc
    （若安装版本的 WebRTC 组件不支持该参数则跳过并记录——服务端出站 UDP
    通常可直连 TURN，届时在 PoC 结论中注明实测连通性）。"""
    import inspect
    kwargs = dict(
        mode="send-receive",
        modality="audio-video",
        rtc_configuration=get_rtc_config(),
        track_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": False,
        },
    )
    params = inspect.signature(WebRTC.__init__).parameters
    token = os.environ.get("HF_TOKEN")
    if token and "server_rtc_configuration" in params:
        from fastrtc import get_cloudflare_turn_credentials
        kwargs["server_rtc_configuration"] = get_cloudflare_turn_credentials(
            hf_token=token, ttl=360_000,  # 服务端凭证要长 TTL
        )
    return kwargs


class EchoPatternHandler(AsyncAudioVideoStreamHandler):
    """麦克风回声 + 音量驱动的测试画面。"""

    def __init__(self):
        super().__init__(expected_layout="mono", output_sample_rate=SR_OUT, fps=FPS)
        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._rms = 0.0
        self._frame_id = 0

    def copy(self):
        return EchoPatternHandler()

    async def receive(self, frame):
        sr, arr = frame
        y = arr.astype(np.float32).reshape(-1) / 32768.0
        self._rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
        await self._audio_q.put((sr, arr))

    async def video_emit(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        bar = int(min(self._rms * 20.0, 1.0) * 500)
        img[500 - bar:500, 100:412, 1] = 255                      # 音量条
        img[10:30, 10:10 + (self._frame_id % 492), 2] = 255      # 帧计数走带
        self._frame_id += 1
        return img

    async def emit(self):
        try:
            sr, arr = self._audio_q.get_nowait()
            return (sr, arr)
        except asyncio.QueueEmpty:
            return (SR_OUT, np.zeros((1, SR_OUT // 50), dtype=np.int16))

    async def shutdown(self):
        pass


with gr.Blocks(title="fastrtc 拓扑 PoC") as app:
    gr.Markdown("# PoC：纯麦克风上行 + 音视频下行\n点击连接后应只弹麦克风权限。")
    webrtc = WebRTC(**build_webrtc_kwargs())
    webrtc.stream(EchoPatternHandler(), inputs=[webrtc], outputs=[webrtc],
                  concurrency_limit=1, time_limit=600)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 3: 本地语法检查**

```bash
python3 -m py_compile scripts/poc_fastrtc_topology.py && echo OK
```
Expected: OK

- [ ] **Step 4: Commit（代码部分）**

```bash
git add scripts/poc_fastrtc_topology.py requirements_realtime.txt
git commit -m "feat(realtime): fastrtc mic-only topology PoC script"
```

- [ ] **Step 5:【RunPod 检查点】执行 PoC 并记录决策**

pod 上：

```bash
source /workspace/venv/bin/activate && pip install -r requirements_realtime.txt
# 先核对安装版本的真实签名（0.0.x API 不稳定,以此为准修正脚本）:
python -c "import inspect; from fastrtc import AsyncAudioVideoStreamHandler as H; print(inspect.signature(H.__init__))"
python -c "import inspect; from fastrtc import WebRTC; print(inspect.signature(WebRTC.__init__))"
HF_TOKEN=hf_xxx python scripts/poc_fastrtc_topology.py
```

浏览器打开 `https://<podID>-7860.proxy.runpod.net`，按脚本头部"通过标准"逐条验证。
把结论（用原版还是 fork、有效的 track_constraints 写法、实际 handler 签名差异）**更新进脚本头部注释**并提交：

```bash
git add scripts/poc_fastrtc_topology.py requirements_realtime.txt
git commit -m "docs(realtime): record PoC decision - <stock fastrtc | gradio-webrtc fork>"
```

---

### Task 5: 实时面板 gradio_app_realtime.py

**Files:**
- Create: `gradio_app_realtime.py`

**Interfaces:**
- Consumes: `RealtimeProcessor`（Task 1-2）、`make_generate_chunk`（Task 3）、`flash_head.inference.get_pipeline/get_base_data/get_infer_params`（现有）、Task 4 的 PoC 决策（import 与 track_constraints 写法）
- Produces: 最终面板（用户入口）

**注意：** 若 Task 4 决策为 fork，把 `from fastrtc import ...` 换成 `from gradio_webrtc import ...` 并同步 `requirements_realtime.txt`；`AsyncAudioVideoStreamHandler` 构造参数以 PoC 核对的真实签名为准。

- [ ] **Step 1: 写面板**

创建 `gradio_app_realtime.py`：

```python
"""SoulX-FlashHead 实时流式数字人面板：麦克风直驱，WebRTC 音画同步下行。

延迟预期 ~1.5-2.5s（0.96s 攒片 + 生成 + 传输），严格实时请用 lite 档。
RunPod 部署需 export HF_TOKEN=...（Cloudflare TURN 中继，免费 10GB/月）。
"""
import asyncio
import os
import threading

import gradio as gr
import numpy as np
import soxr
from loguru import logger
from fastrtc import AsyncAudioVideoStreamHandler, WebRTC

from flash_head.inference import get_pipeline, get_base_data, get_infer_params
from flash_head.realtime.binding import make_generate_chunk
from flash_head.realtime.processor import RealtimeProcessor

OUT_SR = 48000
FPS = 25
RESAMPLE_BUF_SEC = 0.3   # 攒 0.3s 再重采样，摊薄调用开销
AUDIO_BUF_CAP = OUT_SR   # 下行音频缓冲上限 1s，防单侧漂移


class AppState:
    pipeline = None
    processor: RealtimeProcessor = None
    idle_frame: np.ndarray = None      # BGR
    muted: bool = True                 # 默认"自己看"（静音）
    loaded_key = None


STATE = AppState()


def load_model(ckpt_dir, wav2vec_dir, model_type, cond_image, seed, use_face_crop, noise_gate):
    if cond_image is None:
        raise gr.Error("请先选择条件图片")
    key = (ckpt_dir, wav2vec_dir, model_type)
    if STATE.pipeline is None or STATE.loaded_key != key:
        logger.info(f"Loading pipeline: {key}")
        STATE.pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
        STATE.loaded_key = key
    get_base_data(STATE.pipeline, cond_image, int(seed) if seed >= 0 else 9999, use_face_crop)
    params = get_infer_params()

    if STATE.processor is not None:
        STATE.processor.stop()
    STATE.processor = RealtimeProcessor(
        generate_chunk=make_generate_chunk(STATE.pipeline, params),
        slice_len=params["frame_num"] - params["motion_frames_num"],
        sample_rate=params["sample_rate"],
        tgt_fps=params["tgt_fps"],
        cached_audio_duration=params["cached_audio_duration"],
        output_sample_rate=OUT_SR,
        noise_gate_rms=float(noise_gate),
    )
    STATE.processor.start()

    ref = STATE.pipeline.original_color_reference  # (1,C,1,H,W), [-1,1]
    rgb = ((ref[0, :, 0].permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    STATE.idle_frame = rgb[:, :, ::-1].copy()

    warn = "  ⚠️ Pro 档单卡通常达不到 25FPS，会累积延迟" if model_type == "pro" else ""
    return f"✅ 模型已加载（{model_type}），可点击下方连接开始{warn}"


def set_muted(play_audio: bool):
    STATE.muted = not play_audio
    return f"🔊 播放声音：{'开（给别人看）' if play_audio else '关（自己看，防回声）'}"


def get_rtc_config():
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("未设置 HF_TOKEN：仅本地/局域网可连通；RunPod 需 TURN")
        return None
    from fastrtc import get_cloudflare_turn_credentials
    return get_cloudflare_turn_credentials(hf_token=token)


def build_webrtc_kwargs():
    """客户端与服务端 ICE 都配 TURN（设计要求）。track_constraints 写法以 Task 4 PoC 结论为准。"""
    import inspect
    kwargs = dict(
        mode="send-receive",
        modality="audio-video",
        rtc_configuration=get_rtc_config(),
        track_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": False,
        },
    )
    params = inspect.signature(WebRTC.__init__).parameters
    token = os.environ.get("HF_TOKEN")
    if token and "server_rtc_configuration" in params:
        from fastrtc import get_cloudflare_turn_credentials
        kwargs["server_rtc_configuration"] = get_cloudflare_turn_credentials(
            hf_token=token, ttl=360_000,
        )
    return kwargs


def format_stats():
    p = STATE.processor
    if p is None:
        return "⏳ 模型未加载"
    s = p.stats
    slice_len = p.slice_len
    budget_ms = slice_len / p.tgt_fps * 1000
    fps_eff = slice_len / (s["last_chunk_ms"] / 1000) if s["last_chunk_ms"] > 0 else 0
    warn = " ⚠️ 生成速度不足，正在丢帧追赶" if s["dropped_frames"] > 0 else ""
    return (
        f"chunk 耗时 **{s['last_chunk_ms']:.0f}ms** / 预算 {budget_ms:.0f}ms ｜ "
        f"有效 **{fps_eff:.1f} FPS** ｜ 队列 {s['queue_depth']} 帧 ｜ "
        f"已丢弃 {s['dropped_frames']} 帧{warn}"
    )


class AvatarHandler(AsyncAudioVideoStreamHandler):
    def __init__(self):
        super().__init__(expected_layout="mono", output_sample_rate=OUT_SR, fps=FPS)
        self._recv_pending = np.zeros(0, dtype=np.float32)
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._buf_lock = threading.Lock()

    def copy(self):
        return AvatarHandler()

    async def receive(self, frame):
        sr, arr = frame
        y = arr.astype(np.float32).reshape(-1) / 32768.0
        self._recv_pending = np.concatenate([self._recv_pending, y])
        if len(self._recv_pending) < int(sr * RESAMPLE_BUF_SEC):
            return
        seg, self._recv_pending = self._recv_pending, np.zeros(0, dtype=np.float32)
        a16 = soxr.resample(seg, sr, 16000).astype(np.float32) if sr != 16000 else seg
        aout = soxr.resample(seg, sr, OUT_SR).astype(np.float32) if sr != OUT_SR else seg
        if STATE.processor is not None:
            STATE.processor.add_audio(a16, aout)

    async def video_emit(self):
        pair = STATE.processor.get_pair() if STATE.processor else None
        if pair is None:
            silence = np.zeros(OUT_SR // FPS, dtype=np.float32)
            with self._buf_lock:
                self._audio_buf = np.concatenate([self._audio_buf, silence])[-AUDIO_BUF_CAP:]
            if STATE.idle_frame is not None:
                return STATE.idle_frame
            return np.zeros((512, 512, 3), dtype=np.uint8)
        frame_rgb, audio_seg = pair
        with self._buf_lock:
            self._audio_buf = np.concatenate([self._audio_buf, audio_seg])[-AUDIO_BUF_CAP:]
        return frame_rgb[:, :, ::-1].copy()  # RGB -> BGR（以 PoC 实测色彩为准）

    async def emit(self):
        n = OUT_SR // 50  # 20ms
        with self._buf_lock:
            if len(self._audio_buf) >= n:
                seg, self._audio_buf = self._audio_buf[:n], self._audio_buf[n:]
            else:
                seg = np.zeros(n, dtype=np.float32)
        if STATE.muted:
            seg = np.zeros(n, dtype=np.float32)
        pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        return (OUT_SR, pcm)

    async def shutdown(self):
        pass


with gr.Blocks(title="SoulX-FlashHead 实时数字人", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ SoulX-FlashHead 实时数字人（麦克风直驱）")
    gr.Markdown("加载模型 → 点击连接授权麦克风 → 开口说话，~2 秒后画面开始对口型。")

    with gr.Row():
        with gr.Column(scale=1):
            cond_image_input = gr.Image(label="Condition Image", type="filepath",
                                        value="examples/girl.png", height=300)
            load_btn = gr.Button("📦 加载模型", variant="primary")
            load_status = gr.Markdown("⏳ 模型未加载")
            with gr.Accordion("⚙️ 高级设置", open=False):
                ckpt_dir_input = gr.Textbox(label="FlashHead Checkpoint Directory",
                                            value="models/SoulX-FlashHead-1_3B")
                wav2vec_dir_input = gr.Textbox(label="Wav2Vec Directory",
                                               value="models/wav2vec2-base-960h")
                model_type_input = gr.Dropdown(label="Model Type（实时请用 lite）",
                                               choices=["lite", "pro"], value="lite")
                use_face_crop_input = gr.Checkbox(label="Use Face Crop", value=False)
                seed_input = gr.Number(label="Random Seed", value=9999, precision=0)
                noise_gate_input = gr.Slider(label="噪声门限 RMS（0=关闭）",
                                             minimum=0.0, maximum=0.1, step=0.005, value=0.0)
        with gr.Column(scale=1):
            webrtc = WebRTC(**build_webrtc_kwargs())
            play_audio_input = gr.Checkbox(label="播放声音（给别人看时打开）", value=False)
            mute_status = gr.Markdown("🔊 播放声音：关（自己看，防回声）")
            stats_md = gr.Markdown("")
            timer = gr.Timer(1.0)

    load_btn.click(
        fn=load_model,
        inputs=[ckpt_dir_input, wav2vec_dir_input, model_type_input,
                cond_image_input, seed_input, use_face_crop_input, noise_gate_input],
        outputs=[load_status],
    )
    play_audio_input.change(fn=set_muted, inputs=[play_audio_input], outputs=[mute_status])
    timer.tick(fn=format_stats, outputs=[stats_md])
    webrtc.stream(AvatarHandler(), inputs=[webrtc], outputs=[webrtc],
                  concurrency_limit=1, time_limit=3600)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: 本地语法检查**

```bash
python3 -m py_compile gradio_app_realtime.py && echo OK
```
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add gradio_app_realtime.py
git commit -m "feat(realtime): realtime avatar panel with mic-driven WebRTC streaming"
```

---

### Task 6: RunPod 端到端验证 + 文档

**Files:**
- Modify: `README.md`（追加实时面板小节）
- Modify: `docs/superpowers/plans/2026-07-02-realtime-streaming-panel.md`（勾选+记录实测数据）

**Interfaces:**
- Consumes: 全部前置任务
- Produces: 验证记录 + 用户文档

- [ ] **Step 1:【RunPod 检查点】端到端验证清单**

pod 上：

```bash
cd /workspace/SoulX-FlashHead && source /workspace/venv/bin/activate && git pull
export HF_TOKEN=hf_xxx
python gradio_app_realtime.py
```

浏览器开 `https://<podID>-7860.proxy.runpod.net`，逐项验证并记录：

| 检查项 | 通过标准 |
|---|---|
| 加载模型（lite） | 状态栏显示已加载 |
| 连接 | 只弹麦克风权限；连接成功 |
| 说话→画面 | 口型跟随，秒表实测玻璃到玻璃延迟（预期 1.5~2.5s） |
| 状态栏 | chunk 耗时 < 960ms，有效 FPS ≥ 25 |
| 播放声音开关 | 开后能听到自己声音且与口型同步；关后静音 |
| 安静待机 | 不说话时画面自然微动，无嘴部乱动（必要时调噪声门限） |
| 长会话 | 连续 30 分钟，观察音画漂移与队列/丢帧计数 |
| Pro 警告 | 切 pro 档加载出现警告；说话时状态栏出现丢帧提示（预期行为） |

- [ ] **Step 2: README 追加小节**

在 `README.md` 的 Gradio 相关部分之后追加：

````markdown
### Real-time Microphone-driven Panel (WebRTC)

Speak into your microphone and the avatar lip-syncs in real time (~2s glass-to-glass latency, lite model).

```bash
pip install -r requirements_realtime.txt
# Behind an HTTPS-only proxy (e.g. RunPod), a TURN relay is required:
export HF_TOKEN=your_hf_token   # free 10GB/month Cloudflare TURN via fastrtc
python gradio_app_realtime.py
```
````

- [ ] **Step 3: 更新计划勾选与实测数据，提交**

```bash
git add README.md docs/superpowers/plans/2026-07-02-realtime-streaming-panel.md
git commit -m "docs(realtime): README section and E2E validation record"
```

---

## 验证与回归

- 开发机：`python3 -m pytest tests/ -v` 全绿（9 用例）
- RunPod：`scripts/feed_wav_realtime.py` 稳态 chunk < 960ms；面板 E2E 清单全过
- 回归确认：`gradio_app.py` / `gradio_app_streaming.py` 未被改动（`git diff --stat main` 仅新增文件 + README）
