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
    """实时流式处理器，管理音频攒片、滑窗、chunk 推理与帧音频配对。

    线程契约：
    - add_audio() 是唯一的生产者入口，可从任意单一线程调用；
    - _try_extract_slice() / _process_slice() 只能由 worker 线程（start() 启动）串行调用，
      外部代码不得在 worker 运行期间直接调用二者，否则 _window 与 stats 会产生竞态；
    - get_pair() 供单一消费者线程调用；
    - stats 为无锁写入的近似值，仅供状态展示，不可用于精确控制逻辑。
    """
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

        # 无锁写入的近似统计，仅供 UI 展示（见类 docstring 线程契约）
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

    # ---------- 推理与配对 ----------
    def _process_slice(self, s16: np.ndarray, sout: np.ndarray) -> None:
        """一个 chunk：噪声门限 → 滑窗推进 → 推理 → 追赶丢帧 → 逐帧配对入队。

        注意：仅允许 worker 线程串行调用（见类 docstring 线程契约）。"""
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
            if self._worker.is_alive():
                import logging
                logging.getLogger(__name__).warning(
                    "RealtimeProcessor worker 在 3s 内未退出(可能正在推理长 chunk),"
                    "其仍可能短暂持有 pipeline——避免在此期间变更 pipeline 状态"
                )
