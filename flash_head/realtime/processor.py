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
