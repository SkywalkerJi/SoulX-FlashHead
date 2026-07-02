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
