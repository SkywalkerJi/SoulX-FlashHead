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
import sys
import time

# 脚本位于 scripts/ 子目录,把仓库根目录加入 sys.path 以便 import flash_head
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="总超时秒数(首 chunk 含 torch.compile 预热可达数十秒)")
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

    slice_len = params["frame_num"] - params["motion_frames_num"]
    expected_chunks = (n_steps * step16) // proc.slice_samples_16k
    expected_frames = expected_chunks * slice_len

    frames, times = [], []
    last_chunks_done = 0
    fed = 0
    t_start = time.time()
    next_feed = time.monotonic()
    deadline = time.time() + args.timeout
    # 单循环贯穿喂入(按 feed_step 节拍模拟麦克风)与消费;
    # 退出条件用先验的 expected_frames(确定性),不依赖 stats(仅供展示的近似值)
    while len(frames) < expected_frames and time.time() < deadline:
        if fed < n_steps and time.monotonic() >= next_feed:
            proc.add_audio(a16[fed * step16:(fed + 1) * step16],
                           aout[fed * step_out:(fed + 1) * step_out])
            fed += 1
            next_feed += args.feed_step
        cd = proc.stats["chunks_done"]
        if cd > last_chunks_done:  # 每次迭代无条件检查;近似值仅用于日志,不参与控制流
            times.append(proc.stats["last_chunk_ms"])
            last_chunks_done = cd
            logger.info(
                f"chunk-{len(times) - 1}: {times[-1]:.0f}ms "
                f"({slice_len}帧, 预算 {slice_len / params['tgt_fps'] * 1000:.0f}ms)"
            )
        p = proc.get_pair()
        if p is not None:
            frames.append(p[0])
            continue
        time.sleep(0.005)
    proc.stop()

    if len(frames) < expected_frames:
        logger.error(
            f"超时退出: 仅收到 {len(frames)}/{expected_frames} 帧"
            f"(chunk {last_chunks_done}/{expected_chunks});"
            f"若首 chunk 预热超过 {args.timeout:.0f}s 请加大 --timeout 重试"
        )

    total = time.time() - t_start
    steady = times[2:] if len(times) > 3 else times
    if steady:
        logger.info(
            f"总帧数 {len(frames)}, 总耗时 {total:.1f}s, 稳态每chunk {np.mean(steady):.0f}ms "
            f"(实时预算 {slice_len / params['tgt_fps'] * 1000:.0f}ms), "
            f"有效FPS {slice_len / (np.mean(steady) / 1000):.1f}"
        )
    else:
        logger.warning("未采集到 chunk 耗时样本")

    os.makedirs("gradio_results", exist_ok=True)
    if frames:
        out = "gradio_results/feed_wav_preview.mp4"
        with imageio.get_writer(out, format="mp4", mode="I", fps=params["tgt_fps"], codec="h264") as w:
            for f in frames:
                w.append_data(f)
        logger.info(f"预览已存 {out}（无音轨，仅看口型节奏）")
    else:
        logger.warning("无帧可写,跳过预览 mp4")


if __name__ == "__main__":
    main()
