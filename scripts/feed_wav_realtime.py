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
