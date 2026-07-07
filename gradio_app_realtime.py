"""SoulX-FlashHead 实时流式数字人面板：麦克风直驱，WebRTC 音画同步下行。

延迟预期 ~1.5-2.5s（0.96s 攒片 + 生成 + 传输），严格实时请用 lite 档。
RunPod 部署需 export HF_TOKEN=...（Cloudflare TURN 中继，免费 10GB/月）。
"""
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
    # 先停旧 processor(join 推理线程),再动共享 pipeline 状态,避免与在飞 chunk 竞争
    if STATE.processor is not None:
        STATE.processor.stop()
        STATE.processor = None
    key = (ckpt_dir, wav2vec_dir, model_type)
    if STATE.pipeline is None or STATE.loaded_key != key:
        logger.info(f"Loading pipeline: {key}")
        STATE.pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
        STATE.loaded_key = key
    get_base_data(STATE.pipeline, cond_image, int(seed) if seed >= 0 else 9999, use_face_crop)
    params = get_infer_params()

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


# 无 TURN 时的 STUN-only 回退:浏览器可发现 srflx 候选,配合服务端出站 UDP
# 打洞在部分 NAT 下可连通(2026-07 实测 turn.fastrtc.org 免费网关 DNS 故障)
STUN_FALLBACK = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def get_rtc_config(ttl=86_400):
    """TURN 凭证优先级: 自有 Cloudflare key(TURN_KEY_ID/TURN_KEY_API_TOKEN,
    直连 CF API) > HF_TOKEN(fastrtc 免费网关,2026-07 其 DNS 故障中) > STUN-only。
    ttl=24h:配置在组件构造时静态求值一次,太短会导致服务启动一段时间后新连接
    失败;Cloudflare 上限 48h,超过返回 400(已实测)。服务连续运行超过 ttl 需重启。"""
    key_id = os.environ.get("TURN_KEY_ID")
    api_token = os.environ.get("TURN_KEY_API_TOKEN")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        from fastrtc import get_cloudflare_turn_credentials
        if key_id and api_token:
            logger.info("使用自有 Cloudflare TURN key")
            return get_cloudflare_turn_credentials(
                turn_key_id=key_id, turn_key_api_token=api_token, ttl=ttl)
        if hf_token:
            logger.info("使用 HF_TOKEN 经 fastrtc 网关获取 TURN 凭证")
            return get_cloudflare_turn_credentials(hf_token=hf_token, ttl=ttl)
    except Exception as e:
        logger.warning(f"获取 TURN 凭证失败({e})，回退 STUN-only（跨公网 NAT 不保证连通）")
        return STUN_FALLBACK
    logger.warning("无 TURN 凭证（TURN_KEY_ID/HF_TOKEN 均未设置）：STUN-only，RunPod 上建议配 TURN")
    return STUN_FALLBACK


def build_webrtc_kwargs():
    """客户端与服务端 ICE 共用同一份 TURN 凭证（一次获取）。
    上行视频无法关闭（fastrtc 0.0.34 前端硬编码，见 PoC 决策记录），
    压到最低规格省带宽，帧在 video_receive 丢弃。"""
    import inspect
    rtc_config = get_rtc_config()
    kwargs = dict(
        mode="send-receive",
        modality="audio-video",
        rtc_configuration=rtc_config,
        track_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": {"width": {"ideal": 320}, "height": {"ideal": 240},
                      "frameRate": {"ideal": 5}},
        },
    )
    params = inspect.signature(WebRTC.__init__).parameters
    if "server_rtc_configuration" in params:
        kwargs["server_rtc_configuration"] = rtc_config
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
        self._recv_sr = None

    async def video_receive(self, frame):
        pass  # 纯麦克风上行,不消费客户端视频(基类抽象方法需实现)

    def copy(self):
        # 新连接: 排空旧会话积压的配对帧,避免重连后先播过期口型
        p = STATE.processor
        if p is not None:
            while p.get_pair() is not None:
                pass
        return AvatarHandler()

    async def receive(self, frame):
        sr, arr = frame
        if self._recv_sr is not None and sr != self._recv_sr and len(self._recv_pending) > 0:
            logger.warning(
                f"上行音频采样率变化 {self._recv_sr} -> {sr}, 丢弃 {len(self._recv_pending)} 个未处理样本"
            )
            self._recv_pending = np.zeros(0, dtype=np.float32)
        self._recv_sr = sr
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


# 组件底部控制条会与 Gradio 页脚重叠导致按钮难点击 → 隐藏页脚
_CSS = "footer {display: none !important}"

with gr.Blocks(title="SoulX-FlashHead 实时数字人", theme=gr.themes.Soft(), css=_CSS) as app:
    gr.Markdown("# 🎙️ SoulX-FlashHead 实时数字人（麦克风直驱）")
    gr.Markdown("加载模型 → 点击连接授权麦克风与摄像头（摄像头画面会被丢弃，属组件限制）→ 开口说话，~2 秒后画面开始对口型。")

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
            webrtc = WebRTC(height=600, **build_webrtc_kwargs())
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
