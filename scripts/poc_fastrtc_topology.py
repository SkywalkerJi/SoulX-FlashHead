"""PoC：验证 fastrtc「麦克风驱动 + 音视频下行」拓扑。

== 决策门结论（2026-07-07 RunPod 实测,fastrtc==0.0.34 + gradio==5.50.0）==
1. 「纯麦克风上行(不弹摄像头)」在原版 fastrtc 不可行,且 fork 不可用:
   - 前端编译产物 index-*.js 的 getUserMedia 构造为
     `i = {video: t ? {deviceId,...a} : a, audio: n}`,其中 `a = r || 默认HD约束`
     —— video 键永远存在,track_constraints 传 video:False 会因 falsy 被
     默认约束替换,摄像头权限必弹(标准1无法通过,任何约束变体都绕不开);
   - 服务端视频下行轨也只在收到客户端视频轨时创建(webrtc_connection_mixin),
     纯音频上行时 video_emit 不会被轮询——两端都硬依赖客户端视频;
   - OpenAvatarChat 的 gradio-webrtc fork(fastrtc-0.0.19.dev0)在两端都改了
     这个行为,但其 `from gradio import wasm_utils` 与 gradio 5.50 不兼容
     (wasm_utils 已移除),降级 gradio 会破坏现有面板 → 放弃 fork。
2. v1 决策: 接受摄像头权限(帧在 video_receive 丢弃),上行视频约束压到
   320x240@5fps 最小化带宽;「纯麦克风」列为已知限制,待 fastrtc 上游支持。
3. TURN: turn.fastrtc.org(HF_TOKEN 免费网关)权威 DNS 故障(1.1.1.1 DoH 验证
   SERVFAIL),不可用;改用自有 Cloudflare TURN key(TURN_KEY_ID +
   TURN_KEY_API_TOKEN,直连 rtc.live.cloudflare.com)。

通过标准（浏览器实测）:
  1. 页面能看到服务端生成的测试画面（帧计数走带 + 音量条），~25fps 平滑
  2. 能听到自己麦克风的回声（音频下行通了）
  （摄像头权限会弹出,属已知限制,授权即可,画面被服务端丢弃）

RunPod 运行:
  export TURN_KEY_ID=... TURN_KEY_API_TOKEN=...   # Cloudflare Realtime TURN
  python scripts/poc_fastrtc_topology.py
  浏览器开 https://<podID>-7860.proxy.runpod.net
"""
import asyncio
import os

import gradio as gr
import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler, WebRTC

SR_OUT = 48000
FPS = 25


# 无 TURN 凭证时回退 STUN:浏览器可发现 srflx 候选,部分 NAT 下可打洞直连
STUN_FALLBACK = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def get_rtc_config(ttl=360_000):
    """TURN 凭证优先级: 自有 Cloudflare key(直连 CF API) > HF_TOKEN(fastrtc
    网关,2026-07 故障中) > STUN-only 回退。ttl 用长值:配置在组件构造时静态
    求值一次,短 TTL 会导致服务启动一段时间后新连接全部失败。"""
    key_id = os.environ.get("TURN_KEY_ID")
    api_token = os.environ.get("TURN_KEY_API_TOKEN")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        from fastrtc import get_cloudflare_turn_credentials
        if key_id and api_token:
            print("[PoC] 使用自有 Cloudflare TURN key")
            return get_cloudflare_turn_credentials(
                turn_key_id=key_id, turn_key_api_token=api_token, ttl=ttl)
        if hf_token:
            print("[PoC] 使用 HF_TOKEN 经 fastrtc 网关获取 TURN 凭证")
            return get_cloudflare_turn_credentials(hf_token=hf_token, ttl=ttl)
    except Exception as e:
        print(f"[PoC] 获取 TURN 凭证失败({e}),回退 STUN-only")
        return STUN_FALLBACK
    print("[PoC] 无 TURN 凭证(TURN_KEY_ID/HF_TOKEN 均未设置),STUN-only")
    return STUN_FALLBACK


def build_webrtc_kwargs():
    """客户端与服务端 ICE 共用同一份 TURN 凭证(均出站连 TURN,一次获取)。
    上行视频无法关闭(见头部决策门结论),压到最低规格省带宽。"""
    import inspect
    rtc_config = get_rtc_config()
    kwargs = dict(
        mode="send-receive",
        modality="audio-video",
        rtc_configuration=rtc_config,
        track_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            # 摄像头必开(前端硬编码),压到最低规格,帧到服务端即丢弃
            "video": {"width": {"ideal": 320}, "height": {"ideal": 240},
                      "frameRate": {"ideal": 5}},
        },
    )
    params = inspect.signature(WebRTC.__init__).parameters
    if "server_rtc_configuration" in params:
        kwargs["server_rtc_configuration"] = rtc_config
    else:
        print("[PoC] 注意: 安装版 WebRTC 组件不支持 server_rtc_configuration 参数,"
              "服务端走 aiortc 默认 ICE")
    return kwargs


class EchoPatternHandler(AsyncAudioVideoStreamHandler):
    """麦克风回声 + 音量驱动的测试画面。"""

    def __init__(self):
        super().__init__(expected_layout="mono", output_sample_rate=SR_OUT, fps=FPS)
        self._audio_q: asyncio.Queue = asyncio.Queue(maxsize=50)  # ~1s@20ms帧,防回声延迟漂移
        self._rms = 0.0
        self._frame_id = 0

    def copy(self):
        return EchoPatternHandler()

    async def video_receive(self, frame):
        pass  # 纯麦克风上行拓扑,不消费客户端视频(抽象方法需实现)

    async def receive(self, frame):
        sr, arr = frame
        y = arr.astype(np.float32).reshape(-1) / 32768.0
        self._rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
        try:
            self._audio_q.put_nowait((sr, arr))
        except asyncio.QueueFull:
            try:
                self._audio_q.get_nowait()  # 丢最旧,保持回声贴近实时
            except asyncio.QueueEmpty:
                pass
            self._audio_q.put_nowait((sr, arr))

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


# 组件底部控制条会与 Gradio 页脚重叠导致按钮难点击 → 隐藏页脚+固定高度
_CSS = "footer {display: none !important} .webrtc-container {margin-bottom: 40px}"

with gr.Blocks(title="fastrtc 拓扑 PoC", css=_CSS) as app:
    gr.Markdown("# PoC：麦克风驱动 + 音视频下行\n摄像头权限会弹出(已知限制,授权即可,画面被丢弃)。")
    webrtc = WebRTC(height=600, **build_webrtc_kwargs())
    webrtc.stream(EchoPatternHandler(), inputs=[webrtc], outputs=[webrtc],
                  concurrency_limit=1, time_limit=600)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
