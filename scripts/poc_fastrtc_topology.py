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
