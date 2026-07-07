"""PoC：验证 fastrtc 能否做「纯麦克风上行 + 音视频下行」拓扑。

通过标准（浏览器实测）:
  1. 点击连接时只弹麦克风权限（不弹摄像头）
  2. 页面能看到服务端生成的测试画面（帧计数 + 音量条），~25fps 平滑
  3. 能听到自己麦克风的回声（音频下行通了）
决策门:
  - 三条标准全过 → Task 5 用原版 fastrtc
  - 标准1失败(仍弹摄像头权限) → 依次尝试 track_constraints 变体
    ({"video": False} / 省略 video 键 / {"audio": {...}});全部无效 →
    pip install gradio-webrtc(HumanAIGC-Engineering fork)改 import 重试
  - 标准2失败(视频下行不出现) → 同上,先 track_constraints 变体再 fork
  - 标准3失败(听不到回声) → 用 inspect 核对安装版 emit() 的返回格式契约
    (采样率/形状/dtype),修正后重试;仍失败再考虑 fork
  把最终结论(用哪个包、有效的 track_constraints 写法、实测签名差异)
  写进本文件头部注释并提交。

RunPod 运行:
  HF_TOKEN=hf_xxx python scripts/poc_fastrtc_topology.py
  浏览器开 https://<podID>-7860.proxy.runpod.net
"""
import asyncio
import os

import gradio as gr
import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler, WebRTC

SR_OUT = 48000
FPS = 25


# 无 TURN 时回退 STUN:浏览器至少能发现 srflx 候选,配合服务端出站 UDP
# 打洞在部分 NAT 下可直连(2026-07 实测 turn.fastrtc.org 权威 DNS 故障,
# 免费 TURN 网关不可用,STUN-only 是第一后备)
STUN_FALLBACK = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def get_rtc_config():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[PoC] 无 HF_TOKEN,使用 STUN-only 配置(无 TURN 中继)")
        return STUN_FALLBACK
    try:
        from fastrtc import get_cloudflare_turn_credentials
        return get_cloudflare_turn_credentials(hf_token=token)
    except Exception as e:
        print(f"[PoC] 获取 TURN 凭证失败({e}),回退 STUN-only")
        return STUN_FALLBACK


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
        try:
            from fastrtc import get_cloudflare_turn_credentials
            kwargs["server_rtc_configuration"] = get_cloudflare_turn_credentials(
                hf_token=token, ttl=360_000,  # 服务端凭证要长 TTL
            )
        except Exception as e:
            print(f"[PoC] 获取服务端 TURN 凭证失败({e}),服务端走默认 ICE(Google STUN)")
    elif token:
        print("[PoC] 注意: 安装版 WebRTC 组件不支持 server_rtc_configuration 参数,"
              "已跳过服务端 ICE 配置——请在 PoC 结论中记录服务端经默认路径的实际连通性")
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


with gr.Blocks(title="fastrtc 拓扑 PoC") as app:
    gr.Markdown("# PoC：纯麦克风上行 + 音视频下行\n点击连接后应只弹麦克风权限。")
    webrtc = WebRTC(**build_webrtc_kwargs())
    webrtc.stream(EchoPatternHandler(), inputs=[webrtc], outputs=[webrtc],
                  concurrency_limit=1, time_limit=600)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
