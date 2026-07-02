# 实时流式数字人面板设计（gradio_app_realtime.py）

日期：2026-07-02
状态：已确认

## 目标

新增第三个面板 `gradio_app_realtime.py`：浏览器麦克风**流式输入音频**，模型**实时流式输出画面**，通过 WebRTC 音画同步下行。支持两种观看模式切换：

- **自己看**：静音播放（避免回声），只看画面对口型
- **给别人看**：音频+视频同步下行播放

现有 `gradio_app.py` 与 `gradio_app_streaming.py` 保持不动。

## 可行性依据（已验证）

1. **模型侧天然流式**：`generate_video.py` 的 `stream` 模式已实现增量路径——8 秒滑窗音频 deque → 每 0.96s 音频（Lite 档 24 帧 × 640 样本 @16kHz）触发一次 chunk 推理 → `latent_motion_frames` 跨 chunk 传递运动连续性。管线代码零改动。
2. **输出通道选型**：Gradio 原生 `gr.Video(streaming=True)` 是 HLS（约 3s 固有缓冲），`gr.Image(streaming=True)` 是逐帧 SSE（实测 2-3fps）。均不满足实时。Gradio 官方推荐方案是伴生库 **fastrtc**（gradio-app/fastrtc，WebRTC 组件可嵌入 gr.Blocks，兼容 gradio 5.50）。
3. **fastrtc pacing 已核验到源码级**：视频轨按 `handler.fps`（设 25）定时轮询 `video_emit()`，音频轨按 20ms 帧轮询 `emit()`，节拍由库负责——天然匹配"chunk 突发 24 帧 → 入队 → 匀速消费"。
4. **生产级蓝本存在**：OpenAvatarChat（HumanAIGC-Engineering）已官方集成 SoulX-FlashHead，其 `flashhead_processor.py` 实现了滑窗缓冲、配对队列、音画同步、追赶/打断，可直接移植参考。
5. **RunPod 可部署**：fastrtc 信令走普通 HTTP POST（穿现有 HTTPS 代理，不需新端口）；媒体流经 TURN 中继（两端均出站连接，Cloudflare TURN 支持 turns:443 纯 TLS）。fastrtc 官方部署文档点名 RunPod 场景并内置 `get_cloudflare_turn_credentials`（HF_TOKEN 免费 10GB/月）。

## 延迟与硬件预期

- 端到端（麦克风→画面）≈ **1.5~2.5s**：攒 chunk 0.96s + Lite 生成 ~0.25s + WebRTC 传输/播放缓冲。此为 24 帧 chunk 设计的物理下限。
- 严格实时仅 **Lite 档**可保证（每 chunk 预算 0.96s）。Pro 档单卡 PRO 6000 实测 1.21s/chunk（预算 1.12s），会累积落后，面板选 Pro 时显示警告。

## 架构

```
浏览器麦克风 ──WebRTC上行──> FastRTC Handler.receive()
                                   │ 重采样 48k→16k（16k 进管线，原声副本留作下行回放）
                                   v
                         RealtimeProcessor.add_audio()
                                   │ 攒满 0.96s 触发
                                   v
                          推理线程（现有管线，零改动）
                          滑窗 deque → wav2vec → generate() → 24 帧
                                   │
                                   v
                     配对输出队列 [(帧, 对应 40ms 音频段) × 24]
                                   │
              fastrtc 库按 25fps 轮询 video_emit() / 20ms 轮询 emit()
                                   v
       浏览器 <──WebRTC下行(视频轨+音频轨,同一 MediaStream)──
```

与 OpenAvatarChat 的差异：不自建 25fps 节拍器线程，pacing 交给 fastrtc 轮询；不做 idle 推理线程（见下）。

## 组件

### flash_head/realtime/processor.py（新增，纯逻辑，不依赖 Gradio/fastrtc）

- 滑窗 deque（8s）+ `_pending_audio` 攒片，移植自 OpenAvatarChat processor，剥离对话栈（speech_id/打断/ChatEngine）
- **麦克风连续直驱**：所有音频（含安静段）都进管线，安静时模型自然输出待机微动，无需 VAD 和 idle 推理线程
- 可选 RMS 噪声门限：低于阈值喂纯零，防背景噪声嘴部抖动（面板高级设置滑块，默认关闭=0）
- **追赶策略**：配对队列深度超过 2 个 chunk（48 帧 ≈ 落后 2 秒）时丢弃最旧整 chunk，保持口型贴近当下（直播语义），状态栏提示
- 推理在独立线程串行执行；输出为 (BGR 帧, 40ms 原声音频段) 配对项

### gradio_app_realtime.py（新增，面板 + fastrtc 集成）

- `AsyncAudioVideoStreamHandler` 子类：
  - `receive()`：收上行音频帧 → 重采样 → `processor.add_audio()`
  - `video_emit()`：非阻塞取配对帧；队列空回参考图静帧
  - `emit()`：取配对音频段；空则回静音帧——保证两轨 PTS 同速推进（音画同步关键，OpenAvatarChat 验证过：idle 时不发音频会导致视频跑先）
- TURN：检测 `HF_TOKEN` → 自动启用 Cloudflare TURN（`rtc_configuration` + `server_rtc_configuration` 都要配）；无则直连（本地/局域网）
- 面板：左列沿用现有输入（条件图片、ckpt/wav2vec 路径、模型档位默认 `lite`、Pro 警告、Face Crop、Seed）；右列 WebRTC 组件 + "播放声音"开关 + 状态栏（chunk 耗时/队列深度/实测 FPS）
- 复用 `flash_head.inference.get_pipeline / get_base_data`

## 错误处理

- 模型未加载就连接 → 界面提示先加载
- WebRTC 连接失败 → 提示检查 TURN（RunPod 必须有 `HF_TOKEN`）
- 队列积压 → 丢旧 chunk + 状态栏黄色提醒"生成速度不足"

## 风险与对策

| 风险 | 对策 |
|---|---|
| 原版 fastrtc audio-video 模式默认申请摄像头（服务端视频下行轨仅在收到客户端视频轨时创建） | 实现第一步先 PoC：试 `track_constraints` 只申请麦克风；不行换 OpenAvatarChat 维护的 gradio-webrtc fork（HumanAIGC-Engineering/gradio-webrtc，avatar 场景专用） |
| 音画长时漂移（WebRTC 两轨独立计时，无共享时钟） | 逐帧配对出队 + 起点对齐；30 分钟长会话实测，必要时按音频钟丢/补视频帧 |
| Cloudflare 免费 TURN 10GB/月 ≈ 10 小时（512×512@25fps ≈ 0.7~1.1 GB/h） | 开发够用；升级路径：自有 Cloudflare key（$0.05/GB）或自建 coturn，仅换配置函数 |
| 浏览器 autoplay 策略 | 连接需用户点击发起（本身就有交互），音频默认静音（自己看模式）规避 |

## 测试策略

1. **无浏览器 PoC 脚本**：直接喂 wav 进 processor，验证攒片边界（不足一片不触发、余量保留）、配对输出、帧率——隔离 WebRTC 变量
2. **端到端**：RunPod 真机浏览器实测玻璃到玻璃延迟（预期 1.5~2.5s）、长会话音画漂移
3. 单元：攒片逻辑、重采样正确性、噪声门限

## 明确不做（YAGNI）

VAD/ASR/LLM/TTS 对话栈、多用户并发（`concurrency_limit=1`）、打断/双工、HLS 降级模式、录制保存（现有面板已覆盖文件生成场景）。

## 新增依赖

- `fastrtc`（或 PoC 结论换 `gradio-webrtc` fork）——单独加入 requirements，注明仅实时面板需要

## 参考材料

- OpenAvatarChat FlashHead 处理器/Handler 源码副本（调研时已下载）：会话 scratchpad `oac_fh_proc.py`、`oac_fh_handler.py`
- fastrtc pacing 源码：`backend/fastrtc/tracks.py`（`next_timestamp()` 绝对墙钟对齐）
- fastrtc 部署/TURN：`docs/deployment.md`、`backend/fastrtc/credentials.py`
