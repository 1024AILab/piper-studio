"""Local browser console. Acquisition and motion run independently of UI timers."""
from __future__ import annotations
import argparse
import atexit
import html
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
import numpy as np
from .config import CameraConfig, Config
from .devices import discover_cameras
from .engine import Engine
from .preview_image import PreviewImage
from .i18n import t as _t, set_language as _set_language
MODE_NAMES = {'sim': _t('模拟设备'), 'hardware_bus': _t('硬件主从 · 只读采集'), 'software': _t('双 CAN · 软件遥操')}
PHASE_NAMES = {'idle': _t('待命'), 'armed': _t('已准备 · 等待示教'), 'recording': _t('正在录制'), 'saving': _t('正在保存'), 'draft': _t('草稿待处理')}
COLORS = ['#61d6b5', '#f7aa65', '#73aafa', '#c5a4fa', '#f28099', '#d2d97b']

_MODE_NAMES_SOURCE = dict(MODE_NAMES)
_PHASE_NAMES_SOURCE = dict(PHASE_NAMES)

def configure_language(language):
    _set_language(language)
    MODE_NAMES.update({key: _t(value) for key, value in _MODE_NAMES_SOURCE.items()})
    PHASE_NAMES.update({key: _t(value) for key, value in _PHASE_NAMES_SOURCE.items()})

def esc(value):
    return html.escape(str(value))

def table(first, second, labels=(_t('从臂反馈'), _t('主臂目标')), raw=False, stale_channels=()):
    labels = tuple(_t(label) for label in labels)

    def fmt(values, i):
        if values is None or not np.isfinite(values[i]):
            return '—'
        return f'{float(values[i]):.0f}' if raw else f'{float(values[i]) / 1000:.2f}'
    rows = ''.join((f"<tr><td>{('J' + str(i + 1) if i < 6 else _t('夹爪'))}</td><td>{fmt(first, i)}</td><td class='{('stale-target' if i in stale_channels else '')}'>{fmt(second, i)}</td><td>{('0.001°' if raw and i < 6 else 'µm' if raw else '°' if i < 6 else 'mm')}</td></tr>" for i in range(7)))
    return f"""{_t('<div class="signal-table"><table><thead><tr><th>关节</th><th>')}{esc(labels[0])}</th><th>{esc(labels[1])}{_t('</th><th>单位</th></tr></thead><tbody>')}{rows}</tbody></table></div>"""

def target_diagnostics(snapshot):
    if not snapshot['connected']:
        return ''
    held = snapshot['target_status']['held_channels']
    parts = []
    for part in snapshot['target_status']['parts']:
        age = part['age']
        if age is None:
            status = _t('尚未收到')
        elif any((i in held for i in part['channels'])):
            status = _t('静止保留')
        else:
            status = f'{age * 1000:.0f} ms' if age < 1 else f"{age:.1f}{_t(' 秒前')}"
        parts.append(f'''<span title="{esc(part['id'])}">{esc(_t(part['label']))}<b>{status}</b></span>''')
    return '<div class="target-diagnostics">' + ''.join(parts) + '</div>'

def plot(values=None, cursor=None, caption=_t('六轴目标轨迹 · °')):
    caption = _t(caption)
    w, h, left, top, bottom = (760, 175, 40, 15, 145)
    if values is None or len(values) < 2:
        return _t('<div class="plot-empty">连接设备后显示关节轨迹</div>')
    a = np.asarray(values, dtype=float)[:, :6] / 1000
    if len(a) > 300:
        indices = np.linspace(0, len(a) - 1, 300).astype(int)
        a = a[indices]
    low, high = (float(a.min()), float(a.max()))
    margin = max((high - low) * 0.08, 5)
    low, high = (low - margin, high + margin)
    parts = [f"""<svg viewBox="0 0 {w} {h}{_t('" role="img" aria-label="六个关节的角度轨迹">')}"""]
    for v in np.linspace(low, high, 4):
        y = bottom - (v - low) / (high - low) * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - 10}" y2="{y:.1f}" stroke="#233345"/><text x="2" y="{y + 4:.1f}" fill="#8b9db0" font-size="10">{v:.0f}°</text>')
    for j, color in enumerate(COLORS):
        pts = ' '.join((f'{left + i / (len(a) - 1) * (w - left - 10):.1f},{bottom - (v - low) / (high - low) * (bottom - top):.1f}' for i, v in enumerate(a[:, j])))
        parts.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="1.7"/>')
        parts.append(f'<text x="{left + j * 52}" y="170" fill="{color}" font-size="11">J{j + 1}</text>')
    if cursor is not None:
        x = left + max(0, min(1, cursor)) * (w - left - 10)
        parts.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="150" stroke="#fff" stroke-dasharray="4 4"/>')
    parts.append('</svg>')
    return f"""<div class="trajectory"><div class="eyebrow">{esc(caption)}</div>{''.join(parts)}</div>"""

def metrics(snapshot):
    s = snapshot
    phase = s['motion'] or (_t('软件遥操中') if s['teleop'] else PHASE_NAMES[s['phase']])
    fresh = s['sample'] and s['sample'].feedback_age < 0.5 and s['connected']
    badge = 'good' if fresh else 'muted'
    rec = ' recording' if s['phase'] == 'recording' else ''
    age = s['sample'].feedback_age * 1000 if fresh else None
    latency = f'{age:.0f}<small>ms</small>' if age is not None else '—'
    cards = [(_t('工作状态'), f'<span class="state-dot{rec}"></span>{esc(_t(phase))}'), (_t('当前录制'), f"{s['frames']:04d}{_t('<small>帧 / ')}{s['frames'] / 10:.1f}s</small>"), (_t('实际采集频率'), f"{s['fps']:.1f}<small>/ 10 FPS</small>"), (_t('从臂反馈年龄'), latency)]
    markup = ''.join((f'<div class="metric"><span>{title}</span><strong>{value}</strong></div>' for title, value in cards))
    label = _t('● 已准备，等待示教') if s['phase'] == 'armed' else _t('● 采集数据就绪') if s['capture_ready'] else _t('● 可先点击开始录制') if s['can_start_record'] else _t('● 关节反馈在线') if fresh else _t('○ 等待设备')
    connection = f'<span class="pill {badge}">{label}</span>'
    camera_ages = ' · '.join((f'{esc(k)} {v * 1000:.0f} ms' if np.isfinite(v) else f'{esc(k)} —' for k, v in s['camera_ages'].items()))
    errors = list(dict.fromkeys([e for e in [s['error'], *s['readiness_issues']] if e]))
    error = '<div class="error-banner">' + '<br>'.join((esc(_t(e)) for e in errors)) + '</div>' if errors else ''
    hint = ''
    if s['connected'] and (s['phase'] == 'armed' or not s['target_status']['ready'] or s['target_status']['held_channels']):
        text = _t(s['target_status']['message'])
        if s['phase'] == 'armed':
            text = _t('录制已准备，无需再点按钮。') + text + _t('；完整数据到达后自动计帧。')
        elif s['can_start_record'] and (not s['draft_id']):
            text = _t('可先点击「开始录制」，再操作示教器。等待期间不写入空帧。')
        hint = f'<div class="capture-hint">{esc(text)}</div>'
    return f"""<div class="session-bar">{connection}<span>{esc(MODE_NAMES[s['mode']])}</span><span class="camera-ages">{camera_ages}</span></div><div class="metrics">{markup}</div>{hint}{error}"""

def library_summary(engine):
    info = engine.store.info or {}
    return f"""<div class="library-summary"><strong>{info.get('total_episodes', 0):02d}</strong> episodes<span>{info.get('total_frames', 0):,}{_t(' 帧</span><span>')}{info.get('total_tasks', 0)}{_t(' 个任务</span><span>10 FPS · LeRobot v2.1</span></div><div class="dataset-path">')}{esc(engine.store.root)}</div>"""

def build_app(engine: Engine, language=None):
    if language is not None:
        configure_language(language)

    def checked(fn, *args):
        try:
            return _t(fn(*args))
        except Exception as exc:
            raise gr.Error(_t(str(exc))) from exc

    def episode_choices():
        return [(f"{e['episode_index']:03d} · {e['tasks'][0]} · {e['length'] / 10:.1f}s", e['episode_index']) for e in engine.store.episodes()]

    def refresh_library():
        choices = episode_choices()
        return (gr.update(choices=choices, value=choices[-1][1] if choices else None), library_summary(engine))

    def refresh_drafts():
        current = engine.snapshot()['draft_id']
        drafts = [d for d in engine.store.drafts() if d['id'] != current]
        choices = [(f"{d['task']} · {d['frames']}{_t(' 帧 · ')}{d['id'][:8]}", d['id']) for d in drafts]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def finish_episode():
        result = checked(engine.save_record)
        choices, summary = refresh_library()
        return (result, choices, summary)

    def recover(draft_id):
        result = checked(engine.recover_draft, draft_id)
        choices, summary = refresh_library()
        return (result, choices, summary, refresh_drafts())

    def live_tick():
        s = engine.snapshot()
        sample = s['sample']
        target_values = sample.partial_action if sample and sample.partial_action is not None else sample.action if sample else None
        stale_channels = [i for p in s['target_status']['parts'] if p['age'] is None or p['age'] > engine.config.stale_after_s for i in p['channels']]
        signal_table = table(sample.follower if sample and s['connected'] else None, target_values if s['connected'] else None, stale_channels=stale_channels)
        signal_table += target_diagnostics(s)
        values = [r[2] for r in s['trace'] if r[2] is not None]
        if s['motion']:
            execution = f"**{s['motion']}** · {s['motion_elapsed']:.1f} / {s['motion_duration']:.1f}{_t('s · 目标帧 ')}{s['motion_frame'] + 1} / {s['motion_total']}"
        else:
            execution = _t('执行已准备 · 当前无平台轨迹') if s['prepared'] else _t('尚未准备从臂执行')
        can_note = _t('\n最近 1 秒收到的 CAN ID：') + ', '.join((hex(i) for i in s['observed_can_ids'])) if s['observed_can_ids'] else ''
        for bus in s['can_status']:
            can_note += f"\n{bus['name']}：{bus['state']}{_t(' · 已收 ')}{bus['received_frames']}{_t(' 帧 · 异常 ')}{bus['error_frames']}{_t(' · 状态通知 ')}{bus['status_frames']}"
            for event in bus['recent_events'][-4:]:
                can_note += f"\n{event['time']}  {event['message']}"
        return (s['images'].get('image'), s['images'].get('wrist_image'), metrics(s), signal_table, plot(values), _t(s['logs']) + _t(can_note), gr.update(value=_t('已准备 · 等待示教') if s['phase'] == 'armed' else _t('●  正在录制') if s['phase'] == 'recording' else _t('●  开始录制'), interactive=s['can_start_record'] and (not s['draft_id']) and (not s['motion']) and (s['phase'] != 'saving')), gr.update(interactive=bool(s['draft_id']) and s['frames'] > 0 and (s['phase'] != 'saving')), gr.update(value=_t('取消等待') if s['phase'] == 'armed' else _t('丢弃草稿'), interactive=bool(s['draft_id']) and s['phase'] != 'saving'), execution, gr.update(interactive=not s['draft_id'] and s['phase'] != 'saving'))
    preview_cache = [None]

    def preview_tick():
        try:
            row = engine.preview()
        except Exception as exc:
            return (*[gr.skip()] * 6, f"{_t('回放读取失败：')}{exc}")
        if row is None:
            if preview_cache[0] is None:
                return (gr.skip(),) * 7
            preview_cache[0] = None
            return (None, None, 0, table(None, None, ('state', 'actions'), True), plot(), _t('尚未载入 episode'), '')
        key = (str(engine.store.root), row['episode_index'], row['frame_index'])
        n = len(engine.preview_signals['actions'])
        status = f"**Episode {row['episode_index']:03d}{_t('** · 帧 ')}{row['frame_index'] + 1} / {n} · {row['timestamp']:.1f}s"
        if preview_cache[0] == key:
            return (*[gr.skip()] * 5, status, '')
        preview_cache[0] = key
        return (row['image'], row['wrist_image'], row['frame_index'], table(row['state'], row['actions'], ('state', 'actions'), True), plot(engine.preview_signals['actions'], row['frame_index'] / max(n - 1, 1), _t('保存的 actions · 白线为当前帧')), status, '')

    def load_episode(index):
        n = checked(engine.load_preview, index)
        preview_cache[0] = None
        return (gr.update(minimum=0, maximum=max(n - 1, 1), value=0, interactive=True), f"{_t('已载入 ')}{n}{_t(' 帧，可以播放或拖动时间轴')}")

    def play(speed):
        checked(engine.play_preview, speed)
        return _t('正在播放采集画面与关节数据')

    def pause_preview():
        with engine.preview_lock:
            engine.preview_playing = False
        return _t('画面回放已暂停')

    def step_preview(delta):
        checked(engine.seek_preview, engine.preview_frame + delta)

    def apply_config(raw):
        try:
            data = json.loads(raw)
            data['image'] = CameraConfig(**data.get('image', {}))
            data['wrist_image'] = CameraConfig(**data.get('wrist_image', {}))
            config = Config(**data)
            config.validate()
            engine.configure(config)
        except Exception as exc:
            raise gr.Error(_t(str(exc))) from exc
        choices, summary = refresh_library()
        return (_t('配置已应用，请回到实时采集页连接设备'), choices, summary, False)

    def preset(name):
        c = Config(mode=name, dataset_root='datasets/sim_piper_dataset' if name == 'sim' else 'datasets/piper_dataset')
        if name != 'sim':
            c.image = CameraConfig('opencv', '0' if sys.platform == 'darwin' else '/dev/video0')
            c.wrist_image = CameraConfig('opencv', '1' if sys.platform == 'darwin' else '/dev/video2')
        if name == 'software':
            c.follower_port, c.leader_port = ('can_follower', 'can_leader')
        if sys.platform == 'darwin':
            c.can_backend, c.follower_port, c.leader_port = ('gs_usb', '0', '1')
        return json.dumps(c.to_dict(), ensure_ascii=False, indent=2)
    with gr.Blocks(title=_t('Piper Studio · 采集与回放'), analytics_enabled=False, fill_width=True) as app:
        gr.HTML(_t('<header class="studio-header"><div class="brand-symbol">P<span>•</span></div><div><div class="brand">PIPER <b>STUDIO</b></div><div class="brand-sub">采集每一次示范，回看每一个动作。</div></div><div class="header-meta"><span>单主从工作站</span><span class="version">LOCAL / 01</span></div></header>'))
        notice = gr.Markdown(_t('填写采集任务名称 → 点击开始录制 → 操作示教器 → 保存 episode。可以先点开始，再动机械臂。'), elem_id='notice')
        with gr.Tabs():
            with gr.Tab(_t('实时采集'), id='capture'):
                saved = engine.store.episodes()
                task = gr.Textbox(label=_t('采集任务名称'), value=saved[-1]['tasks'][0] if saved else '', lines=1, interactive=True, elem_id='capture-task-name', placeholder=_t('例如：抓起小黄鸭，放入笔筒'), info=_t('同一任务的多条示范使用相同名称；名称会保存到任务列表，并用于训练。'))
                with gr.Row(elem_classes='capture-layout'):
                    with gr.Column(scale=8, min_width=540):
                        live_metrics = gr.HTML(metrics(engine.snapshot()))
                        with gr.Row():
                            front = PreviewImage(label=_t('01 / 场景相机 · image'), interactive=False, height=320, format='jpeg', buttons=['fullscreen'], elem_classes='camera-view')
                            wrist = PreviewImage(label=_t('02 / 腕部相机 · wrist_image'), interactive=False, height=320, format='jpeg', buttons=['fullscreen'], elem_classes='camera-view')
                        live_plot = gr.HTML(plot())
                        with gr.Accordion(_t('运行日志'), open=False):
                            logs = gr.Textbox(label=_t('最近事件'), lines=5, interactive=False, show_label=False)
                            can_diagnose = gr.Button(_t('查看 CAN 具体状态'), size='sm')
                            can_details = gr.JSON(label=_t('CAN 接收统计与最近错误详情'))
                    with gr.Column(scale=3, min_width=285, elem_classes='control-panel'):
                        gr.HTML(_t('<div class="section-label"><span>01</span> 本次示范</div>'))
                        gr.Markdown(_t('先点开始，再操作示教器。收到完整目标后自动录制；完成一次操作后保存。'))
                        with gr.Row():
                            connect = gr.Button(_t('连接设备'), size='sm', elem_id='connect-button')
                            disconnect = gr.Button(_t('断开'), size='sm')
                        record = gr.Button(_t('●  开始录制'), variant='primary', interactive=False, elem_id='record-button')
                        with gr.Row():
                            save = gr.Button(_t('保存 episode'), interactive=False, size='sm')
                            discard = gr.Button(_t('丢弃草稿'), interactive=False, size='sm')
                        stop = gr.Button(_t('■  停止平台运动 / 录制'), variant='stop', size='sm', elem_id='stop-button')
                        gr.HTML(_t('<p class="microcopy">硬件主从由设备直接跟随。此按钮停止平台发送和录制；硬件直连的停止请使用设备自身控制。</p>'))
                        gr.HTML(_t('<div class="section-label"><span>02</span> 关节监视</div>'))
                        live_table = gr.HTML(table(None, None))
                        gr.HTML(_t('<p class="microcopy">从臂反馈是实际位置，主臂目标是示教指令。灰色为上次收到的目标，横线表示尚未收到。面板单位为 ° / mm。</p>'))
                        with gr.Accordion(_t('从臂执行与起始位'), open=False):
                            gr.Markdown(_t('硬件采集时可用示教器人工回到起始位。由平台复位或回放前，需要停止主臂向从臂发指令。'))
                            ack = gr.Checkbox(label=_t('已断开主臂对从臂的控制连接，运动路径已检查'), value=False)
                            prepare = gr.Button(_t('准备从臂执行'), size='sm')
                            with gr.Row():
                                remember = gr.Button(_t('记住起始位'), size='sm')
                                home = gr.Button(_t('回到起始位'), size='sm')
                            teleop = gr.Button(_t('启动软件遥操 / 模拟示范'), size='sm')
                            gr.Markdown(_t('起始位是本次任务的姿态。平台不会执行 `ResetPiper` 或重设机械零点。'))
            with gr.Tab(_t('数据与回放'), id='replay'):
                summary = gr.HTML(library_summary(engine))
                with gr.Row():
                    episodes = gr.Dropdown(choices=episode_choices(), value=None, label=_t('已保存的 episode'), scale=8)
                    refresh = gr.Button(_t('刷新列表'), size='sm', scale=1)
                    load = gr.Button(_t('载入回放'), variant='primary', scale=1)
                with gr.Row():
                    with gr.Column(scale=8, min_width=540):
                        with gr.Row():
                            replay_front = PreviewImage(label=_t('场景画面'), interactive=False, height=320, format='jpeg', buttons=['fullscreen'])
                            replay_wrist = PreviewImage(label=_t('腕部画面'), interactive=False, height=320, format='jpeg', buttons=['fullscreen'])
                        frame_status = gr.Markdown(_t('尚未载入 episode'))
                        scrub = gr.Slider(0, 1, value=0, step=1, label=_t('时间轴 / 帧'), interactive=False)
                        with gr.Row():
                            play_button = gr.Button(_t('▶  播放画面'), variant='primary', scale=2)
                            pause_button = gr.Button(_t('Ⅱ  暂停'), scale=1)
                            visual_speed = gr.Dropdown([0.25, 0.5, 1.0, 2.0], value=1.0, label=_t('画面倍率'), scale=1)
                        with gr.Row():
                            previous_frame = gr.Button(_t('←  上一帧'), size='sm')
                            next_frame = gr.Button(_t('下一帧  →'), size='sm')
                        replay_plot = gr.HTML(plot())
                        preview_error = gr.Markdown('')
                    with gr.Column(scale=3, min_width=285, elem_classes='control-panel'):
                        gr.HTML(_t('<div class="section-label"><span>01</span> 当前帧 · 原始值</div>'))
                        replay_table = gr.HTML(table(None, None, ('state', 'actions'), raw=True))
                        gr.Markdown(_t('`state[t] = actions[t−1]`\n\n每条首帧使用当前动作初始化。相机画面与动作按同一帧索引回看。'))
                        gr.HTML(_t('<div class="section-label"><span>02</span> 机械臂回放</div>'))
                        execution_status = gr.Markdown(_t('尚未准备从臂执行'))
                        gr.Markdown(_t('将已载入的 `actions` 发给从臂。先在实时页完成“准备从臂执行”。回放会缓慢靠近首帧，并限制关节速度。'))
                        gr.Markdown(_t('执行时可切换到实时采集页观察当前相机画面和从臂反馈。'))
                        robot_speed = gr.Slider(0.1, 1, value=0.5, step=0.1, label=_t('机械臂回放倍率'))
                        replay_button = gr.Button(_t('在从臂上执行已载入轨迹'), elem_id='robot-replay-button')
                        replay_stop = gr.Button(_t('■  停止平台运动'), variant='stop', size='sm')
                with gr.Accordion(_t('格式检查与中断草稿恢复'), open=False):
                    verify = gr.Button(_t('检查数据集格式与图像'), size='sm')
                    validation = gr.JSON(label=_t('检查结果'))
                    drafts = gr.Dropdown(label=_t('尚未保存的草稿'), choices=[], value=None)
                    with gr.Row():
                        find_drafts = gr.Button(_t('查找中断草稿'), size='sm')
                        restore = gr.Button(_t('恢复并保存选中草稿'), size='sm')
            with gr.Tab(_t('设备配置'), id='settings'):
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown(_t('选择接入方式，再修改配置中的 CAN 名称、相机设备和数据目录。应用配置会断开当前设备。'))
                        mode = gr.Radio(choices=[(v, k) for k, v in MODE_NAMES.items()], value=engine.config.mode, label=_t('填入配置模板'))
                        config_text = gr.Code(value=json.dumps(engine.config.to_dict(), ensure_ascii=False, indent=2), language='json', label=_t('设备配置 JSON'), lines=24, interactive=True)
                        apply_button = gr.Button(_t('应用配置'), variant='primary')
                    with gr.Column(scale=4, elem_classes='config-help'):
                        gr.Markdown(_t('### 你的主臂带示教器\n\n如果主臂已经通过 CAN 直接带着从臂运动，选择 **硬件主从**。电脑接从臂所在总线，读取主臂控制帧和从臂反馈，采集过程不发送运动指令。\n\n如果主、从臂分别连接电脑的两个 CAN 口，选择 **软件遥操**。`leader_source: control` 读取示教器的目标；普通被动拖动主臂可使用 `feedback`。\n\n### 两路相机\n\n- USB：`backend: opencv`，`device` 使用 `/dev/video0` 等路径。\n- Mac 普通 RGB 相机：`backend: opencv`，`device` 使用 `0`、`1` 等索引，核对两路画面。\n- RealSense：`backend: realsense`，`device` 填相机序列号。\n- Mac D435：若 SDK 报 USB 权限错误，使用 `realsense_bridge`，另开“启动RealSense彩色流.command”。\n- 图像必须为 640 × 480，建议相机 30 FPS；保存固定为 10 FPS。\n\n### 在采集电脑上运行\n\nLinux 使用已激活的 `socketcan`；Mac 使用 `gs_usb` 和兼容的 USB-CAN 转接器，并安装 libusb。Mac 的 `follower_port` 填 `0` 等设备索引。\n\n请配置两台 D435 的实际序列号，并使用独立程序读取 RGB。关节反馈与相机就绪后即可点击开始；主臂目标尚未到达时会等待示教。`hold_stationary_targets` 仅用于硬件采集：目标停发时，只有从臂已到达最后目标且持续静止才保留该目标继续记录；未收到过的值不会被补造。浏览器默认仅监听本机。\n'))
                        discover = gr.Button(_t('查找相机和 CAN 接口'), size='sm')
                        devices = gr.JSON(label=_t('当前电脑上的设备'))
        gr.HTML(_t('<div class="studio-footer"><span>PIPER STUDIO</span><span>RGB × 2 / 7-DOF / 10 FPS</span><span>兼容 LeoApollo / piper_single</span></div>'))
        connect.click(lambda: checked(engine.connect), outputs=notice, api_name='connect', concurrency_id='operations')
        disconnect.click(lambda: checked(engine.disconnect) or _t('设备通信已断开，草稿保留'), outputs=notice, api_name='disconnect', concurrency_id='operations')
        record.click(lambda t: checked(engine.start_record, t), inputs=task, outputs=notice, api_name='record')
        save.click(finish_episode, outputs=[notice, episodes, summary], api_name='save', concurrency_id='operations')
        discard.click(lambda: checked(engine.discard_record), outputs=notice, api_name='discard', concurrency_id='operations')
        stop.click(lambda: checked(engine.stop), outputs=notice, queue=False, api_name='stop')
        replay_stop.click(lambda: checked(engine.stop), outputs=notice, queue=False, api_name='stop_replay')
        prepare.click(lambda a: checked(engine.prepare, a), inputs=ack, outputs=notice, api_name='prepare')
        remember.click(lambda: checked(engine.set_home), outputs=notice, api_name='remember_home')
        home.click(lambda: checked(engine.go_home), outputs=notice, api_name='go_home')
        teleop.click(lambda: checked(engine.start_teleop), outputs=notice, api_name='teleop')
        refresh.click(refresh_library, outputs=[episodes, summary], api_name='episodes')
        load.click(load_episode, inputs=episodes, outputs=[scrub, notice], api_name='load_episode')
        play_button.click(play, inputs=visual_speed, outputs=notice, api_name='play')
        pause_button.click(pause_preview, outputs=notice, api_name='pause')
        scrub.input(lambda f: engine.seek_preview(f), inputs=scrub, api_name='seek', queue=False)
        previous_frame.click(lambda: step_preview(-1), api_name='previous_frame', queue=False)
        next_frame.click(lambda: step_preview(1), api_name='next_frame', queue=False)
        replay_button.click(lambda v: checked(engine.replay_robot, engine.preview_index, v), inputs=robot_speed, outputs=notice, api_name='replay_robot')
        verify.click(lambda: checked(engine.store.check, True), outputs=validation, api_name='validate')
        find_drafts.click(refresh_drafts, outputs=drafts, api_name='drafts')
        restore.click(recover, inputs=drafts, outputs=[notice, episodes, summary, drafts], api_name='recover', concurrency_id='operations')
        mode.change(preset, inputs=mode, outputs=config_text, api_name=False)
        apply_button.click(apply_config, inputs=config_text, outputs=[notice, episodes, summary, ack], api_name='configure', concurrency_id='operations')
        discover.click(discover_cameras, outputs=devices, api_name='discover')
        can_diagnose.click(engine.can_diagnostics, outputs=can_details, api_name='can_diagnostics', queue=False)
        live_timer = gr.Timer(0.25)
        live_timer.tick(live_tick, outputs=[front, wrist, live_metrics, live_table, live_plot, logs, record, save, discard, execution_status, task], queue=False, show_progress='hidden', api_name='live_status')
        preview_timer = gr.Timer(0.1)
        preview_timer.tick(preview_tick, outputs=[replay_front, replay_wrist, scrub, replay_table, replay_plot, frame_status, preview_error], queue=False, show_progress='hidden', api_name='preview')
    return app

def main():
    parser = argparse.ArgumentParser(description=_t('Piper Studio 本地数采与回放工作站'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--open', action='store_true', help=_t('启动后打开浏览器'))
    parser.add_argument('--connect-sim', action='store_true', help=_t('仅模拟模式可自动连接'))
    parser.add_argument('--lang', choices=['en', 'zh'], default='en', help='UI language')
    args = parser.parse_args()
    configure_language(args.lang)
    engine = Engine(Config.load(args.config))
    if args.connect_sim:
        if engine.config.mode != 'sim':
            parser.error(_t('--connect-sim 仅允许模拟配置'))
        engine.connect()
    atexit.register(engine.close)
    app = build_app(engine)
    theme = gr.themes.Base(primary_hue='orange', secondary_hue='teal', neutral_hue='slate', font=['Inter', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'], font_mono=['SFMono-Regular', 'Consolas', 'monospace'])
    try:
        app.queue(default_concurrency_limit=1, max_size=32).launch(server_name='127.0.0.1', server_port=args.port, inbrowser=args.open, share=False, theme=theme, css_paths=Path(__file__).with_name('styles.css'), js="() => { document.body.classList.add('dark'); }", footer_links=[], show_error=True)
    finally:
        engine.close()
if __name__ == '__main__':
    main()
