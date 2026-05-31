"""
中国象棋对战机器人系统 - Flask 主应用
"""
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import threading
import time
import json
import cv2
import numpy as np
from typing import Optional
import logging
import asyncio

# 抑制 YOLO/Ultralytics 的详细日志
logging.getLogger('ultralytics').setLevel(logging.WARNING)

import config
from modules.camera import CameraManager
from modules.vision import VisionSystem
from modules.robot_arm import RobotArm
from modules.board_state import BoardStateManager
from modules.move_detector import MoveDetector
from modules.chess_ai import ChineseChessAI

# 语音交互模块
try:
    from modules.voice_interaction import VoiceInteractionManager
    from modules.voice_interaction.llm.dashscope_provider import DashScopeLLMProvider
    from modules.voice_interaction.tts.edge_tts import EdgeTTSProvider
    from modules.voice_interaction.asr import create_asr_provider
    VOICE_MODULE_AVAILABLE = True
except ImportError as e:
    VOICE_MODULE_AVAILABLE = False
    print(f"[App] 警告: 语音交互模块未加载 ({e})")

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局组件实例
camera = CameraManager()
vision = VisionSystem()
robot = RobotArm()
board_state = BoardStateManager()
move_detector = MoveDetector()
chess_ai = ChineseChessAI()

# 语音交互管理器
voice_manager = None
if VOICE_MODULE_AVAILABLE and config.VOICE_INTERACTION_ENABLED:
    try:
        llm_provider = DashScopeLLMProvider()
        tts_provider = EdgeTTSProvider()
        asr_provider = create_asr_provider()  # 使用工厂模式创建 ASR
        voice_manager = VoiceInteractionManager(
            llm_provider=llm_provider,
            tts_provider=tts_provider,
            asr_provider=asr_provider,
            socketio=socketio
        )
        voice_manager.start()
        print("[App] 语音交互模块已启动")
    except Exception as e:
        print(f"[App] 语音交互模块启动失败: {e}")
        voice_manager = None

# 棋子类型 -> 中文短名（用于 move_history）
PIECE_SHORT_MAP = {
    'shuai': '帅', 'jiang': '将',
    'shi': '仕', 'xiang': '相', 'xiang_s': '象',
    'ma': '马', 'che': '车', 'pao': '炮',
    'zu': '卒', 'bing': '兵',
}

def get_piece_short(class_name: str) -> str:
    """从 class_name（如 'red_ma'）提取中文短名（如 '马'）"""
    if not class_name:
        return ''
    piece_type = class_name.split('_')[-1]
    return PIECE_SHORT_MAP.get(piece_type, piece_type)


# 游戏状态
game_state = {
    'status': 'waiting',  # waiting, calibrating, playing, paused, finished
    'current_player': 'red',
    'calibration_complete': False,
    'robot_connected': False,
    'last_detected_move': None,
    'move_history': [],
    'ai_moving': False,  # AI 走棋期间跳过检测
}

# 全局状态（已移除自动检测的 monitor_loop）

# 棋子标定状态
piece_calibration_state = {
    'active': False,
    'current_index': 0,
    'calibration_points': [],
}


def get_frame_with_overlay():
    """获取带叠加信息的帧"""
    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return None

    display_frame = frame.copy()

    # 绘制ROI区域（棋盘检测范围）
    roi = config.BOARD_ROI
    # ROI参数已经是相对于缩放后的图像，直接使用
    x_min, x_max = int(roi['x_min']), int(roi['x_max'])
    y_min, y_max = int(roi['y_min']), int(roi['y_max'])
    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), config.COLORS['blue'], 2)
    cv2.putText(display_frame, "ROI", (x_min + 5, y_min + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS['blue'], 1)

    # 绘制棋盘网格（如果已标定）
    if vision.calibration_complete:
        vision.draw_board_grid(display_frame)

    # 检测棋子
    detections = vision.detect_chess_pieces(frame)
    vision.draw_detections(display_frame, detections)

    # 显示状态信息
    status_text = f"State: {game_state['status']} | Player: {game_state['current_player']}"
    cv2.putText(display_frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLORS['white'], 2)

    if vision.calibration_complete:
        calib_text = f"Calib: {vision.calibration_type} (err:{vision.calibration_error:.1f}mm)"
        cv2.putText(display_frame, calib_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS['green'], 2)

    # 显示检测到的棋子数量
    cv2.putText(display_frame, f"Pieces: {len(detections)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS['yellow'], 2)

    return display_frame


def generate_video_stream():
    """生成 MJPEG 视频流"""
    while True:
        frame = get_frame_with_overlay()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # 30fps


# ==================== 走棋检测函数 ====================

def detect_and_validate_human_move():
    """
    检测并验证人类走棋
    返回: (move, error) - move 为走棋信息字典，error 为错误信息
    """
    global board_state

    # ====== 新增：检查当前回合 ======
    if game_state['current_player'] != 'red':
        return None, '现在是黑方回合，请等待AI走棋完成后再走'

    if game_state['status'] != 'playing':
        return None, f'游戏状态为 {game_state["status"]}，当前不能走棋'

    if game_state.get('ai_moving', False):
        return None, 'AI正在走棋，请等待完成'

    # 获取当前帧
    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return None, '无法获取摄像头画面'

    # 检测棋子
    detections = vision.detect_chess_pieces(frame)
    new_state = BoardStateManager()
    new_state.from_detections(detections)

    # 调试：打印检测到的棋子数量
    print(f"[检测] 当前检测到 {len(new_state.pieces)} 个棋子，内存中有 {len(board_state.pieces)} 个棋子")

    # 如果检测数量异常，打印所有棋子位置
    if len(new_state.pieces) < 30:
        print(f"[检测] 棋子数量异常（{len(new_state.pieces)}），打印所有位置：")
        for pos in sorted(new_state.pieces.keys()):
            piece = new_state.pieces[pos]
            print(f"  {pos}: {piece.class_name}")

    # 与内存状态比较
    diff = new_state.compare(board_state)

    # 调试：打印比较结果
    print(f"[检测] 比较结果: is_valid_move={diff.get('is_valid_move')}, added={len(diff.get('added', []))}, removed={len(diff.get('removed', []))}")

    if not diff.get('is_valid_move', False):
        # 打印详细变化信息帮助调试
        if diff.get('added') or diff.get('removed'):
            return None, f'检测到棋子变化但不是有效走棋：新增{len(diff["added"])}个，移除{len(diff["removed"])}个'
        return None, '未检测到有效走棋，请先走棋后再点击AI走棋'

    # 获取走棋信息（直接从 diff 中获取）
    from_pos = diff.get('from_pos')
    to_pos = diff.get('to_pos')

    if not from_pos or not to_pos:
        return None, '走棋坐标信息不完整'

    # ====== 新增：检查移动的棋子颜色 ======
    moving_piece = board_state.pieces.get(from_pos)
    if moving_piece is None:
        return None, f'起始位置 {from_pos} 没有棋子'

    if moving_piece.color != 'red':
        return None, f'只能移动红方棋子，{from_pos} 是黑方棋子 ({moving_piece.class_name})'

    # 验证走法合法性
    is_valid = chess_ai.is_valid_move(board_state.to_dict(), from_pos, to_pos)

    if not is_valid:
        return None, f'走法不合法: {from_pos} -> {to_pos}，请重新走棋'

    # 返回走棋信息和新的棋盘状态
    return {
        'from_pos': from_pos,
        'to_pos': to_pos,
        'moving_piece': diff.get('moving_piece'),
        'captured': diff.get('captured'),
        'new_state': new_state  # 内部使用，不发送给前端
    }, None


def execute_ai_move_with_verification(ai_move, from_col, from_row, to_col, to_row):
    """
    执行AI走棋（无验证，直接更新内存状态）
    返回: (success, error_message)
    """
    global board_state, game_state

    game_state['ai_moving'] = True

    try:
        # 获取起始位置的棋子（使用 YOLO 检测的实际中心点）
        from_piece = board_state.get_piece_at(from_col, from_row)
        if from_piece and from_piece.center_x is not None and from_piece.center_y is not None:
            from_coords = vision.pixel_to_robot(from_piece.center_x, from_piece.center_y)
            print(f"[AI] 使用 YOLO 中心点抓取: {ai_move['from']} -> 像素({from_piece.center_x:.1f}, {from_piece.center_y:.1f}) -> 机械臂({from_coords[0]:.1f}, {from_coords[1]:.1f})")
        else:
            from_coords = vision.board_to_robot(from_col, from_row)
            print(f"[AI] 使用交叉点坐标抓取: {ai_move['from']} -> 机械臂({from_coords[0]:.1f}, {from_coords[1]:.1f})")

        # 目标位置：优先使用被吃棋子的中心点，否则使用交叉点
        target_piece = board_state.get_piece_at(to_col, to_row)
        is_capture = target_piece is not None
        capture_color = target_piece.color if target_piece else None

        if target_piece and target_piece.center_x is not None and target_piece.center_y is not None:
            to_coords = vision.pixel_to_robot(target_piece.center_x, target_piece.center_y)
            print(f"[AI] 目标位置使用 YOLO 中心点: {ai_move['to']} -> 机械臂({to_coords[0]:.1f}, {to_coords[1]:.1f})")
        else:
            to_coords = vision.board_to_robot(to_col, to_row)
            print(f"[AI] 目标位置使用交叉点坐标: {ai_move['to']} -> 机械臂({to_coords[0]:.1f}, {to_coords[1]:.1f})")

        if from_coords[0] is None or to_coords[0] is None:
            return False, '坐标转换失败'

        # 执行机械臂动作
        result = robot.execute_move(
            from_coords[0], from_coords[1], to_coords[0], to_coords[1],
            capture=is_capture, capture_zone='C' if capture_color == 'red' else 'A'
        )

        print(f"机械臂执行完成，result={result}")

        if not result:
            return False, '机械臂执行失败'

        # 直接更新内存棋盘状态（不做视觉验证）
        # 打印调试信息
        print(f"[AI] 更新前: from({from_col},{from_row})={board_state.get_piece_at(from_col, from_row)}")
        print(f"[AI] 更新前: to({to_col},{to_row})={board_state.get_piece_at(to_col, to_row)}")
        print(f"[AI] 更新前棋子数: {len(board_state.pieces)}")

        success = board_state.move_piece(from_col, from_row, to_col, to_row)

        print(f"[AI] 更新后棋子数: {len(board_state.pieces)}")
        print(f"[AI] move_piece返回: {success}")
        move_detector.force_update(board_state)
        print(f"[AI] 内存状态已更新: {ai_move['from']} -> {ai_move['to']}")

        return True, None

    except Exception as e:
        print(f"execute_ai_move_with_verification 异常：{e}")
        import traceback
        traceback.print_exc()
        return False, f'执行异常: {str(e)}'

    finally:
        game_state['ai_moving'] = False


# ==================== 路由 ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/camera/feed')
def video_feed():
    """视频流"""
    return Response(generate_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/snapshot', methods=['POST'])
def camera_snapshot():
    """拍照"""
    frame = camera.get_frame()
    if frame is not None:
        # 保存到文件
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'success': False, 'error': '无法获取图像'})


@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    """启动摄像头"""
    success = camera.start()
    return jsonify({'success': success})


@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    """停止摄像头"""
    camera.stop()
    return jsonify({'success': True})


@app.route('/api/robot/connect', methods=['POST'])
def robot_connect():
    """连接机械臂"""
    success = robot.connect()
    if success:
        game_state['robot_connected'] = True
        robot.go_zero()
        robot.set_angles(config.MOVE_ANGLES['ready'])
    return jsonify({'success': success, 'connected': game_state['robot_connected']})


@app.route('/api/robot/disconnect', methods=['POST'])
def robot_disconnect():
    """断开机械臂"""
    robot.disconnect()
    game_state['robot_connected'] = False
    return jsonify({'success': True})


@app.route('/api/robot/status')
def robot_status():
    """获取机械臂状态"""
    return jsonify({
        'connected': game_state['robot_connected'],
        'status': robot.get_status()
    })


@app.route('/api/calibration/start', methods=['POST'])
def calibration_start():
    """开始棋子标定流程"""
    global piece_calibration_state

    if not camera.is_running():
        return jsonify({'success': False, 'error': '请先启动摄像头'})

    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '请先连接机械臂'})

    # 重置标定状态
    piece_calibration_state = {
        'active': True,
        'current_index': 0,
        'calibration_points': [],
    }
    game_state['status'] = 'calibrating'

    first_pos = config.PIECE_CALIBRATION_ORDER[0]
    first_info = config.PIECE_CALIBRATION_POSITIONS[first_pos]

    return jsonify({
        'success': True,
        'message': '棋子标定已开始',
        'current': {
            'index': 1,
            'total': len(config.PIECE_CALIBRATION_ORDER),
            'id': first_pos,
            'name': first_info['name'],
            'robot': first_info['robot'],
            'instruction': '请将棋子放到吸泵上，然后点击放置按钮'
        }
    })


@app.route('/api/calibration/place', methods=['POST'])
def calibration_place():
    """放置标定点棋子并检测"""
    global piece_calibration_state

    if not piece_calibration_state['active']:
        return jsonify({'success': False, 'error': '请先开始标定'})

    current_index = piece_calibration_state['current_index']
    if current_index >= len(config.PIECE_CALIBRATION_ORDER):
        return jsonify({'success': False, 'error': '所有标定点已完成'})

    # 获取当前要放置的标定点
    pos_id = config.PIECE_CALIBRATION_ORDER[current_index]
    pos_info = config.PIECE_CALIBRATION_POSITIONS[pos_id]
    robot_x, robot_y = pos_info['robot']

    # 执行放置
    robot.pump_on()
    time.sleep(0.5)

    success = robot._place_piece_internal(robot_x, robot_y)
    robot.move_to(150, 0, config.Z_HEIGHTS['safe'])

    if not success:
        return jsonify({'success': False, 'error': '放置失败'})

    # 等待机械臂完全离开画面
    print("[标定] 等待机械臂离开画面...")
    time.sleep(3)

    # 检测棋子位置
    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return jsonify({'success': False, 'error': '无法获取图像'})

    detections = vision.detect_chess_pieces(frame)
    print(f"[标定] 检测到 {len(detections)} 个棋子")

    if len(detections) == 0:
        return jsonify({'success': False, 'error': '未检测到棋子，请确保棋子已正确放置'})

    # 找到最接近放置位置的棋子
    min_dist = float('inf')
    best_detection = None
    for det in detections:
        center = det['center']
        # 简单距离计算（像素距离）
        # 使用图像中心作为参考点计算相对距离
        img_center_x, img_center_y = frame.shape[1] / 2, frame.shape[0] / 2
        # 这是一个近似，实际距离会在标定后更准确
        dist = np.sqrt((center[0] - img_center_x)**2 + (center[1] - img_center_y)**2)
        if dist < min_dist or best_detection is None:
            # 如果已经有最佳检测结果，检查是否更接近预期位置
            if best_detection is not None:
                # 使用已有标定粗略估计距离
                if vision.transform_matrix is not None:
                    est_robot = vision.pixel_to_robot(center[0], center[1])
                    est_dist = np.sqrt((est_robot[0] - robot_x)**2 + (est_robot[1] - robot_y)**2)
                    if est_dist < min_dist:
                        min_dist = est_dist
                        best_detection = det
                else:
                    min_dist = dist
                    best_detection = det
            else:
                min_dist = dist
                best_detection = det

    if best_detection is None:
        return jsonify({'success': False, 'error': '无法匹配检测到的棋子'})

    # 记录标定点数据
    pixel = best_detection['center']
    calibration_point = {
        'id': pos_id,
        'pixel': [float(pixel[0]), float(pixel[1])],
        'robot': [float(robot_x), float(robot_y)]
    }
    piece_calibration_state['calibration_points'].append(calibration_point)
    piece_calibration_state['current_index'] += 1

    print(f"[标定] 放置 {pos_id}: 像素{pixel}, 机械臂坐标[{robot_x}, {robot_y}]")

    # 检查是否完成所有标定点
    if piece_calibration_state['current_index'] >= len(config.PIECE_CALIBRATION_ORDER):
        # 计算仿射变换矩阵
        success = vision.calculate_calibration(piece_calibration_state['calibration_points'])

        if success:
            # 保存标定数据（透视变换矩阵已计算完成）
            vision.save_calibration()

            # 用四车定位棋盘交叉点（可选，如果棋盘上有四个车的话）
            frame = camera.get_frame_scaled(config.FRAME_SCALE)
            if frame is not None:
                detections = vision.detect_chess_pieces(frame)
                corners = vision.detect_board_corners_from_cars(detections)

                if corners:
                    vision.board_intersections = vision.calculate_intersections_from_corners(corners, config.BOARD_ROTATION)
                    vision.save_calibration()
                    print(f"[标定] 棋盘定位完成，交叉点数: {len(vision.board_intersections)}")
                else:
                    print("[标定] 提示：棋盘上没有四个车，跳过交叉点计算")
                    print("[标定] 请在开始游戏前放置四个车(a1,i1,a10,i10)并点击'扫描棋盘更新交叉点'")

            game_state['calibration_complete'] = True
            game_state['status'] = 'waiting'

            # 确保返回的数据可 JSON 序列化
            calib_points_json = [
                {
                    'id': p['id'],
                    'pixel': [float(p['pixel'][0]), float(p['pixel'][1])],
                    'robot': [float(p['robot'][0]), float(p['robot'][1])]
                }
                for p in piece_calibration_state['calibration_points']
            ]

            return jsonify({
                'success': True,
                'completed': True,
                'message': f'标定完成！误差: {vision.calibration_error:.2f}mm',
                'error': vision.calibration_error,
                'intersections': len(vision.board_intersections),
                'calibration_points': calib_points_json,
                'hint': '请放置四个车(a1,i1,a10,i10)并点击"扫描棋盘更新交叉点"' if len(vision.board_intersections) == 0 else None
            })
        else:
            return jsonify({'success': False, 'error': '标定计算失败'})
    else:
        next_pos = config.PIECE_CALIBRATION_ORDER[piece_calibration_state['current_index']]
        next_info = config.PIECE_CALIBRATION_POSITIONS[next_pos]
        return jsonify({
            'success': True,
            'completed': False,
            'message': f'第 {piece_calibration_state["current_index"]}/{len(config.PIECE_CALIBRATION_ORDER)} 个已放置',
            'current': {
                'index': piece_calibration_state['current_index'] + 1,
                'total': len(config.PIECE_CALIBRATION_ORDER),
                'id': next_pos,
                'name': next_info['name'],
                'robot': next_info['robot'],
                'instruction': '请将棋子放到吸泵上，然后点击放置按钮'
            },
            'last_point': calibration_point
        })


@app.route('/api/calibration/scan_board', methods=['POST'])
def calibration_scan_board():
    """扫描棋盘并用四车定位更新交叉点"""
    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})

    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return jsonify({'success': False, 'error': '无法获取图像'})

    detections = vision.detect_chess_pieces(frame)
    corners = vision.detect_board_corners_from_cars(detections)

    if not corners:
        return jsonify({
            'success': False,
            'error': f'未找到四个车，检测到 {len(detections)} 个棋子',
            'cars_needed': ['a1', 'i1', 'a10', 'i10']
        })

    vision.board_intersections = vision.calculate_intersections_from_corners(corners, config.BOARD_ROTATION)
    vision.save_calibration()

    return jsonify({
        'success': True,
        'intersections_count': len(vision.board_intersections),
        'corners': corners
    })


@app.route('/api/calibration/status')
def calibration_status():
    """获取标定状态"""
    return jsonify({
        'calibrated': vision.calibration_complete,
        'calibration_type': vision.calibration_type,
        'error': vision.calibration_error if vision.calibration_complete else None,
        'intersections': len(vision.board_intersections),
        'points_count': len(vision.calibration_points_data)
    })


@app.route('/api/calibration/reset', methods=['POST'])
def calibration_reset():
    """重置标定"""
    global piece_calibration_state

    vision.reset_calibration()
    vision.reset_smoother()  # 同时重置平滑器
    game_state['calibration_complete'] = False
    piece_calibration_state = {
        'active': False,
        'current_index': 0,
        'calibration_points': [],
    }

    return jsonify({'success': True, 'message': '标定已重置'})


@app.route('/api/calibration/test_grab', methods=['POST'])
def test_calibration_grab():
    """测试标定精度 - 抓取置信度最高的棋子放到放置区"""
    # 1. 检查前置条件
    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})
    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})
    if not camera.is_running():
        return jsonify({'success': False, 'error': '摄像头未启动'})

    # 2. 获取画面并检测棋子
    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return jsonify({'success': False, 'error': '无法获取图像'})

    detections = vision.detect_chess_pieces(frame)

    if len(detections) == 0:
        return jsonify({'success': False, 'error': '未检测到棋子'})

    # 3. 选取置信度最高的棋子
    best_det = max(detections, key=lambda d: d['confidence'])

    # 4. 计算机械臂坐标
    pixel_x, pixel_y = best_det['center']
    robot_x, robot_y = vision.pixel_to_robot(pixel_x, pixel_y)

    # 5. 确定放置区域（红棋→C区，黑棋→A区）
    place_zone = 'C' if best_det['color'] == 'red' else 'A'
    place_coord = config.PLACE_COORDS[place_zone]

    # 6. 异步执行抓取+放置
    def execute_test():
        try:
            print(f"[测试抓取] 抓取 {best_det['class_name']} @ ({robot_x:.1f}, {robot_y:.1f})")
            robot.grab_piece(robot_x, robot_y)
            print(f"[测试抓取] 放置到 {place_zone} 区")
            robot.place_piece(place_coord[0], place_coord[1])
            # 放置后回到安全位置
            robot.move_to(150, 0, config.Z_HEIGHTS['safe'])
            print("[测试抓取] 完成")
        except Exception as e:
            print(f"[测试抓取] 失败: {e}")

    threading.Thread(target=execute_test).start()

    return jsonify({
        'success': True,
        'piece': best_det['class_name'],
        'confidence': best_det['confidence'],
        'pixel': [float(pixel_x), float(pixel_y)],
        'robot': [float(robot_x), float(robot_y)],
        'place_zone': place_zone,
        'message': f'正在抓取 {best_det["class_name"]} 放置到 {place_zone} 区'
    })


@app.route('/api/test/grab-offset', methods=['POST'])
def test_grab_offset():
    """
    抓取偏差测试：检测棋子偏差并追加到 CSV

    每次调用：
    1. 拍照，YOLO 检测所有棋子
    2. 对每个棋子，找最近的交叉点
    3. 计算像素偏差和 mm 偏差
    4. 追加写入 static/offset_results.csv
    5. 返回标注图像 + 偏差数据
    """
    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})
    if not camera.is_running():
        return jsonify({'success': False, 'error': '摄像头未启动'})

    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return jsonify({'success': False, 'error': '无法获取图像'})

    detections = vision.detect_chess_pieces(frame)
    if not detections:
        return jsonify({'success': False, 'error': '未检测到棋子'})

    intersections = vision.board_intersections
    if not intersections:
        return jsonify({'success': False, 'error': '交叉点未计算'})

    results = []
    annotated = frame.copy()

    for det in detections:
        yolo_px, yolo_py = det['center']

        min_dist = float('inf')
        nearest_pos = None
        nearest_px, nearest_py = 0, 0
        for (col, row), (ix, iy) in intersections.items():
            dist = ((yolo_px - ix) ** 2 + (yolo_py - iy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_pos = (col, row)
                nearest_px, nearest_py = ix, iy

        if nearest_pos is None:
            continue

        dx_px = yolo_px - nearest_px
        dy_px = yolo_py - nearest_py

        yolo_rx, yolo_ry = vision.pixel_to_robot(yolo_px, yolo_py)
        grid_rx, grid_ry = vision.pixel_to_robot(nearest_px, nearest_py)
        dx_mm = yolo_rx - grid_rx if yolo_rx and grid_rx else 0
        dy_mm = yolo_ry - grid_ry if yolo_ry and grid_ry else 0
        dist_mm = (dx_mm**2 + dy_mm**2) ** 0.5

        results.append({
            'piece': det['class_name'],
            'confidence': round(det['confidence'], 3),
            'board_pos': f"{chr(ord('a') + nearest_pos[0])}{nearest_pos[1] + 1}",
            'yolo_pixel': [round(yolo_px, 1), round(yolo_py, 1)],
            'grid_pixel': [round(nearest_px, 1), round(nearest_py, 1)],
            'offset_px': [round(dx_px, 1), round(dy_px, 1)],
            'offset_mm': [round(dx_mm, 2), round(dy_mm, 2)],
            'distance_mm': round(dist_mm, 2)
        })

        cv2.circle(annotated, (int(nearest_px), int(nearest_py)), 8, (0, 255, 0), -1)
        cv2.circle(annotated, (int(yolo_px), int(yolo_py)), 8, (0, 0, 255), -1)
        cv2.line(annotated, (int(nearest_px), int(nearest_py)),
                 (int(yolo_px), int(yolo_py)), (0, 255, 255), 2)
        label = f"{det['class_name']} {dist_mm:.1f}mm"
        cv2.putText(annotated, label, (int(yolo_px) + 10, int(yolo_py) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 保存标注图像
    import time as _time
    img_name = f"grab_offset_{int(_time.time())}.jpg"
    cv2.imwrite(os.path.join('static', img_name), annotated)

    # 追加写入 CSV
    csv_path = os.path.join('static', 'offset_results.csv')
    csv_exists = os.path.exists(csv_path)
    import csv
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['位置', '棋子', '置信度',
                             'YOLO像素X', 'YOLO像素Y',
                             '交叉点像素X', '交叉点像素Y',
                             '偏差px_X', '偏差px_Y',
                             '偏差mm_X', '偏差mm_Y', '距离mm'])
        for r in results:
            writer.writerow([
                r['board_pos'], r['piece'], r['confidence'],
                r['yolo_pixel'][0], r['yolo_pixel'][1],
                r['grid_pixel'][0], r['grid_pixel'][1],
                r['offset_px'][0], r['offset_px'][1],
                r['offset_mm'][0], r['offset_mm'][1],
                r['distance_mm']
            ])

    distances = [r['distance_mm'] for r in results]
    avg_dist = sum(distances) / len(distances) if distances else 0
    max_dist = max(distances) if distances else 0

    return jsonify({
        'success': True,
        'image': f'/static/{img_name}',
        'results': results,
        'csv': f'/static/offset_results.csv',
        'summary': {
            'count': len(results),
            'avg_distance_mm': round(avg_dist, 2),
            'max_distance_mm': round(max_dist, 2),
        }
    })


@app.route('/api/test/reset-csv', methods=['POST'])
def test_reset_csv():
    """重置偏差测试 CSV 文件和放置计数器"""
    global test_place_index
    test_place_index = 0
    csv_path = os.path.join('static', 'offset_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    return jsonify({'success': True, 'message': 'CSV 和计数器已重置'})


test_place_index = 0  # 当前放置的交叉点索引


@app.route('/api/test/place-piece', methods=['POST'])
def test_place_piece():
    """按顺序将吸泵上的棋子放置到下一个交叉点"""
    global test_place_index

    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})
    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})
    if not vision.board_intersections:
        return jsonify({'success': False, 'error': '交叉点未计算'})

    # 按顺序排列交叉点 (a1..i10)
    sorted_points = sorted(vision.board_intersections.keys(),
                           key=lambda x: (x[1], x[0]))

    if test_place_index >= len(sorted_points):
        test_place_index = 0
        return jsonify({'success': False, 'error': '全部90个交叉点已放完，已重置'})

    target = sorted_points[test_place_index]
    test_place_index += 1

    robot_x, robot_y = vision.board_to_robot(target[0], target[1])
    if robot_x is None:
        return jsonify({'success': False, 'error': '坐标转换失败'})

    robot.pump_on()
    time.sleep(0.5)
    success = robot._place_piece_internal(robot_x, robot_y)
    robot.move_to(150, 0, config.Z_HEIGHTS['safe'])

    pos_name = f"{chr(ord('a') + target[0])}{target[1] + 1}"
    return jsonify({
        'success': success,
        'position': pos_name,
        'index': test_place_index,
        'total': len(sorted_points),
        'robot': [round(robot_x, 2), round(robot_y, 2)]
    })


@app.route('/api/board/scan', methods=['POST'])
def board_scan():
    """扫描棋盘"""
    frame = camera.get_frame_scaled(config.FRAME_SCALE)
    if frame is None:
        return jsonify({'success': False, 'error': '无法获取图像'})

    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})

    detections = vision.detect_chess_pieces(frame)

    # 更新棋盘状态
    board_state.from_detections(detections)
    move_detector.force_update(board_state)

    return jsonify({
        'success': True,
        'pieces': board_state.to_dict()['pieces'],
        'count': len(detections)
    })


@app.route('/api/board/state')
def board_get_state():
    """获取当前棋盘状态"""
    return jsonify(board_state.to_dict())


@app.route('/api/game/start', methods=['POST'])
def game_start():
    """开始游戏 - 支持标准开局验证和自定义开局模式"""
    global game_state

    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})

    if not camera.is_running():
        return jsonify({'success': False, 'error': '请先启动摄像头'})

    # 解析开局模式参数（silent=True 避免 JSON 解析错误）
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'standard')  # 'standard' 或 'custom'

    try:
        import time

        # 重置平滑器，清空历史
        vision.reset_smoother()

        print(f"[开局] 开始扫描棋盘，共 {config.GAME_START_SCAN_FRAMES} 帧...")

        # 多次扫描棋盘，收集检测结果
        all_detections = []
        for i in range(config.GAME_START_SCAN_FRAMES):
            frame = camera.get_frame_scaled(config.FRAME_SCALE)
            if frame is None:
                return jsonify({'success': False, 'error': '无法获取图像，请检查摄像头'})

            detections = vision.detect_chess_pieces(frame)
            all_detections.append(detections)

            if i < config.GAME_START_SCAN_FRAMES - 1:
                time.sleep(config.GAME_START_SCAN_DELAY)

        # 对每个位置进行投票，取最常见的识别结果
        position_votes = {}  # {pos: {class_name: count}}
        position_best = {}   # {pos: best_detection}

        for detections in all_detections:
            for det in detections:
                pos = det.get('board_pos')
                if pos:
                    if pos not in position_votes:
                        position_votes[pos] = {}
                        position_best[pos] = det  # 保存第一个检测作为参考

                    class_name = det['class_name']
                    position_votes[pos][class_name] = position_votes[pos].get(class_name, 0) + 1

                    # 更新最佳检测（保存置信度最高的）
                    if det['confidence'] > position_best[pos]['confidence']:
                        position_best[pos] = det

        # 取每个位置投票数最多的结果
        final_detections = []
        vote_details = []
        for pos, votes in position_votes.items():
            if votes:
                best_class = max(votes.keys(), key=lambda k: votes[k])
                vote_count = votes[best_class]
                total_votes = sum(votes.values())

                # 使用最佳检测，但修正类别为投票结果
                best_det = position_best[pos].copy()
                best_det['class_name'] = best_class
                best_det['color'] = 'red' if 'red' in best_class else 'black'

                final_detections.append(best_det)

                # 记录投票详情（用于调试）
                if len(votes) > 1:  # 有争议的位置
                    vote_details.append({
                        'pos': pos,
                        'winner': best_class,
                        'votes': dict(votes),
                        'confidence': round(best_det['confidence'], 2)
                    })

        print(f"[开局] 扫描完成，识别到 {len(final_detections)} 个棋子")
        if vote_details:
            print(f"[开局] 有争议的位置: {vote_details}")

        # 构建棋盘状态用于验证
        temp_board = BoardStateManager()
        temp_board.from_detections(final_detections)

        # 根据模式选择验证方式
        if mode == 'standard':
            # 标准开局验证
            is_valid, errors = temp_board.validate_standard_setup()
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': '棋盘布局错误',
                    'errors': errors,
                    'vote_details': vote_details
                })
        elif mode == 'custom':
            # 自定义开局：仅验证基本条件
            if len(temp_board.pieces) < 2:
                return jsonify({
                    'success': False,
                    'error': '棋盘棋子太少，无法开始游戏',
                    'detected_count': len(temp_board.pieces)
                })
            # 验证红黑双方都有棋子
            red_count = sum(1 for p in temp_board.pieces.values() if p.color == 'red')
            black_count = sum(1 for p in temp_board.pieces.values() if p.color == 'black')
            if red_count == 0:
                return jsonify({'success': False, 'error': '红方没有棋子，无法开始游戏'})
            if black_count == 0:
                return jsonify({'success': False, 'error': '黑方没有棋子，无法开始游戏'})
            print(f"[开局] 自定义模式：红方 {red_count} 个棋子，黑方 {black_count} 个棋子")
        else:
            return jsonify({'success': False, 'error': f'未知开局模式: {mode}'})

        # 验证通过，更新棋盘状态
        board_state.from_detections(final_detections)
        move_detector.force_update(board_state)

        # 重置检测平滑器（新局开始）
        vision.reset_smoother()

        game_state['status'] = 'playing'
        game_state['current_player'] = 'red'
        game_state['move_history'] = []

        print(f"[开局] 游戏开始，棋盘状态已更新")

        # 游戏开始后自动启动语音监听
        if voice_manager and config.AUTO_START_LISTENING:
            voice_manager.start_commentary_listening()

        return jsonify({
            'success': True,
            'status': game_state['status'],
            'mode': mode,
            'board': board_state.to_dict(),
            'scan_frames': config.GAME_START_SCAN_FRAMES,
            'vote_details': vote_details
        })
    except Exception as e:
        import traceback
        print(f"启动游戏失败：{e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'启动失败：{str(e)}'})


@app.route('/api/game/stop', methods=['POST'])
def game_stop():
    """停止游戏"""
    game_state['status'] = 'waiting'
    return jsonify({'success': True, 'status': game_state['status']})


@app.route('/api/game/reset', methods=['POST'])
def game_reset():
    """重置游戏"""
    global game_state
    game_state['status'] = 'waiting'
    game_state['current_player'] = 'red'
    game_state['last_detected_move'] = None
    game_state['move_history'] = []
    board_state.clear()
    move_detector.reset()
    return jsonify({'success': True})


@app.route('/api/game/state')
def game_get_state():
    """获取游戏状态"""
    return jsonify(game_state)


@app.route('/api/game/difficulty', methods=['GET', 'POST'])
def game_difficulty():
    """获取或设置 AI 难度"""
    if request.method == 'GET':
        return jsonify({
            'difficulty': config.ENGINE_DIFFICULTY,
            'use_engine': config.USE_ENGINE,
            'available_levels': list(range(1, 21)),
            'engine_available': True  # 引擎文件存在
        })

    # POST: 设置难度
    data = request.get_json()
    level = data.get('level')
    use_engine = data.get('use_engine')

    if level is not None:
        if not 1 <= level <= 20:
            return jsonify({'success': False, 'error': '难度等级应在 1-20 范围'})

        # 更新配置
        config.ENGINE_DIFFICULTY = level

        # 更新引擎难度
        if hasattr(chess_ai, 'set_difficulty'):
            chess_ai.set_difficulty(level)

        return jsonify({
            'success': True,
            'difficulty': level,
            'message': f'难度已设置为 {level}'
        })

    if use_engine is not None:
        config.USE_ENGINE = use_engine
        chess_ai.use_engine = use_engine
        return jsonify({
            'success': True,
            'use_engine': use_engine,
            'message': f'引擎模式已{":开启" if use_engine else "关闭"}'
        })

    return jsonify({'success': False, 'error': '缺少 level 或 use_engine 参数'})


@app.route('/api/game/move/human', methods=['POST'])
def game_human_move():
    """提交人类走棋"""
    data = request.get_json()
    from_pos = data.get('from')
    to_pos = data.get('to')

    if not from_pos or not to_pos:
        return jsonify({'success': False, 'error': '缺少 from 或 to 参数'})

    # ====== 新增：检查当前回合 ======
    if game_state['current_player'] != 'red':
        return jsonify({'success': False, 'error': '现在是黑方回合，请等待AI走棋'})

    if game_state['status'] != 'playing':
        return jsonify({'success': False, 'error': f'游戏状态为 {game_state["status"]}，当前不能走棋'})

    # ====== 新增：检查棋子颜色 ======
    moving_piece = board_state.pieces.get(from_pos)
    if moving_piece is None:
        return jsonify({'success': False, 'error': f'起始位置 {from_pos} 没有棋子'})

    if moving_piece.color != 'red':
        return jsonify({'success': False, 'error': f'只能移动红方棋子，{from_pos} 是黑方棋子'})

    # 验证走法是否合法
    if not chess_ai.is_valid_move(board_state.to_dict(), from_pos, to_pos):
        return jsonify({'success': False, 'error': '非法走法'})

    # 更新棋盘
    col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    from_col = col_names.index(from_pos[0])
    from_row = int(from_pos[1:])
    to_col = col_names.index(to_pos[0])
    to_row = int(to_pos[1:])

    board_state.move_piece(from_col, from_row, to_col, to_row)
    move_detector.force_update(board_state)

    # 记录走棋
    moved_piece = board_state.get_piece_at(to_col, to_row)
    game_state['move_history'].append({
        'player': game_state['current_player'],
        'from': from_pos,
        'to': to_pos,
        'piece_short': get_piece_short(moved_piece.class_name) if moved_piece else ''
    })

    # 切换玩家
    game_state['current_player'] = 'black' if game_state['current_player'] == 'red' else 'red'

    # 通知前端玩家已切换（带上棋盘状态）
    socketio.emit('player_changed', {
        'current_player': 'black',
        'board': board_state.to_dict()
    }, namespace='/')

    return jsonify({
        'success': True,
        'board': board_state.to_dict(),
        'next_player': game_state['current_player']
    })


@app.route('/api/game/move/ai', methods=['POST'])
def game_ai_move():
    """
    AI 走棋 - 新流程
    1. 检测人类走棋
    2. 验证走法合法性
    3. 更新棋盘状态
    4. AI计算最佳走法
    5. 执行机械臂动作
    6. 验证AI走棋结果
    """
    global board_state

    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})

    if not vision.calibration_complete:
        return jsonify({'success': False, 'error': '请先完成标定'})

    if game_state['status'] != 'playing':
        return jsonify({'success': False, 'error': '游戏未开始'})

    # ====== 步骤1: 检测人类走棋 ======
    human_move, error = detect_and_validate_human_move()

    if error:
        # 未检测到有效走棋或走法不合法 - 触发语音提示
        if voice_manager:
            # 通知解说监听循环走棋失败
            asyncio.run_coroutine_threadsafe(
                voice_manager.handle_commentary_move_result({
                    'success': False,
                    'error': error
                }),
                voice_manager._async_loop
            )
        return jsonify({'success': False, 'error': error})

    # ====== 步骤1.5: 获取人类走棋后的局势分数（用于解说中的分数变化分析） ======
    if human_move:
        before_score_cp = chess_ai.get_current_score(board_state.to_dict(), 'red')
        human_move['before_score_cp'] = before_score_cp
        print(f"[解说] 人类走棋后局势分数: {before_score_cp}")

    # ====== 步骤2: 更新棋盘状态（人类走棋） ======
    if human_move:
        board_state = human_move['new_state']
        move_detector.force_update(board_state)

        # 记录人类走棋历史
        game_state['move_history'].append({
            'player': 'red',
            'from': human_move['from_pos'],
            'to': human_move['to_pos'],
            'captured': human_move.get('captured'),
            'piece_short': get_piece_short(human_move.get('moving_piece', {}).get('class_name', ''))
        })

        print(f"人类走棋: {human_move['from_pos']} -> {human_move['to_pos']}")

    # ====== 步骤3: AI计算最佳走法 ======
    best_move = chess_ai.get_best_move(board_state.to_dict(), 'black')

    if best_move is None:
        return jsonify({'success': False, 'error': 'AI无法计算走法'})

    # 检测 AI 是否无合法走法（被将死或困毙）
    if best_move.get('no_legal_moves'):
        game_state['status'] = 'finished'
        reason = '黑方无合法走法'
        if best_move.get('is_checkmate_received'):
            reason = '黑方被将死'
            print("[游戏结束] 黑方被将死")
        elif best_move.get('is_stalemate'):
            reason = '黑方困毙'
            print("[游戏结束] 黑方困毙")
        socketio.emit('game_over', {
            'winner': 'red',
            'reason': reason
        })
        return jsonify({'success': True, 'game_over': True, 'winner': 'red'})

    from_pos = best_move['from']
    to_pos = best_move['to']

    # 检测游戏结束 - 多种条件
    is_game_over = False
    game_over_reason = None

    # 条件 1: AI 走棋将死红方 (mate 1 表示一步将死)
    if best_move.get('is_checkmate') and best_move.get('mate_in') == 1:
        is_game_over = True
        game_over_reason = '将死'
        print(f"[游戏检测] AI 将死红方: mate_in=1")

    # 条件 2: AI 吃掉红帅
    captured_by_ai = best_move.get('captured')
    if captured_by_ai and captured_by_ai.get('class_name') == 'red_shuai':
        is_game_over = True
        game_over_reason = '吃掉红帅'
        print(f"[游戏检测] AI 吃掉红帅，游戏结束")

    # 转换为坐标
    col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    from_col = col_names.index(from_pos[0])
    from_row = int(from_pos[1:])
    to_col = col_names.index(to_pos[0])
    to_row = int(to_pos[1:])

    # ====== 步骤3.5: 立即触发语音解说（与机械臂并行） ======
    if voice_manager and config.AUTO_COMMENTARY:
        voice_manager.on_move_complete(
            board_state=board_state.to_dict(),
            human_move={
                'from_pos': human_move['from_pos'],
                'to_pos': human_move['to_pos'],
                'moving_piece': human_move.get('moving_piece'),
                'captured': human_move.get('captured'),
                'before_score_cp': human_move.get('before_score_cp')  # 人类走棋后的分数
            } if human_move else None,
            ai_move=best_move,  # 包含 AI 走棋后的分数 score_cp
            move_history=game_state['move_history']
        )

        # 通知解说监听循环走棋成功
        asyncio.run_coroutine_threadsafe(
            voice_manager.handle_commentary_move_result({
                'success': True,
                'human_move': human_move,
                'ai_move': best_move
            }),
            voice_manager._async_loop
        )

    # ====== 步骤4: 后台执行机械臂动作 ======
    def execute_and_notify():
        global board_state

        try:
            # 执行AI走棋并验证
            success, error_msg = execute_ai_move_with_verification(
                best_move, from_col, from_row, to_col, to_row
            )

            if success:
                # 记录AI走棋历史
                game_state['move_history'].append({
                    'player': 'black',
                    'from': from_pos,
                    'to': to_pos,
                    'ai': True,
                    'piece_short': PIECE_SHORT_MAP.get(best_move.get('piece', ''), ''),
                    'captured': best_move.get('captured')
                })

                # 切换到红方
                game_state['current_player'] = 'red'

                # 通知前端（不发送 new_state，因为它不是 JSON 可序列化的）
                socketio.emit('player_changed', {
                    'current_player': 'red',
                    'board': board_state.to_dict(),
                    'human_move': {
                        'from_pos': human_move['from_pos'],
                        'to_pos': human_move['to_pos'],
                        'moving_piece': human_move.get('moving_piece'),
                        'captured': human_move.get('captured')
                    } if human_move else None,
                    'ai_move': best_move
                }, namespace='/')

                print(f"AI走棋完成: {from_pos} -> {to_pos}")

                # ====== 游戏结束处理（执行完机械臂后） ======
                if is_game_over:
                    game_state['status'] = 'finished'
                    socketio.emit('game_over', {
                        'winner': 'black',
                        'reason': game_over_reason,
                        'ai_move': best_move
                    })
                    print(f"[游戏结束] 黑方将死红方: {game_over_reason}")

                    # 触发游戏结束语音
                    if voice_manager:
                        asyncio.run_coroutine_threadsafe(
                            voice_manager.on_game_over(winner='black', reason=game_over_reason),
                            voice_manager._async_loop
                        )

            else:
                # AI走棋失败
                socketio.emit('ai_move_failed', {
                    'error': error_msg,
                    'move': best_move
                }, namespace='/')
                print(f"AI走棋失败: {error_msg}")

        except Exception as e:
            print(f"execute_and_notify 异常：{e}")
            import traceback
            traceback.print_exc()
            socketio.emit('ai_move_failed', {
                'error': str(e),
                'move': best_move
            }, namespace='/')

    thread = threading.Thread(target=execute_and_notify)
    thread.daemon = True
    thread.start()

    # 立即返回响应（机械臂在后台执行）
    # 注意：不返回 human_move 中的 new_state，因为它不是 JSON 可序列化的
    return jsonify({
        'success': True,
        'human_move': {
            'from_pos': human_move['from_pos'],
            'to_pos': human_move['to_pos'],
            'moving_piece': human_move.get('moving_piece'),
            'captured': human_move.get('captured')
        } if human_move else None,
        'ai_move': best_move,
        'board': board_state.to_dict(),
        'game_over': is_game_over,  # 前端可以提前知道将死
        'winner': 'black' if is_game_over else None
    })


@app.route('/api/robot/move', methods=['POST'])
def robot_manual_move():
    """手动控制机械臂走棋"""
    data = request.get_json()
    from_x = data.get('from_x')
    from_y = data.get('from_y')
    to_x = data.get('to_x')
    to_y = data.get('to_y')

    if from_x is None or from_y is None or to_x is None or to_y is None:
        return jsonify({'success': False, 'error': '缺少坐标参数'})

    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})

    threading.Thread(target=robot.execute_move,
                    args=(from_x, from_y, to_x, to_y)).start()

    return jsonify({'success': True, 'message': '走棋任务已启动'})


@app.route('/api/robot/grab', methods=['POST'])
def robot_grab():
    """手动抓取"""
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({'success': False, 'error': '缺少坐标参数'})

    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})

    threading.Thread(target=robot.grab_piece, args=(x, y)).start()

    return jsonify({'success': True})


@app.route('/api/robot/place', methods=['POST'])
def robot_place():
    """手动放置"""
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({'success': False, 'error': '缺少坐标参数'})

    if not game_state['robot_connected']:
        return jsonify({'success': False, 'error': '机械臂未连接'})

    threading.Thread(target=robot.place_piece, args=(x, y)).start()

    return jsonify({'success': True})


# ==================== 语音交互 API ====================

@app.route('/api/voice/status')
def voice_status():
    """获取语音交互状态"""
    if not voice_manager:
        return jsonify({'enabled': False, 'available': VOICE_MODULE_AVAILABLE})
    return jsonify({
        'enabled': True,
        'available': True,
        'status': voice_manager.get_status()
    })

@app.route('/api/voice/interrupt', methods=['POST'])
def voice_interrupt():
    """打断当前播报"""
    if voice_manager:
        voice_manager.interrupt_speech()
    return jsonify({'success': True})

@app.route('/api/voice/settings', methods=['GET', 'POST'])
def voice_settings():
    """获取或设置语音配置"""
    if request.method == 'GET':
        return jsonify({
            'enabled': config.VOICE_INTERACTION_ENABLED,
            'auto_commentary': config.AUTO_COMMENTARY,
            'llm_provider': config.LLM_PROVIDER,
            'tts_provider': config.TTS_PROVIDER,
            'dialogue_enabled': config.DIALOGUE_MODE_ENABLED,
            'default_character': config.DEFAULT_CHARACTER
        })

    # POST: 更新配置
    data = request.get_json()
    if 'auto_commentary' in data:
        config.AUTO_COMMENTARY = data['auto_commentary']
    return jsonify({'success': True})


# ========== 对话 API ==========

@app.route('/api/voice/dialogue/start', methods=['POST'])
def dialogue_start():
    """启动对话模式"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    if not config.DIALOGUE_MODE_ENABLED:
        return jsonify({'success': False, 'error': '对话模式未启用'})

    result = voice_manager.start_dialogue()
    return jsonify(result)


@app.route('/api/voice/dialogue/stop', methods=['POST'])
def dialogue_stop():
    """停止对话模式"""
    if not voice_manager:
        return jsonify({'success': False})

    result = voice_manager.stop_dialogue()
    return jsonify(result)


@app.route('/api/voice/dialogue/history')
def dialogue_history():
    """获取对话历史"""
    if not voice_manager:
        return jsonify({'messages': [], 'message_count': 0})

    result = voice_manager.get_dialogue_history()
    return jsonify(result)


@app.route('/api/voice/dialogue/clear', methods=['POST'])
def dialogue_clear():
    """清空对话历史"""
    if not voice_manager:
        return jsonify({'success': False})

    result = voice_manager.clear_dialogue_history()
    return jsonify(result)


@app.route('/api/voice/dialogue/input', methods=['POST'])
def dialogue_input():
    """手动输入对话文本（不使用语音）"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'success': False, 'error': '请输入文本'})

    result = voice_manager.process_dialogue_input(text_input=text)
    return jsonify(result)


# ========== 角色 API ==========

@app.route('/api/voice/character/list')
def character_list():
    """获取可用角色列表（会话模式的对话角色）"""
    if not voice_manager:
        return jsonify({'characters': [
            {'id': 'novice_teacher', 'name': '新手导师', 'description': '耐心指导，帮助新手快速入门', 'is_current': True},
            {'id': 'classical', 'name': '古风棋手', 'description': '文雅古韵，引用典故', 'is_current': False},
            {'id': 'humorous_player', 'name': '幽默棋手', 'description': '诙谐风趣，轻松愉快', 'is_current': False},
            {'id': 'sarcastic', 'name': '嘲讽棋手', 'description': '毒舌傲慢，嘲讽对手', 'is_current': False}
        ]})

    # 获取对话角色列表
    from modules.voice_interaction.character.presets import list_dialogue_characters
    characters = list_dialogue_characters()
    return jsonify({'characters': characters})


@app.route('/api/voice/character/set', methods=['POST'])
def character_set():
    """设置当前角色"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    data = request.get_json()
    character_id = data.get('character_id', '')

    if not character_id:
        return jsonify({'success': False, 'error': '请提供角色 ID'})

    result = voice_manager.set_character(character_id)
    return jsonify(result)


@app.route('/api/voice/character/current')
def character_current():
    """获取当前角色"""
    if not voice_manager:
        return jsonify({'character': None})

    character = voice_manager.get_current_character()
    return jsonify({'character': character})


# ========== 解说角色 API ==========

@app.route('/api/voice/commentary-character/list')
def commentary_character_list():
    """获取解说角色列表"""
    if not voice_manager:
        return jsonify({'characters': [
            {'id': 'professional', 'name': '专业解说', 'description': '专业术语，深入分析'},
            {'id': 'humorous', 'name': '幽默解说', 'description': '诙谐风趣，轻松愉快'}
        ]})

    characters = voice_manager.get_commentary_characters()
    return jsonify({'characters': characters})


@app.route('/api/voice/commentary-character/set', methods=['POST'])
def commentary_character_set():
    """设置解说角色"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    data = request.get_json()
    character_id = data.get('character_id', 'professional')

    result = voice_manager.set_commentary_character(character_id)
    return jsonify(result)


@app.route('/api/voice/commentary-character/current')
def commentary_character_current():
    """获取当前解说角色"""
    if not voice_manager:
        return jsonify({'character': {'id': 'professional', 'name': '专业解说'}})

    character = voice_manager.get_commentary_character()
    return jsonify({'character': character})


# ========== 解说模式持续监听 API ==========

@app.route('/api/voice/commentary/listening/start', methods=['POST'])
def start_commentary_listening():
    """启动解说模式持续监听"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块未初始化'})

    result = voice_manager.start_commentary_listening()
    return jsonify(result)


@app.route('/api/voice/commentary/listening/stop', methods=['POST'])
def stop_commentary_listening():
    """停止解说模式持续监听"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块未初始化'})

    result = voice_manager.stop_commentary_listening()
    return jsonify(result)


@app.route('/api/game/state_for_voice', methods=['GET'])
def game_state_for_voice():
    """为语音模块提供游戏状态验证"""
    can_move = (
        game_state['status'] == 'playing' and
        game_state['current_player'] == 'red' and
        not game_state.get('ai_moving', False)
    )

    reason = ""
    if game_state['status'] != 'playing':
        reason = "游戏未开始，请先点击开始游戏"
    elif game_state['current_player'] != 'red':
        reason = "现在是黑方回合，请等待AI"
    elif game_state.get('ai_moving', False):
        reason = "AI正在思考"

    return jsonify({
        'can_move': can_move,
        'reason': reason,
        'status': game_state['status'],
        'current_player': game_state['current_player']
    })


# ========== 会话模式 API ==========

@app.route('/api/voice/mode', methods=['GET', 'POST'])
def voice_mode():
    """获取或设置交互模式"""
    if request.method == 'GET':
        if not voice_manager:
            return jsonify({'mode': 'commentary'})
        return jsonify({'mode': voice_manager.get_mode()})

    # POST: 设置模式
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    data = request.get_json()
    mode = data.get('mode', 'commentary')
    result = voice_manager.set_mode(mode)
    return jsonify(result)


@app.route('/api/voice/session/start', methods=['POST'])
def session_start():
    """启动会话模式"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    if not config.DIALOGUE_MODE_ENABLED:
        return jsonify({'success': False, 'error': '会话模式未启用'})

    result = voice_manager.start_session()
    return jsonify(result)


@app.route('/api/voice/session/stop', methods=['POST'])
def session_stop():
    """停止会话模式"""
    if not voice_manager:
        return jsonify({'success': False})

    result = voice_manager.stop_session()
    return jsonify(result)


@app.route('/api/voice/session/input', methods=['POST'])
def session_input():
    """文本输入（测试或前端手动输入）"""
    if not voice_manager:
        return jsonify({'success': False, 'error': '语音模块不可用'})

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'success': False, 'error': '请输入文本'})

    result = voice_manager.process_text_input(text)
    return jsonify(result)


# WebSocket 事件
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    emit('status', {'message': '已连接到服务器'})


@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开')


@socketio.on('player_move_complete')
def handle_player_move_complete(data):
    """处理玩家走棋完成，切换到 AI"""
    global game_state
    # 人类走棋完成后，切换到 AI
    game_state['current_player'] = 'black'
    emit('player_changed', {'current_player': 'black'})


@socketio.on('session_text_input')
def handle_session_text_input(data):
    """处理会话模式的文本输入"""
    if not voice_manager:
        emit('voice_error', {'error': '语音模块不可用'})
        return

    text = data.get('text', '')
    if text:
        voice_manager.process_text_input(text)


# 启动应用
if __name__ == '__main__':
    # 启动摄像头
    camera.start()

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    finally:
        camera.stop()
