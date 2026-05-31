"""
机械臂控制模块
"""
import time
import threading
import serial.tools.list_ports
from typing import Optional, Tuple
import config

# 机械臂版本检测
try:
    import pymycobot
    from packaging import version
    MAX_VERSION = '3.9.1'
    current_version = pymycobot.__version__

    if version.parse(current_version) > version.parse(MAX_VERSION):
        from pymycobot.ultraArmP340 import ultraArmP340
        ROBOT_CLASS = ultraArmP340
        CLASS_NAME = 'new'
    else:
        from pymycobot.ultraArm import ultraArm
        ROBOT_CLASS = ultraArm
        CLASS_NAME = 'old'
except ImportError:
    ROBOT_CLASS = None
    CLASS_NAME = None


class RobotArm:
    """机械臂控制器"""

    def __init__(self, port: str = config.SERIAL_PORT, baud: int = config.SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self.ua = None
        self.connected = False
        self.lock = threading.Lock()
        self.current_position = None

    def detect_port(self) -> str:
        """自动检测串口"""
        ports = [str(x).split(" - ")[0].strip() for x in serial.tools.list_ports.comports()]
        if ports:
            print(f"检测到串口：{ports}")
            return ports[2] if len(ports) > 2 else ports[0]
        return self.port

    def connect(self) -> bool:
        """连接机械臂"""
        if ROBOT_CLASS is None:
            print("pymycobot 未安装")
            return False

        try:
            port = self.detect_port()
            self.ua = ROBOT_CLASS(port, self.baud)
            self.connected = True
            print(f"机械臂已连接 ({CLASS_NAME}版本)")
            return True
        except Exception as e:
            print(f"机械臂连接失败：{e}")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        if self.ua:
            try:
                self.pump_off()
            except:
                pass
        self.ua = None
        self.connected = False
        print("机械臂已断开")

    def go_zero(self):
        """回零位"""
        if not self.connected:
            return False
        try:
            self.ua.go_zero()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"回零失败：{e}")
            return False

    def set_angles(self, angles: list, speed: int = 50) -> bool:
        """设置关节角度"""
        if not self.connected:
            return False
        try:
            print(f"设置角度：{angles}, speed={speed}")
            if CLASS_NAME == 'new':
                # ultraArmP340 可能需要不同的 API
                self.ua.set_angles(angles, speed)
            else:
                self.ua.set_angles(angles, speed)
            return True
        except Exception as e:
            print(f"设置角度失败：{e}")
            return False

    def set_coords(self, coords: list, speed: int = 50) -> bool:
        """设置笛卡尔坐标"""
        if not self.connected:
            return False
        try:
            self.ua.set_coords(coords, speed)
            self.current_position = coords
            return True
        except Exception as e:
            print(f"设置坐标失败：{e}")
            return False

    def wait_for_stop(self, timeout: float = 10.0) -> bool:
        """
        等待机械臂停止运动（使用坐标检测）

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否在超时前停止
        """
        if not self.connected:
            return False

        start_time = time.time()
        last_coords = None
        stable_count = 0

        while time.time() - start_time < timeout:
            try:
                # 获取当前坐标（API 方法名是 get_coords_info）
                current = self.ua.get_coords_info()
                if current and len(current) >= 3:
                    # 检查坐标是否稳定（连续2次变化小于1mm）
                    if last_coords is not None:
                        diff = sum(abs(current[i] - last_coords[i]) for i in range(3))
                        if diff < 1.0:
                            stable_count += 1
                            if stable_count >= 2:
                                # 坐标稳定，认为到位
                                return True
                        else:
                            stable_count = 0
                    last_coords = current[:]
            except Exception as e:
                print(f"  [ERROR] 获取坐标异常: {e}")
            time.sleep(0.1)

        print("  等待到位超时，继续执行...")
        return False

    def move_to(self, x: float, y: float, z: float, speed: int = 60) -> bool:
        """移动到指定坐标（等待到位）"""
        result = self.set_coords([x, y, z], speed)
        if result:
            self.wait_for_stop(timeout=10.0)
        return result

    def pump_on(self):
        """开启吸泵"""
        if not self.connected:
            return False
        try:
            self.ua.set_gpio_state(0)
            print("吸泵开启")
            return True
        except Exception as e:
            print(f"吸泵开启失败：{e}")
            return False

    def pump_off(self):
        """关闭吸泵"""
        if not self.connected:
            return False
        try:
            self.ua.set_gpio_state(1)
            print("吸泵关闭")
            return True
        except Exception as e:
            print(f"吸泵关闭失败：{e}")
            return False

    def grab_piece(self, x: float, y: float, z_grab: float = config.Z_HEIGHTS['grab'],
                   z_safe: float = config.Z_HEIGHTS['safe']) -> bool:
        """
        抓取棋子流程
        """
        if not self.connected:
            print("机械臂未连接")
            return False

        with self.lock:
            try:
                print(f"开始抓取棋子：({x:.1f}, {y:.1f})")

                # 1. 移动到抓取点上方的安全高度（move_to 已等待到位）
                self.move_to(x, y, z_safe)

                # 2. 下降抓取（move_to 已等待到位）
                self.move_to(x, y, z_grab)
                time.sleep(0.5)

                # 3. 开启吸泵
                self.pump_on()
                time.sleep(0.5)

                # 4. 抬起（move_to 已等待到位）
                self.move_to(x, y, z_safe)

                print("抓取完成")
                return True

            except Exception as e:
                print(f"抓取失败：{e}")
                self.pump_off()
                return False

    def place_piece(self, x: float, y: float, z_place: float = config.Z_HEIGHTS['place'],
                    z_safe: float = config.Z_HEIGHTS['safe']) -> bool:
        """
        放置棋子流程
        """
        if not self.connected:
            print("机械臂未连接")
            return False

        with self.lock:
            try:
                print(f"开始放置棋子：({x:.1f}, {y:.1f})")

                # 1. 移动到放置点上方的安全高度（move_to 已等待到位）
                self.move_to(x, y, z_safe)

                # 2. 下降放置（move_to 已等待到位）
                self.move_to(x, y, z_place)
                time.sleep(0.3)

                # 3. 关闭吸泵
                self.pump_off()
                time.sleep(0.3)

                # 4. 抬起（move_to 已等待到位）
                self.move_to(x, y, z_safe)

                print("放置完成")
                return True

            except Exception as e:
                print(f"放置失败：{e}")
                return False

    def execute_move(self, from_x: float, from_y: float, to_x: float, to_y: float,
                     capture: bool = False, capture_zone: str = 'C') -> bool:
        """
        执行完整的移动棋子流程
        Args:
            from_x, from_y: 起始位置坐标
            to_x, to_y: 目标位置坐标
            capture: 是否是吃子操作
            capture_zone: 被吃棋子的放置区域 ('A', 'B', 'C', 'D')
        """
        if not self.connected:
            print("机械臂未连接")
            return False

        print(f"执行走棋：({from_x:.1f}, {from_y:.1f}) -> ({to_x:.1f}, {to_y:.1f})")
        print(f"capture={capture}, capture_zone={capture_zone}")

        try:
            # 如果是吃子，先把目标棋子移走
            if capture:
                print("Step 1: 开始吃子流程...")
                # 获取放置区域坐标
                place_area = config.PLACE_COORDS.get(capture_zone, config.PLACE_COORDS['C'])
                capture_x, capture_y, capture_z = place_area
                print(f"放置区域 {capture_zone}: ({capture_x:.1f}, {capture_y:.1f}, {capture_z:.1f})")

                # 抓取目标位置的棋子（被吃的棋子）
                print(f"Step 1a: 抓取被吃棋子：({to_x:.1f}, {to_y:.1f})")
                if not self._grab_piece_internal(to_x, to_y):
                    print("抓取被吃棋子失败")
                    self.pump_off()
                    return False

                # 放置到指定区域
                print(f"Step 1b: 放置到区域 {capture_zone}: ({capture_x:.1f}, {capture_y:.1f})")
                if not self._place_piece_internal(capture_x, capture_y):
                    print("放置被吃棋子失败")
                    self.pump_off()
                    return False

                print("吃子完成，继续执行走棋")

            # 抓取要移动的棋子
            print(f"Step 2: 抓取要走棋的棋子：({from_x:.1f}, {from_y:.1f})")
            if not self._grab_piece_internal(from_x, from_y):
                print("抓取要走棋的棋子失败")
                self.pump_off()
                return False

            # 放置到目标位置
            print(f"Step 3: 放置棋子到目标位置：({to_x:.1f}, {to_y:.1f})")
            if not self._place_piece_internal(to_x, to_y):
                print("放置棋子失败")
                self.pump_off()
                return False

            # 返回准备位置
            print("Step 4: 返回准备位置...")
            self.move_to(150, 0, config.Z_HEIGHTS['safe'])

            print("走棋执行完成")
            return True

        except Exception as e:
            print(f"走棋执行失败：{e}")
            import traceback
            traceback.print_exc()
            self.pump_off()
            return False

    def _grab_piece_internal(self, x: float, y: float,
                              z_grab: float = config.Z_HEIGHTS['grab'],
                              z_safe: float = config.Z_HEIGHTS['safe']) -> bool:
        """内部抓取方法（不加锁，供 execute_move 调用）"""
        try:
            print(f"  抓取：({x:.1f}, {y:.1f})")
            # 1. 移动到抓取点上方的安全高度（move_to 已等待到位）
            self.move_to(x, y, z_safe)
            time.sleep(0.3)  # 短暂稳定
            # 2. 下降抓取（move_to 已等待到位）
            self.move_to(x, y, z_grab)
            time.sleep(0.5)  # 等待下降稳定
            # 3. 开启吸泵
            self.pump_on()
            time.sleep(0.5)  # 等待吸泵吸附
            # 4. 抬起（move_to 已等待到位）
            self.move_to(x, y, z_safe)
            print("  抓取完成")
            return True
        except Exception as e:
            print(f"  抓取失败：{e}")
            return False

    def _place_piece_internal(self, x: float, y: float,
                               z_place: float = config.Z_HEIGHTS['place'],
                               z_safe: float = config.Z_HEIGHTS['safe']) -> bool:
        """内部放置方法（不加锁，供 execute_move 调用）"""
        try:
            print(f"  放置：({x:.1f}, {y:.1f})")
            # 1. 移动到放置点上方的安全高度（move_to 已等待到位）
            self.move_to(x, y, z_safe)
            # 2. 下降放置（move_to 已等待到位）
            self.move_to(x, y, z_place)
            time.sleep(0.3)  # 等待下降稳定
            # 3. 关闭吸泵
            self.pump_off()
            time.sleep(0.3)  # 等待棋子放下
            # 4. 抬起（move_to 已等待到位）
            self.move_to(x, y, z_safe)
            print("  放置完成")
            return True
        except Exception as e:
            print(f"  放置失败：{e}")
            return False

    def get_status(self) -> dict:
        """获取机械臂状态"""
        return {
            'connected': self.connected,
            'port': self.port,
            'current_position': self.current_position
        }

    def place_calibration_piece(self, to_x: float, to_y: float,
                                 ready_x: float = 200.0, ready_y: float = 0.0) -> bool:
        """
        棋子标定：从准备位置放置棋子到标定点

        前提：用户已将棋子放到吸泵上（吸泵已开启）

        Args:
            to_x, to_y: 标定点坐标
            ready_x, ready_y: 准备位置坐标（未使用，保留兼容性）

        Returns:
            是否成功
        """
        # 直接使用内部放置方法（经过验证的放置流程）
        return self._place_piece_internal(to_x, to_y)

    def return_calibration_piece(self, from_x: float, from_y: float,
                                  ready_x: float = 200.0, ready_y: float = 0.0) -> bool:
        """
        棋子标定：从标定点取回棋子到准备位置

        Args:
            from_x, from_y: 标定点坐标
            ready_x, ready_y: 准备位置坐标

        Returns:
            是否成功
        """
        # 使用成熟的抓取和放置方法
        if not self._grab_piece_internal(from_x, from_y):
            return False
        if not self._place_piece_internal(ready_x, ready_y):
            return False
        return True
