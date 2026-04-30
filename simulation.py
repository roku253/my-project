import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

# センサデータの読み込み
def load_sensor_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    current_pattern = None
    for line in lines:
        if line.startswith('>>> Measurement Start!'):
            parts = line.split('Pattern:')
            if len(parts) > 1:
                try:
                    current_pattern = int(parts[1].split(',')[0].strip())
                except ValueError:
                    current_pattern = None
            continue
        if line.startswith('>>> Measurement Finished'):
            current_pattern = None
            continue
        if line.startswith('ResultLog:'):
            continue
        if line.strip() == '' or ',' not in line:
            continue

        parts = line.strip().split(',')
        if len(parts) >= 7:
            try:
                row = [float(parts[0]), int(float(parts[1])), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])]
                if current_pattern is not None:
                    row[1] = current_pattern
                data.append(row)
            except ValueError:
                continue

    columns = ['Time(s)', 'Pattern', 'CompX(deg)', 'CompY(deg)', 'Vst_y(m/s)', 'Vsy(m/s)', 'Vsx(m/s)']
    df = pd.DataFrame(data, columns=columns)
    return df

# EKF クラス
class ExtendedKalmanFilter:
    def __init__(self, dt, Q, R):
        self.dt = dt
        self.Q = Q  # プロセスノイズ
        self.R = R  # 測定ノイズ
        self.x = np.zeros(6)  # [x, y, vx, vy, θ, ω]
        self.P = np.eye(6) * 0.1  # 初期共分散

    def predict(self, u, dt=None):
        if dt is None:
            dt = self.dt
        # 状態遷移行列 F
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt
        # 制御入力 B (簡易)
        B = np.zeros((6, 2))  # u = [ax, ay]
        B[2, 0] = dt
        B[3, 1] = dt

        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # 測定モデル H (速度と姿勢を測定)
        H = np.zeros((3, 6))
        H[0, 2] = 1  # vx
        H[1, 3] = 1  # vy
        H[2, 4] = 1  # θ

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

# 姿勢監視
def monitor_posture(theta_est, threshold=10.0):
    if abs(theta_est) > threshold:
        print(f"姿勢異常検知: θ = {theta_est:.2f} deg")
        return True
    return False

# メインシミュレーション
def main():
    # パラメータ
    dt = 0.01
    Q = np.eye(6) * 0.01  # プロセスノイズ
    R = np.eye(3) * 0.1   # 測定ノイズ
    ekf = ExtendedKalmanFilter(dt, Q, R)

    # データ読み込み
    data = load_sensor_data('sensor_log.txt')
    available_patterns = sorted(data['Pattern'].astype(int).unique().tolist())
    pattern_list_text = ', '.join(str(p) for p in available_patterns)
    print(f"利用可能なパターン: {pattern_list_text}")

    pattern = None
    while pattern not in available_patterns:
        try:
            pattern = int(input('再生したいパターン番号を入力してください (例: 1): ').strip())
        except ValueError:
            pattern = None

    pattern_data = data[data['Pattern'] == pattern].copy()
    if pattern_data.empty:
        print(f"パターン {pattern} のデータが見つかりません。")
        return

    pattern_data['Time(s)'] = pattern_data['Time(s)'] - pattern_data['Time(s)'].iloc[0]
    times = pattern_data['Time(s)'].astype(float).values
    compx = pattern_data['CompX(deg)'].astype(float).values
    compy = pattern_data['CompY(deg)'].astype(float).values
    vsy = pattern_data['Vsy(m/s)'].astype(float).values
    vsx = pattern_data['Vsx(m/s)'].astype(float).values

    # ログ
    states = []

    last_time = 0.0
    print(f"パターン {pattern} をリアルタイム再生します。総サンプル数: {len(times)}, 再生時間: {times[-1]:.2f}s")

    for i, t in enumerate(times):
        if i > 0:
            sleep_time = t - last_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        last_time = t

        # 測定値
        z = np.array([vsx[i], vsy[i], compx[i] * np.pi / 180])

        # 制御はデータに含まれているため、EKFのみ実行
        u = np.zeros(2)

        # EKF予測
        ekf.predict(u, dt=(t - times[i-1]) if i > 0 else dt)

        # EKF更新
        ekf.update(z)

        # 姿勢監視
        monitor_posture(ekf.x[4] * 180 / np.pi)

        print(f"t={t:.2f}s | x={ekf.x[0]:.3f} y={ekf.x[1]:.3f} vx={ekf.x[2]:.3f} vy={ekf.x[3]:.3f} θ={ekf.x[4]*180/np.pi:.2f}deg")

        states.append(ekf.x.copy())
    print()  # 最後に改行

    # 可視化
    states = np.array(states)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(times, states[:, 0], label='x')
    plt.title('Position X')
    plt.subplot(2, 3, 2)
    plt.plot(times, states[:, 1], label='y')
    plt.title('Position Y')
    plt.subplot(2, 3, 3)
    plt.plot(times, states[:, 2], label='vx')
    plt.title('Velocity X')
    plt.subplot(2, 3, 4)
    plt.plot(times, states[:, 3], label='vy')
    plt.title('Velocity Y')
    plt.subplot(2, 3, 5)
    plt.plot(times, states[:, 4] * 180 / np.pi, label='θ')
    plt.title('Posture θ')
    plt.subplot(2, 3, 6)
    plt.plot(times, states[:, 5], label='ω')
    plt.title('Angular Velocity ω')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()