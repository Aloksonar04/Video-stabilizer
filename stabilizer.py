import cv2
import numpy as np

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = []

    for _ in range(n_frames - 1):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m = cv2.estimateAffine2D(prev_pts, curr_pts)[0]

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms.append([dx, dy, da])
        prev_gray = curr_gray

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = smooth(trajectory)

    diff = smoothed - trajectory
    transforms_smooth = transforms + diff

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(len(transforms)):
        success, frame = cap.read()
        if not success:
            break
        dx, dy, da = transforms_smooth[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ])
        stabilized = cv2.warpAffine(frame, m, (w, h))
        out.write(stabilized)

    cap.release()
    out.release()

def smooth(trajectory, radius=30):
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed

def moving_average(curve, radius):
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    curve_padded = np.pad(curve, (radius, radius), mode='edge')
    return np.convolve(curve_padded, kernel, mode='same')[radius:-radius]
