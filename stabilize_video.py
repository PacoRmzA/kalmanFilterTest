import numpy as np
import cv2

def smooth(trajectory, radius):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = np.convolve(trajectory[:,i], np.ones(radius)/radius, mode='same')
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 30% without moving the center (no black outline)
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.3)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

cap = cv2.VideoCapture('video_cel_kalman.mp4')
 
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # video codec
 
out = cv2.VideoWriter('video_cel_kalman_stabilized.mp4', fourcc, fps, (w, h))

# Read first frame
_, prev = cap.read() 
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


transforms = np.zeros((n_frames-1, 3), np.float32) 
 
for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=250,
                                     qualityLevel=0.1,
                                     minDistance=30)
 
  success, curr = cap.read()
  if not success:
    break
 
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
 
  # Track feature points
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
 
  assert prev_pts.shape == curr_pts.shape 
 
  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]
 
  #Find transformation matrix
  m,_ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
  dx = m[0][2]
  dy = m[1][2] 
  da = np.arctan2(m[1][0], m[0][0])
 
  transforms[i] = [-dx,-dy,-da] # minus signs keep scene static instead of stabilizing cam
  prev_gray = curr_gray

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  success, frame = cap.read()
  if not success:
    break
 
  dx = trajectory[i,0]
  dy = trajectory[i,1]
  da = trajectory[i,2]
 
  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
 
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
 
  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 
 
  cv2.imshow("Before", frame)
  cv2.imshow("After", frame_stabilized)
  cv2.waitKey(10)
  out.write(frame_stabilized)