import numpy as np
from collections import deque


class FrameStacker:
    def __init__(self, stack_size=4, frame_shape=(84, 84)):
        self.stack_size = stack_size
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_frame):
        # Fill the deque with initial_frame repeated
        for _ in range(self.stack_size):
            self.frames.append(initial_frame)
        pass

    def append(self, frame):
        # Add a new frame to the deque
        self.frames.append(frame)
        pass

    def get_stacked_state(self):
        # Stack frames along the last axis → shape: (84, 84, 4)
        np.stack(self.frames, axis=-1)
        pass
