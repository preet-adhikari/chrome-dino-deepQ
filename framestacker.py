import numpy as np
from collections import deque


class FrameStacker:
    def __init__(self, stack_size=4, frame_shape=(75, 75)):
        self.stack_size = stack_size
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_frame):
        # Fill the deque with initial_frame repeated
        self.frames = deque(
            [initial_frame] * self.stack_size, maxlen=self.stack_size)

    def append(self, frame):
        # Add a new frame to the deque
        self.frames.append(frame)
        pass

    def get_stacked_state(self):
        # Stack frames along the last axis → shape: (75, 75, 3)
        return np.dstack(self.frames)
        pass
