import numpy as np
from typing import Tuple
import cv2

def rgb_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converts RGB frame to grayscale using luminosity method"""
    R,G,B= frame[:,:,0],frame[:,:,1],frame[:,:,2]
    # Convert RGB to grayscale using standard luminosity weights.
    Z = 0.2126 * R + 0.7152 * G + 0.0722 * B
    # Return the result as uint8 for memory efficiency.
    return(Z.astype(np.uint8))

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """Resizes frame using cv2.INTER_AREA interpolation"""
    # Use OpenCV to resize the frame to the target dimensions.
    resized= cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    # INTER_AREA interpolation works well for downscaling.
    return resized

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalizes pixel values to [0.0, 1.0] range"""
    # Scale pixel values from [0, 255] to [0.0, 1.0] range.
    frame= frame.astype(np.float32)
    normalized= frame/255
    # Convert to float32 for neural network compatibility.
    return normalized.astype(np.float32)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Chains grayscale, resize, and normalize operations"""
    # Combine all preprocessing steps into a single pipeline.
    # Apply transformations in the correct order for optimal results.
    Z= rgb_to_grayscale(frame)
    resized= resize_frame(Z)
    normalized= normalize_frame(resized)
    return normalized




from collections import deque

class FrameStack:
    def __init__(self, maxlen: int = 4):
        """Initialize deque with maxlen capacity"""
        # Set up a data structure to store frames with automatic size management.
        self.deque_of_frames =deque(maxlen=maxlen)
        # Store the maximum length for reference in other methods.
        self.maxlen=maxlen
        self.frame_shape=None
    
    def push(self, frame: np.ndarray) -> None:
        """Add preprocessed frame to deque"""
        if self.frame_shape is None:
            self.frame_shape = frame.shape
        self.deque_of_frames.append(frame)
        # Add the new frame to the collection.
        # The data structure should automatically handle overflow.
    
    def get_stack(self) -> np.ndarray:
        """Return stacked frames as (maxlen, height, width)"""
        # Create a properly shaped array containing all frames.
        current_frames = list(self.deque_of_frames)
        # Handle cases where fewer frames are available by padding with zeros.
        padding_needed = self.maxlen - len(current_frames)
        if padding_needed > 0:
            zeros = [np.zeros(self.frame_shape, dtype=np.float32) for _ in range(padding_needed)]
            current_frames = zeros + current_frames
        # Return frames in channel-first format for CNN input.
        return(np.stack(current_frames,axis=0))
    
    def reset(self) -> None:
        """Clear the deque"""
        self.deque_of_frames.clear()
        self.frame_shape = None
        # Remove all stored frames to start fresh.
