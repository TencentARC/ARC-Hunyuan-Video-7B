import av
import numpy as np


class VideoReaderAV:
    def __init__(self, video_path, num_threads=1, width=None, height=None):
        """
        Initializes the video reader.

        Args:
            video_path (str): Path to the video file.
            resize_width (int, optional): The target width for resizing. If None, aspect ratio is maintained based on height.
            resize_height (int, optional): The target height for resizing. If None, aspect ratio is maintained based on width.
        """
        self.video_path = video_path
        self.container = av.open(video_path)
        
        video_stream = self.container.streams.video[0]
        video_stream.thread_type = "FRAME"
        video_stream.thread_count = num_threads

        self.video_fps = video_stream.average_rate
        self.time_base = video_stream.time_base

        self.width, self.height = self.get_width_height(width, height)

        self.video_num_frames = 0

        if video_stream.duration is not None:
            duration_in_seconds = video_stream.duration * self.time_base
            self.video_num_frames = int(duration_in_seconds * self.video_fps)

        # Fallback for videos that are missing duration
        if self.video_num_frames == 0:
            self.video_num_frames = video_stream.frames
        
        # Fallback for videos that has corrupted meta, i.e. duration is None and frames is 0
        if self.video_num_frames == 0:
            self.video_num_frames = self.iter_video_num_frames()

    def iter_video_num_frames(self):
        frame_count = 0
        for frame in self.container.decode(video=0):
            frame_count += 1
        return frame_count
    
    def get_width_height(self, width=None, height=None):
        original_width = self.container.streams.video[0].width
        original_height = self.container.streams.video[0].height
        if width is None and height is None:
            return original_width, original_height
        elif width is None:
            width = int(original_width * (height / original_height))
            return width, height
        elif height is None:
            height = int(original_height * (width / original_width))
            return width, height
        else:
            return width, height

    def get_avg_fps(self):
        return self.video_fps

    def __len__(self):
        return self.video_num_frames
    
    def __del__(self):
        self.close()
    
    def close(self):
        try:
            self.container.close()
        except Exception:
            pass
    
    def __getitem__(self, idx):
        pts = int(idx / self.video_fps / self.time_base)
        self.container.seek(pts, any_frame=False, backward=True, stream=self.container.streams.video[0])
        for i, frame in enumerate(self.container.decode(video=0)):
            if frame.pts >= pts:
                #print(frame.height, frame.width)
                if frame.width % 2 != 0 or frame.height % 2 != 0:
                    new_width = (frame.width // 2) * 2
                    new_height = (frame.height // 2) * 2
                    frame = frame.reformat(width=new_width, height=new_height)
                frame = frame.reformat(format='rgb24')
                return frame.to_ndarray()
        return None
    
    def get_batch(self, sampled_indices):
        sampled_pts = [int(indice / self.video_fps / self.time_base) for indice in sampled_indices]
        frames = []
        
        # To avoid seeking for every frame if they are close, we can be smarter.
        # This basic implementation seeks for each frame as in the original code.
        for pts in sampled_pts:
            self.container.seek(pts, any_frame=False, backward=True, stream=self.container.streams.video[0])
            frame_found = False
            for i, frame in enumerate(self.container.decode(video=0)):
                if frame.pts >= pts:
                    if self.width is not None and self.height is not None:
                        resized_frame = frame.reformat(width=self.width, height=self.height, format='rgb24')

                    frames.append(resized_frame.to_ndarray())
                    
                    frame_found = True
                    break

                # Safety break to prevent infinite loops in case of decoding issues
                if i > 1000:
                    print(f"Warning: Could not find frame near PTS {pts} after 1000 attempts.")
                    break
            
            if not frame_found:
                # If a frame is not found, it might be the end of the video or a corrupted part.
                # We can either raise an error or, for robustness, append the last good frame.
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    # This case is tricky. If the very first frame isn't found, we might need to
                    # either raise an error or return an empty array, depending on desired behavior.
                    # Here we stop processing for this batch.
                    break
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be decoded for indices: {sampled_indices}")
            
        # If we couldn't find all requested frames (e.g., seeking past EOF)
        # fill the rest with the last available frame, consistent with Decord.
        if len(frames) < len(sampled_indices):
            frames.extend([frames[-1]] * (len(sampled_indices) - len(frames)))

        return np.array(frames)

