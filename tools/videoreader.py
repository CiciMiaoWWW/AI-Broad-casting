import av

class VideoReader:
    def __init__(self, path):
        self.container = av.container.open(path)
        self.position = 0
        self.stream = self.container.streams.video[0]
        self.total_frames = self.stream.frames
        self.seek(0)

    def iter_frames(self):
        for packet in self.container.demux(self.stream):
            if packet.dts is None:
                continue
            for frame in packet.decode():
                yield frame

    def close(self):
        self.container.close()

    def read(self):
        try:
            frame = next(self.iter)
        except StopIteration:
            self.end = True
            return None
        self.position += 1
        return frame.to_rgb().to_ndarray(format='bgr24')

    def seek(self, frame):
        search_range = 300 # keyframe intervals
        pts = int(frame * self.stream.duration / self.stream.frames)
        self.container.seek(pts, stream=self.stream)
        for j, f in enumerate(self.iter_frames()):
            if j > search_range:
                raise RuntimeError('Did not find target within', search_range, 'frames of seek')
            if f.pts >= pts - 1:
                break
        self.end = False
        self.position = frame
        self.iter = iter(self.iter_frames())