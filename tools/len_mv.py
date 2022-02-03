import cv2, os

class VideoGenerator:
    def __init__(self, width=1920, height=1080, step=300):
        self.width = width
        self.height = height
        self.step = step

    def linear(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, c = image.shape
        for i in range(self.step):
            dh = (h - self.height) // 2
            dw = int((w - self.width) / self.step * i)
            out = image[dh:dh + self.height, dw:dw + 1920, :]
            video_writer.write(out)

    def center_zoomout(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        for i in range(self.step):
            ch = self.height + int((h - self.height) / self.step * i)
            cw = self.width + int((w - self.width) / self.step * i)
            dh = (h - ch) // 2
            dw = (w - cw) // 2
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

            video_writer.write(out)

    def left_zoomout(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        scale_h = h / self.height
        scale_w = w / self.width
        largest_scale = min(scale_h, scale_w)
        scale_step = (largest_scale - 1) / self.step
        largest_w = w - self.width * largest_scale
        w_step = largest_w / self.step
        for i in range(self.step):
            cscale = scale_step * i + 1.0
            ch = int(self.height * cscale)
            cw = int(self.width * cscale)
            dh = (h - ch) // 2
            dw = int(i * w_step)
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

            video_writer.write(out)

    def move_right_zoomin(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        tar_scale = 0.8
        scale_step = (tar_scale - 1.0) / self.step
        tar_width = self.width * tar_scale
        #tar_height = 1080 * tar_scale
        w_step = (w - tar_width) / self.step

        for i in range(self.step):
            ch = int(self.height * (1.0 + scale_step * i))
            cw = int(self.width * (1.0 + scale_step * i))
            dh = (h - ch) // 2
            dw = int(i * w_step)
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            #print(ch, cw, dh, dw, dw + cw)

            video_writer.write(out)
        print(video_path)

if __name__ == '__main__':
    gen = VideoGenerator(step=256)
    #gen.linear()
    #gen.center_zoomout('/Users/vici/alis1023.png', '/Users/vici/PycharmProjects/diffuse/czoomout.avi')
    #gen.left_zoomout('/Users/vici/alis1023.png', '/Users/vici/PycharmProjects/diffuse/lzoomout.avi')
    #gen.move_right_zoomin('/Users/vici/alis1023.png', '/Users/vici/PycharmProjects/diffuse/movezoomin.avi')
    src_root = 'out1_pick'
    tar_root = 'movie_rightzoomout'
    import numpy as np
    files = os.listdir(src_root)
    i = 0
    part = 1
    num = len(files)//3 + 1
    for f in files:
        #rd = np.random.randint(
        if i % num == 0:
            tar_dir1 = tar_root + '_part{}'.format(part)
            os.makedirs(tar_dir1, exist_ok=True)
            part += 1
        
        gen.move_right_zoomin(os.path.join(src_root, f), os.path.join(tar_dir1, os.path.splitext(f)[0] + '.avi'))
        #gen.left_zoomout(os.path.join(src_root, f), os.path.join(tar_dir1, os.path.splitext(f)[0] + '.avi'))
        #gen.linear(os.path.join(src_root, f), os.path.join(tar_dir1, os.path.splitext(f)[0] + '.avi'))
        i += 1
        print(i)

