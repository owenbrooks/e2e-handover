import cv2
import numpy as np
import rembg

class Segmentor():
    def __init__(self):
        self.ort_session = rembg.detect.ort_session('u2netp') # u2netp is the smaller model compared to u2net

    def inference(self, im):
        fg_mask = rembg.remove((im*255).astype(np.uint8), only_mask=True, session=self.ort_session)
        bg_removed = im.copy()
        bg_removed[fg_mask == 0, :] = 255 # set background to white

        return bg_removed

if __name__ == "__main__":
    im = cv2.imread("input.png")
    seg = Segmentor()
    result = seg.inference(im)

    while True:
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break