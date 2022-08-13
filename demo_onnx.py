import os
import sys
import cv2
import numpy as np
import timeit
import onnxruntime
import argparse

class Yolov5Face():
    def __init__(self, model_path):
        anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.objThreshold = 0.5

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # ORT_ENABLE_EXTENDED ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL # ORT_PARALLEL ORT_SEQUENTIAL
        self.ort_session = onnxruntime.InferenceSession(model_path, sess_options=so)   

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
       
        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[15]
            if detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = detection[5:15] * np.tile(np.float32([ratiow,ratioh]), 5)
                landmarks.append(landmark.astype(np.int32))
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        rect = []
        for i in indices:
            #i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            landmark = landmarks[i]
            #print(left, top, left + width, top + height)
            rect.append([left, top, left + width, top + height])
            #frame = 
            #self.drawPred(frame, confidences[i], left, top, left + width, top + height, landmark)
        return rect
    def drawPred(self, frame, conf, left, top, right, bottom, landmark):
        #print((left, top), (right, bottom))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        cv2.imwrite('detect.png',frame)
        return frame #
      # Expand the area around the detected face by margin {ratio} pixels
    def margin_face(self, box, img_HW, margin=0.5):
        x1, y1, x2, y2 = [c for c in box]
        w, h = x2 - x1, y2 - y1
        new_x1 = max(0, x1 - margin*w)
        new_x2 = min(img_HW[1], x2 + margin * w)
        x_d = min(x1-new_x1, new_x2-x2)
        new_w = x2 -x1 + 2 * x_d  
        new_x1 = x1-x_d
        new_x2 = x2+x_d

        # new_h = 1.25 * new_w   
        new_h = 1.0 * new_w   

        if new_h>=h:
            y_d = new_h-h  
            new_y1 = max(0, y1 - y_d//2)
            new_y2 = min(img_HW[0], y2 + y_d//2)
        else:
            y_d = abs(new_h - h) 
            new_y1 = max(0, y1 + y_d // 2)
            new_y2 = min(img_HW[0], y2 - y_d // 2)
        return list(map(int, [new_x1, new_y1, new_x2, new_y2]))
    
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)

        ort_inputs = {self.ort_session.get_inputs()[0].name: blob}
        ort_outs = self.ort_session.run(None, ort_inputs)
        outs = ort_outs[0]
        outs[..., [0,1,2,3,4,15]] = 1 / (1 + np.exp(-outs[..., [0,1,2,3,4,15]]))   ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)
            
            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length, 5:7] * a_g_i + g_i * int(self.stride[i])   # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind + length, 7:9] * a_g_i + g_i * int(self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind + length, 9:11] * a_g_i + g_i * int(self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind + length, 11:13] * a_g_i + g_i * int(self.stride[i])  # landmark x4 y4
            outs[row_ind:row_ind + length, 13:15] = outs[row_ind:row_ind + length, 13:15] * a_g_i + g_i * int(self.stride[i])  # landmark x5 y5
            row_ind += length
        
        left, top, right, bottom = self.postprocess(srcimg, outs)[0]
        [left, top, right, bottom] = self.margin_face([left, top, right, bottom],srcimg.shape[:2])
        return [left, top, right, bottom]


class AnimeGANv3:
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def process_image(self, img, x32=True):
        h, w = img.shape[:2]
        ratio = h/w
        if x32: # resize image to multiple of 32s
            def to_32s(x):
                return 256 if x < 256 else x - x%32
            new_h = to_32s(h)
            new_w = int(new_h/ratio) - int(new_h/ratio)%32
            img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
        return img

    def forward(self, img):
        img = self.process_image(img)
        img = np.float32(img[np.newaxis,:,:,:])
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        output_image = (np.squeeze(output) + 1.) / 2 * 255
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--useFace', action='store_true', help='use face detection inference')
    args = parser.parse_args()
    face = Yolov5Face('./models/yolov5s-face.onnx')

    anime = AnimeGANv3('./models/AnimeGANv3_PortraitSketch.onnx')
    img = cv2.imread('portrait.jpg')
    if args.useFace:
        [left, top, right, bottom] = face.detect(img)
        img = img[top:bottom, left:right, :]
    output = anime.forward(img)
    cv2.imwrite('output_onnx.png', output)

# python demo_onnx.py --useFace