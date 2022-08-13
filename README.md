# AnimeGANv3-onnxruntime-demo
This is the onnxruntime inference code for AnimeGANv3. Official code: https://github.com/TachibanaYoshino/AnimeGANv3 <br>

It includes face detection (yolov5-face，or any other way) to corp face for portrait stylization. 

## test onnxruntime demo 
For example, portrait sketch. For other style, just replace the onnx model.
```
python demo_onnx.py --useFace
```
It also works well for panoramic image. Without use face detection.
```
python demo_onnx.py
```
| input | useFace| panoramic image|
| :-: |:-:| :-:|
|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/raw/main/portrait.jpg" height="60%" width="60%">|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/blob/main/imgs/output_onnx.png" height="60%" width="60%">|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/raw/main/imgs/output_onnx1.png" height="60%" width="60%">|
|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/raw/main/body.jpg" height="60%" width="60%">|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/blob/main/imgs/output_onnx3.png" height="60%" width="60%" >|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/raw/main/imgs/output_onnx2.png" height="60%" width="60%">|


