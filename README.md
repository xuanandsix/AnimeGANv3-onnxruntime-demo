# AnimeGANv3-onnxruntime-demo
This is the onnxruntime inference code for AnimeGANv3. Official code: https://github.com/TachibanaYoshino/AnimeGANv3 <br>

It includes face detection (yolov5-face，or any other way) to corp face for portrait stylization. 

## test portrait sketch
```
python demo_onnx.py --useFace
```
It also works well for panoramic portraits. Without use face detection.
```
python demo_onnx.py
```
| input | useFace| without useFace|
| :-: |:-:| :-:|
|<img src="https://github.com/xuanandsix/AnimeGANv3-onnxruntime-demo/raw/main/portrait.jpg" >|<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_onnx2.png" >||<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_onnx1.png" >||

