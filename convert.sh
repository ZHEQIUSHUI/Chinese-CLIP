 python cn_clip/deploy/pytorch_to_onnx.py --convert-text --convert-vision --model-arch RN50 --pytorch-ckpt-path weights/clip_cn_rn50.pt --save-onnx-path onnx_models/r50
 python cn_clip/deploy/pytorch_to_onnx.py --convert-text --convert-vision --model-arch ViT-B-16 --pytorch-ckpt-path weights/clip_cn_vit-b-16.pt --save-onnx-path onnx_models/vitb16 
 python cn_clip/deploy/pytorch_to_onnx.py --convert-text --convert-vision --model-arch ViT-L-14 --pytorch-ckpt-path weights/clip_cn_vit-l-14.pt --save-onnx-path onnx_models/vitl14 
 python cn_clip/deploy/pytorch_to_onnx.py --convert-text --convert-vision --model-arch ViT-L-14-336 --pytorch-ckpt-path weights/clip_cn_vit-l-14-336.pt --save-onnx-path onnx_models/vitl14-336
#  python cn_clip/deploy/pytorch_to_onnx.py --convert-text --convert-vision --model-arch ViT-H-14 --pytorch-ckpt-path weights/clip_cn_vit-h-14.pt --save-onnx-path onnx_models/vith14