####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import argparse
import json
from PIL import Image
from utils.decoder import ImageCaptioner
from utils.vision_encoder import CNNEncoder
import os 

#import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def model_to_onnx(model_path, onnx_path, voc_path,quantization, opset_version):
    # Load vocabulary
    try:
        with open(voc_path, 'r') as f:
            voc = json.load(f)
    except:
        print ("No voc.json found, exiting...")
        exit(-1)
    voc_size = len(voc)

    # Load model
    model = ImageCaptioner(
        dim=768,
        num_layers=6,
        num_heads=8,
        vocab_size=voc_size,
        max_len=100,
        VisionEncoder=CNNEncoder,
        vis_out_dimension=512,
        vis_hxw_out=49
    )
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=False)
    model.eval()

    # Create dummy input
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_caption = torch.zeros(1, 100).long()

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_caption),
        onnx_path,
        input_names=['images', 'captions'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'captions': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size', 1: 'seq_len'}
        },
        opset_version=opset_version
    )

    print(f"Model saved to {onnx_path}")

    if quantization:
        quantized_model_path = onnx_path.replace('.onnx', '_quantized.onnx')
        quantize_dynamic(
            onnx_path,
            quantized_model_path,
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model saved to {quantized_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a trained model to ONNX format.')
    parser.add_argument('--model_path', type=str, default='sic_model.pth', help='Path to the trained PyTorch model.')
    parser.add_argument('--voc_path', type=str, default='voc.json', help='Path to the vocabulary file.')
    parser.add_argument('--onnx_path', type=str, default='sic.onnx', help='Path to save the ONNX model.')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic quantization to the ONNX model.')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version.')
    args = parser.parse_args()

    # get script dir
    script_path = os.path.dirname(__file__)
    script_path +="/"
    print(script_path)

    model_to_onnx(model_path=script_path + args.model_path, 
                  onnx_path=script_path + args.onnx_path, 
                  voc_path= script_path + args.voc_path,
                  quantization=args.quantize, opset_version=args.opset_version)
    print ("Done.")
