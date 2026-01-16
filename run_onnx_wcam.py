####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import onnxruntime as ort
import numpy as np
import argparse
import json
import os
import time
import cv2


def preprocess_image(image,img_size=224):
    """
    Preprocesses a PIL image for the ONNX model
    """        
    image = cv2.resize(image,(img_size, img_size))

    # To tensor and normalize
    image_np = np.array(image, dtype=np.float32)
    image_np = image_np / 255.0
    image_np = (image_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225],dtype=np.float32)
    
    # HWC to CHW
    image_np = image_np.transpose(2, 0, 1)
    
    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np


def generate_caption(image, onnx_session, word2idx, idx2word, max_len=100, img_size=224):
    
    image_np = preprocess_image(image,img_size=img_size).astype(np.float32)
    caption = [word2idx['<start>']]

    for _ in range(max_len):
        caption_tensor = np.array(caption, dtype=np.int64).reshape(1, -1)

        ort_inputs = {
            "images": image_np,
            "captions": caption_tensor
        }
        
        output = onnx_session.run(None, ort_inputs)[0]
        
        predicted = np.argmax(output, axis=2)[:, -1].item()
        caption.append(predicted)
        
        if predicted == word2idx['<end>']:
            break
            
    caption_words = [idx2word[str(idx)] for idx in caption]
    return " ".join(caption_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for images using a trained ONNX model.')
    parser.add_argument('--model_path', type=str, default='sic.onnx', help='Path to the ONNX model.')
    parser.add_argument('--image', type=str, default='info/image2.png', help='Paths to the images to caption.')
    parser.add_argument('--voc_path', type=str, default='voc.json', help='Path to vocabulary json file.')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size, usually 224x224')
    parser.add_argument('--camid', type=int, default=2, help='Camera input ID, default 0.')
    args = parser.parse_args()

    # get script dir
    script_path = os.path.dirname(__file__)
    script_path +="/"
    print(script_path)

    # Load vocabulary
    with open(script_path + args.voc_path, 'r') as f:
        idx2word = json.load(f)
    
    word2idx = {word: int(idx) for idx, word in idx2word.items()}
    
    # Configure ONNX session to use GPU if available
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        provider = 'CUDAExecutionProvider'
    else:
        provider = 'CPUExecutionProvider'

    onnx_session = ort.InferenceSession(script_path + args.model_path, providers=[provider])

    # run captioning
    #try:
    cap = cv2.VideoCapture(args.camid)
    while True:
        #image = Image.open(script_path + args.image).convert("RGB")
        r,image = cap.read()
        t = time.time()
        caption = generate_caption(image, onnx_session, word2idx, idx2word, img_size=args.img_size)
        exec_time = time.time() - t
        print(f"Caption for {script_path + args.image}: {caption}, {exec_time:.2f}s")
    
        k = cv2.waitKey(1)

        # on ESC key press exit
        if k == 27:
            exit(0)

        #cv2.imshow("sic",image)


    #except Exception as e:
    #    print(f"Could not process {script_path + args.image}: {e}")
