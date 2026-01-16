# Small image captioning transformer 

A small yet computationally efficient image captioning model with fast token processing. There is an image and a text encoding part that is fused together using a cross-attention module [2]. The captions are generated in an autoregressive manner.

![architecture](/info/image.png)


The repository holds utility functions to export the traned captiong model to onnx, then to execute it.
```code
model2onnx.py - convert model to onnx (opset selectable see arguments with --help)
run_onnx.py - executes the model on a single picture (see arguments)
run_onnx_wcam.py - executes the script using the webcam (cam id selectable over command line arguments).
```

Traing dataset is the COCO captioning dataset [1].

### How does it work?

During the traning process a vocabulary will be generated, that is needed to decode correctly the text. Call the ```generate_caption``` function, provide an image and the model to the input, then the model will generate a list of captions, as shown in the images below.
```code
test_model = model = ImageCaptioner(vocab_size=len(voc),
                       dim=768, 
                       num_heads=8,
                       num_layers=4, 
                       vis_out_dimension=512, 
                       vis_hxw_out=49,
                       max_len = MAX_LEN,
                       VisionEncoder=CNNEncoder).to(device)

test_model.load_state_dict(torch.load('./sic_model.pth', map_location=device), strict=False)

transform_nn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

cap = generate_caption(test_model,image,voc,50, return_att=return_att)
print (cap[0])
```


![kitchen](info/image1.png)

![flowers](info/image2.png)

### Model complexity

|Backbone|FLOPS[G]|Params[M]|
|---|---|---|
|CNNEncoder (resnet18 based)|6.41|39.31|
|VitEncoder (Vision Transformer)|11.46|87.40|

```Note: With the hybrid method (CNN + transformer), the model is small and fast, but has some limitations in capturing all the scene dependencies. Using a ViT, for a higher computational cost, better accuracy can be achieved.```

### References
[1. COCO - Common Objects Context](https://cocodataset.org/#home)  
[2. Attention Is All You Need](https://arxiv.org/abs/1706.03762)

/Enjoy.
