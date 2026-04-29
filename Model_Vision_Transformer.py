import torch
# import torchvision.transforms as transforms
from PIL import Image
# from vit_pytorch import ViT

from Evaluation import evaluation


def Model_Vision_Transformer(image, Activation_Function):
    # Load pre-trained Vision Transformer model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )


    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Extract features using the Vision Transformer model
    with torch.no_grad():
        features = model(input_tensor)

    return features

def Model_ViT(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function):
    # Load pre-trained Vision Transformer model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_Data, Train_Target, steps_per_epoch=100, epochs=50, validation_data=(Test_Data, Test_Target))

    pred = model.predict(Test_Data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Test_Target)
    return Eval, pred
