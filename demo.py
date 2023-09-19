import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models


class FeatureMatMul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_per_image, logits_per_text):
        logit_scale = 100  # self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image.softmax(1), logits_per_text.softmax(1)


print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name(
    "ViT-B-16", device=device, download_root='./weights')
model.eval()
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
matmul = FeatureMatMul()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.cpu().numpy()
    # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
    print("Label probs:", probs)

    logits_per_image, logits_per_text = matmul(image_features, text_features)
    probs = logits_per_image.cpu().numpy()
    # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
    print("Label probs:", probs)
