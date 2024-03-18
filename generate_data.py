import numpy as np
import cn_clip.clip as clip
import torch

texts = clip.tokenize([
    "穿红色衣服的男人",
    "坐在椅子上的高跟鞋的小女孩",
    "一只黄色的猫咪坐在绿色的车上",
    "吃披萨的狗和吃汉堡包的人",
    "正在打电话的人",
    "开心的小女孩",
    "背包女正在牵着狗散步",
    "小男孩在看电视",
    "奔跑中的老虎",
    "一条蛇在下水道",
    "好多泰迪熊玩偶坐在阳台围墙上",
    "一男一女正在床上肉搏",
    "沙漠中的骆驼与老鹰",
    "一地的垃圾，塑料瓶，果皮等",
    "天空中的飞机",
    "一群人在喝咖啡",
    ]).to(torch.int32).numpy()

for idx,text in enumerate(texts):
    text = text.reshape(1,-1)
    np.save("quant_data/%02d"%idx,text)