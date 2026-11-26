# VLA基础

## 零、先理解我们要做什么

我们的目标：让电脑学会"看图 + 听指令 → 输出机械臂该怎么动"。

就像一个学生：

· 看：一张图，上面有红色方块和蓝色末端
· 听：一句英文指令 "move to the red cube"
· 做：输出 6 个数字，代表机械臂该怎么移动

我们要写 5 个文件，分别负责 5 件事：

文件 作用 类比
generate_data.py 造训练数据 编教材
dataset.py 把数据喂给模型 把教材拆成一页页
model.py 定义模型结构 学生的大脑
train.py 训练模型 上课学习
inference.py 用模型预测 考试答题

---

## 一、generate_data.py —— 造数据

思路：我们没有真实机械臂数据，那就用代码"画"一批假数据。每张图就是白底 + 红方块 + 蓝圆圈。动作就是"朝方块方向移动"。

### 1.导入

```python
import h5py
import numpy as np
import os
from PIL import Image, ImageDraw
```

逐行：

· import h5py：导入一个工具库，用来读写 h5 文件。h5 是一种专门存大数据的文件格式，像"科学界的 Excel"。
· import numpy as np：导入数值计算库。np 是它的外号，后面写 np.xxx 就是用它。它擅长处理数组（一堆数字）。
· import os：导入操作系统工具，用来创建文件夹、拼路径。
· from PIL import Image, ImageDraw：从 PIL 库导入两个工具：
  · Image：处理图像（打开、保存、转换）。
  · ImageDraw：在图像上画东西（画方块、画圆）。

为什么用这些：造数据需要画图（PIL）、存数字（numpy）、写文件（h5py）、建文件夹（os）。

---

### 2.画一帧图像

```python
def render_frame(cube_xy, ee_xy, size=64):
```

· def：定义函数的关键字。函数就是"一段可以重复使用的代码"。
· render_frame：函数名，意思是"渲染一帧"。
· 括号里是参数，也就是这个函数需要的外部信息：
  · cube_xy：红色方块的位置，比如 [20, 30] 表示 x=20, y=30。
  · ee_xy：机械臂末端（蓝色圆圈）的位置。
  · size=64：图像大小，默认 64×64 像素。

思路：我们想画一张图，得知道方块在哪、末端在哪、图多大。

```python
    img = Image.new("RGB", (size, size), (255, 255, 255))
```

· Image.new(...)：新建一张空白图。
· "RGB"：颜色模式，R=红、G=绿、B=蓝三通道。
· (size, size)：图像宽高，都是 64。
· (255, 255, 255)：背景色，三通道都是 255 = 白色。

结果：img 是一张 64×64 的白色图片。

```python
    draw = ImageDraw.Draw(img)
```

· 创建一个"画笔"对象 draw，绑定到 img 上。
· 之后用 draw.rectangle(...) 就是在这张图上画。

为什么：PIL 里画图要通过 Draw 对象操作。

```python
    cx, cy = int(cube_xy[0]), int(cube_xy[1])
```

· cube_xy 是 [x, y]。
· cube_xy[0] 取第一个值 x，cube_xy[1] 取第二个值 y。
· int(...)：转成整数。因为画图坐标必须是整数。
· cx, cy = ...：同时给两个变量赋值。

```python
    draw.rectangle([cx-5, cy-5, cx+5, cy+5], fill=(255, 0, 0))
```

· 画一个矩形。
· [cx-5, cy-5, cx+5, cy+5]：左上角 (cx-5, cy-5)，右下角 (cx+5, cy+5)，所以是个 10×10 的方块，中心在 (cx, cy)。
· fill=(255, 0, 0)：填充红色。

```python
    ex, ey = int(ee_xy[0]), int(ee_xy[1])
    draw.ellipse([ex-3, ey-3, ex+3, ey+3], fill=(0, 0, 255))
```

· 同理，画一个蓝色圆圈代表机械臂末端。
· ellipse 是画椭圆，宽高相等时就是圆。
· (0, 0, 255)：蓝色。

```python
    return np.array(img, dtype=np.uint8)
```

· np.array(img, ...)：把 PIL 图像转成 numpy 数组。
· dtype=np.uint8：数据类型是 8 位无符号整数，范围 0~255。图像像素正好是这个范围。
· return：把结果返回给调用者。

为什么转 numpy：模型训练用的是数字数组，numpy 数组是标准格式。

最终结果：返回一个形状 (64, 64, 3) 的数组，代表一张彩色图。

---

### 3.生成一个 episode

名词：episode = 一段完整的演示。比如"机械臂从起点走到方块"整个过程。

```python
def generate_episode(seed, T=3, H=W=64):
```

· seed：随机种子。用来保证每次运行生成的数据一样（可复现）。
· T=3：时间步数，也就是这段演示有几帧。
· H=W=64：图像高和宽。

```python
    rng = np.random.RandomState(seed)
```

· 创建一个随机数生成器。
· 用 seed 初始化，保证同样的 seed 生成同样的随机数。
· rng 是外号。

为什么：我们要随机放方块，但希望结果可复现，所以固定种子。

```python
    cube_xy = rng.randint(15, 49, size=2).astype(np.float32)
```

· rng.randint(15, 49, size=2)：生成 2 个随机整数，范围 [15, 49)。
· .astype(np.float32)：转成 32 位浮点数。
· 结果比如 [20.0, 35.0]，代表方块位置。

为什么范围 15~49：64×64 的图，边缘留 15 像素，让方块不出界。

```python
    ee_xy = np.array([32.0, 32.0], dtype=np.float32)
```

· 末端初始位置固定在图像中心 (32, 32)。

```python
    instruction = "move to the red cube"
```

· 指令固定。真实项目里指令会千变万化，这里简化。

```python
    images, actions = [], []
```

· 建两个空列表，准备存每一帧的图像和动作。

思路：一段演示有多帧，每帧都要存图像和对应动作。

```python
    for step in range(T):
```

· 循环 T 次，step 从 0 到 T-1。
· 每次循环生成一帧。

```python
        frame = render_frame(cube_xy, ee_xy, size=H)
        images.append(frame)
```

· 调用 render_frame 画一帧。
· images.append(frame)：把这帧加到列表末尾。

```python
        delta = (cube_xy - ee_xy) * 0.3
```

· cube_xy - ee_xy：从末端指向方块的向量。
· * 0.3：缩小 30%，表示"只走一小步"。

思路：动作不是一步跳到方块，而是每步走 30% 的剩余距离，模拟平滑移动。

```python
        action = np.array([delta[0], delta[1], 0, 0, 0, 0], dtype=np.float32)
        actions.append(action)
```

· 构造 6 维动作：
  · 前两个是 x、y 方向的移动量。
  · 后四个是 0（z 方向、三个旋转，这里不涉及）。
· 加到动作列表。

为什么是 6 维：真实机械臂末端位姿通常用 6 个自由度表示：位置 (x, y, z) + 姿态 (roll, pitch, yaw)。我们这里只用前两个。

```python
        ee_xy = ee_xy + delta
```

· 更新末端位置，模拟"执行了动作"。

```python
    return {
        "image": np.stack(images),
        "instruction": instruction,
        "action": np.stack(actions),
        "action_mask": np.ones((T, 6), dtype=np.uint8),
    }
```

· np.stack(images)：把 T 张图堆成一个数组，形状 (T, 64, 64, 3)。
· np.stack(actions)：把 T 个动作堆成 (T, 6)。
· np.ones((T, 6))：全是 1 的 (T, 6) 数组，表示"所有动作维度都有效"。
· 返回一个字典，包含四样东西。

（这里instruction是字符串不能用np.stack）

为什么用字典：把相关的数据打包在一起，方便传递。

---

### 4.主函数：批量生成

```python
def main(out_dir="./data", n_episodes=200, T=3):
```

· out_dir：输出文件夹，默认 ./data。
· n_episodes=200：生成 200 个 episode。
· T=3：每个 episode 3 帧。

```python
    os.makedirs(out_dir, exist_ok=True)
```

· 创建文件夹 ./data。
· exist_ok=True：如果已经存在就不报错。

```python
    for i in range(n_episodes):
```

· 循环 200 次，i 从 0 到 199。

```python
        ep = generate_episode(seed=i, T=T)
```

· 调用上面的函数，生成第 i 个 episode。
· seed 用 i，保证每个 episode 不同但可复现。

```python
        path = os.path.join(out_dir, f"episode_{i:04d}.h5")
```

· f"episode_{i:04d}.h5"：格式化字符串。
  · i=5 → "episode_0005.h5"。
  · :04d 表示至少 4 位数字，不够补 0。
· os.path.join(...)：拼路径，自动加斜杠。
· 结果比如 "./data/episode_0005.h5"。

为什么补 0：文件排序时 0005 排在 0010 前面，方便。

```python
        with h5py.File(path, "w") as f:
```

· 打开一个 h5 文件，模式 "w" 是写入。
· with ... as f：用完后自动关闭文件，不用手动 f.close()。

```python
            f.create_dataset("image", data=ep["image"], compression="gzip")
```

· 在 h5 文件里创建一个数据集，叫 "image"。
· data=...：写入的数据。
· compression="gzip"：压缩存储，省空间。

```python
            f.create_dataset("instruction", data=ep["instruction"])
            f.create_dataset("action", data=ep["action"])
            f.create_dataset("action_mask", data=ep["action_mask"])
```

· 同样创建其他三个数据集。

```python
    print(f"生成 {n_episodes} 个 episode 到 {out_dir}")
```

· 打印完成信息。

```python
if __name__ == "__main__":
    main()
```

· 这是一个 Python 惯例：
  · __name__ 是当前模块的名字。
  · 直接运行这个文件时，__name__ 等于 "__main__"，所以执行 main()。
  · 如果被别的文件 import，就不执行。

为什么：让代码既能直接跑，又能被其他文件引用。

---

## 二、dataset.py —— 把数据喂给模型

思路：模型训练时要一批一批地拿数据，不能一次性加载所有。Dataset 负责"给定一个编号，返回一条样本"。

### 1。导入

```python
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
```

· glob：用来按通配符找文件，比如 episode_*.h5。
· torch：PyTorch 深度学习框架。
· Dataset：PyTorch 提供的基类，我们继承它。

---

### 2.定义类

```python
class VLAEpisodeDataset(Dataset):
```

· class：定义一个"类"，类是一种把数据和函数打包的方式。
· VLAEpisodeDataset：类名。
· (Dataset)：继承 PyTorch 的 Dataset，这样 DataLoader 才认识它。

思路：类的写法是 PyTorch 的规矩，必须继承 Dataset 并实现两个方法。

```python
    def __init__(self, pattern, T=3, image_size=64):
```

· __init__：初始化函数，创建对象时自动调用。
· pattern：文件通配符，比如 "./data/episode_*.h5"。
· T=3：历史帧数。
· image_size=64：图像尺寸。

```python
        self.files = sorted(glob.glob(pattern))
```

· glob.glob(pattern)：找出所有匹配的文件，返回列表。
· sorted(...)：排序，保证顺序稳定。
· self.files：存到对象上，后面能用。

为什么排序：不排序的话每次顺序可能不一样，影响复现。

```python
        self.T = T
        self.image_size = image_size
```

· 把参数存到 self 上，后面用。

```python
        self.index = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                n = f["action"].shape[0]
            for t in range(n):
                self.index.append((fi, t))
```

· self.index：准备一个索引列表。
· enumerate(self.files)：遍历文件，fi 是编号，path 是路径。
· h5py.File(path, "r")：只读模式打开。
· f["action"].shape[0]：动作数据集的第一个维度长度，即时间步数 n。
· 内层循环把 (文件编号, 时间步) 加到索引里。

思路：把所有 (哪个文件, 第几帧) 组合都列出来，这样 __getitem__ 就能根据编号快速定位。

为什么：数据分散在多个文件里，需要一个全局索引来统一编号。

```python
    def __len__(self):
        return len(self.index)
```

· __len__：让 len(dataset) 返回样本总数。

```python
    def __getitem__(self, i):
```

· __getitem__：让 dataset[i] 返回第 i 条样本。
· 这是 Dataset 必须实现的方法。

```python
        fi, t = self.index[i]
```

· 从索引里取出文件编号和时间步。

```python
        with h5py.File(self.files[fi], "r") as f:
```

· 打开对应的 h5 文件。

```python
            start = max(0, t - self.T + 1)
            idxs = list(range(start, t + 1))
            while len(idxs) < self.T:
                idxs.insert(0, idxs[0])
```

· 我们要取 T 帧历史。
· start = max(0, t-T+1)：从 t 往前数 T-1 帧，但不小于 0。
· idxs：从 start 到 t 的编号列表。
· 如果开头历史不够（比如 t=0 只有 1 帧），循环在开头重复第一帧。

.insert（m，n）是在序列第m位插入n这个数

思路：模型需要固定 T 帧输入。开头不足就重复第一帧，保证形状一致。

```python
            images = f["image"][idxs]
            instr = f["instruction"][()]
            if isinstance(instr, bytes):
                instr = instr.decode("utf-8")
```

· f["image"][idxs]：按索引取多帧，结果形状 (T, 64, 64, 3)。
· f["instruction"][()]：取标量字符串。[()] 是 h5py 的语法。
· isinstance(instr, bytes)：h5py 有时返回字节串。
· .decode("utf-8")：转成普通字符串。

为什么：h5py 存字符串可能返回 b"..."，模型需要普通字符串。

```python
            action = f["action"][t].astype(np.float32)
            mask = f["action_mask"][t].astype(np.float32)
```

· 取第 t 步的动作和掩码。
· .astype(np.float32)：转成 float32。

为什么转 float32：PyTorch 默认浮点类型是 float32，统一格式。

```python
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
```

· torch.from_numpy(images)：numpy 数组转 PyTorch 张量。
· .float()：转 float32。
· / 255.0：归一化到 [0, 1]。

为什么归一化：神经网络喜欢小数值，0~255 太大，训练不稳定。

· images.permute(0, 3, 1, 2)：重排维度。
  · 原形状 (T, 64, 64, 3)，含义 (时间, 高, 宽, 通道)。
  · 新形状 (T, 3, 64, 64)，含义 (时间, 通道, 高, 宽)。
· PyTorch 的卷积层要求通道在前。

```python
        return {
            "images": images,
            "instruction": instr,
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(mask),
        }
```

· 返回一条样本的字典。

---

### 3.collate_fn

名词：collate = 把多条样本"拼"成一个 batch。

```python
def collate_fn(batch):
```

· batch：DataLoader 传进来的列表，里面有 B 条样本。

为什么需要：默认的 collate 不会处理字符串，会报错。我们自己写一个。

```python
    images = torch.stack([b["images"] for b in batch])
    actions = torch.stack([b["action"] for b in batch])
    masks = torch.stack([b["action_mask"] for b in batch])
```

· [b["images"] for b in batch]：列表推导式，把每条样本的 images 取出来。
· torch.stack(...)：把它们堆成一个张量，形状 (B, T, 3, 64, 64)。

```python
    instructions = [b["instruction"] for b in batch]
```

· 字符串不能 stack，直接放进列表。

```python
    return {
        "images": images,
        "instructions": instructions,
        "actions": actions,
        "action_masks": masks,
    }
```

· 返回一个 batch 的字典。

为什么加 s：images vs image，表示"一批"而不是"一条"。

---

## 三、model.py —— 模型结构

### 1.思路：模型分三块：

1. 视觉塔：从图像提取特征。
2. 语言塔：从指令提取特征。
3. 融合 + 动作头：合并两者，输出动作。

1. 导入

```python
import torch
import torch.nn as nn
```

· nn：PyTorch 的神经网络模块，里面有各种层。

### 2.定义模型

```python
class TinyVLA(nn.Module):
    def __init__(self, action_dim=6, text_vocab=100, text_dim=32, hidden=128):
        super().__init__()
```

· TinyVLA 继承 nn.Module，这是所有模型的基类。
· super().__init__()：调用父类初始化，必须写。

参数：

· action_dim=6：输出动作维度。
· text_vocab=100：词表大小，最多 100 个不同的词。
· text_dim=32：每个词映射成 32 维向量。
· hidden=128：融合层的隐藏维度。

```python
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
```

· nn.Sequential(...)：把多个层按顺序串起来，输入依次经过。
· 第一层 nn.Conv2d(3, 16, 3, stride=2, padding=1)：
  · 卷积层，输入 3 通道（RGB），输出 16 通道。
  · 卷积核 3×3。
  · stride=2：步长 2，图像缩小一半。
  · padding=1：边缘补 1 圈 0，保持尺寸。
· nn.ReLU()：激活函数，负数变 0，正数不变，给网络非线性。
· 第二层：16 → 32 通道，再缩一半。
· nn.AdaptiveAvgPool2d(1)：把空间维压成 1×1，输出 (B, 32, 1, 1)。

思路：CNN 层层提取特征，最后把每张图变成一个 32 维向量。

为什么用 CNN：教学简化。真实 VLA 用 ViT。

```python
        self.text_embed = nn.Embedding(text_vocab, text_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)
```

· nn.Embedding(100, 32)：一张查找表，输入整数 ID，输出 32 维向量。
· nn.Linear(32, 32)：线性变换，给语。

思路：把词 ID 转成向量，再做一次变换。

```python
        self.fuse = nn.Sequential(
            nn.Linear(32 + text_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
```

· 输入是 32（视觉）+ 32（语言）= 64 维。
· 第一层 Linear：64 → 128。
· ReLU 激活。
· 第二层 Linear：128 → 6，输出动作。

思路：把融合后的特征逐步映射到动作空间。

---

### 3.语言编码

```python
    def encode_text(self, instructions):
        B = len(instructions)
        ids = torch.zeros(B, 8, dtype=torch.long, device=self.text_embed.weight.device)
```

· B：batch 大小，几条指令。
· ids：准备一个 (B, 8) 的全 0 张量，存每个词的 ID。
· device=...：和 embedding 在同一设备（CPU 或 GPU）。

为什么是 8：假设每句话最多 8 个词，短的补 0。

```python
        for i, s in enumerate(instructions):
            toks = s.lower().split()[:8]
```

· 遍历每条指令。
· s.lower()：转小写。
· .split()：按空格切分成词列表。
· [:8]：只取前 8 个词。

```python
            for j, tok in enumerate(toks):
                ids[i, j] = hash(tok) % self.text_embed.num_embeddings
```

· 遍历每个词。
· hash(tok)：Python 内置的哈希函数，把字符串转成整数。
· % self.text_embed.num_embeddings：取模，保证落在 [0, 99]。
· ids[i, j] = ...：写入 ID。

为什么用 hash：教学简化。真实项目用 tokenizer 查词表。缺点是不同词可能冲突。

```python
        emb = self.text_embed(ids).mean(dim=1)
        return self.text_proj(emb)
```

· self.text_embed(ids)：把 (B, 8) 的 ID 转成 (B, 8, 32) 的向量。
· .mean(dim=1)：对 8 个词取平均，得到 (B, 32)。

这一步就是"词袋"：平均抹掉了词序。

· self.text_proj(emb)：再做一次线性变换。

---

### 4.前向传播

```python
    def forward(self, images, instructions):
```

· forward：模型的核心函数，调用 model(images, instructions) 时自动执行。

```python
        B, T, C, H, W = images.shape
```

· 解包图像形状：batch、时间步、通道、高、宽。
· 对应 (B, T, 3, 64, 64)。

```python
        x = images.view(B * T, C, H, W)
```

· .view(...)：重塑形状，不改变数据。
· 把 (B, T, 3, 64, 64) 变成 (B*T, 3, 64, 64)。

为什么：CNN 一次处理一张图，所以把 B 和 T 合并。

```python
        feat = self.cnn(x).view(B, T, -1).mean(dim=1)
```

· self.cnn(x)：每张图过 CNN，输出 (B*T, 32, 1, 1)。
· .view(B, T, -1)：变回 (B, T, 32)。-1 表示自动算。
· .mean(dim=1)：对 T 帧取平均，得到 (B, 32)。

为什么平均：简化处理，真实模型用注意力。

```python
        txt = self.encode_text(instructions)
```

· 得到 (B, 32) 的语言特征。

```python
        h = torch.cat([feat, txt], dim=-1)
```

· torch.cat：拼接。
· dim=-1：在最后一维拼。
· 结果 (B, 64)。

这就是融合：视觉 32 维 + 语言 32 维 = 64 维。

```python
        pred = self.fuse(h)
        return {"pred_actions": pred}
```

· 过融合层，输出 (B, 6) 的动作预测。
· 返回字典，和真实 VLA 接口一致。

---

## 四、train.py —— 训练

思路：反复喂数据，让模型预测，比较预测和真实，调整参数让误差变小。

### 1.导入

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import VLAEpisodeDataset, collate_fn
from model import TinyVLA
```

· DataLoader：把 Dataset 包装成可迭代的 batch。
· AdamW：优化器，负责更新参数。
· CosineAnnealingLR：学习率调度器。
· 从自己的文件导入 Dataset 和 Model。

### 2.主函数

```python
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
```

· 有 GPU 就用 GPU，否则 CPU。
· 固定随机种子，保证可复现。

```python
    dataset = VLAEpisodeDataset("./data/episode_*.h5", T=3)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
```

· 创建数据集。
· DataLoader：
  · batch_size=16：每次取 16 条。
  · shuffle=True：打乱顺序。
  · num_workers=0：不用多进程（调试简单）。
  · collate_fn=collate_fn：用我们自定义的拼接函数。

```python
    model = TinyVLA().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=500)
```

· 创建模型，搬到 GPU。
· 优化器：学习率 0.001，权重衰减 0.01。
· 调度器：500 步内余弦衰减。

```python
    model.train()
```

· 切到训练模式，启用 dropout 等。

```python
    step = 0
    max_steps = 500
```

· step：当前训练步数。
· max_steps：最多训练 500 步。

```python
    for epoch in range(100):
        for batch in loader:
```

· 外层 epoch，内层遍历 batch。

```python
            images = batch["images"].to(device)
            instructions = batch["instructions"]
            actions = batch["actions"].to(device)
            masks = batch["action_masks"].to(device)
```

· 从 batch 取数据，搬到 GPU。
· 指令是字符串列表，不需要搬。

```python
            outputs = model(images, instructions)
            pred = outputs["pred_actions"]
```

· 前向传播，拿到预测。

```python
            loss = (torch.abs(pred - actions) * masks).sum() / masks.sum()
```

· torch.abs(pred - actions)：逐元素绝对误差。
· * masks：无效维度置 0。
· .sum()：求和。
· / masks.sum()：除以有效元素数，得到平均误差。

这就是 L1 损失 + mask。

```python
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
```

· zero_grad()：清空旧梯度。
· loss.backward()：反向传播，算每个参数的梯度。
· clip_grad_norm_：梯度裁剪，防止爆炸。
· optimizer.step()：用梯度更新参数。
· scheduler.step()：调整学习率。

```python
            if step % 50 == 0:
                print(f"step {step}, loss {loss.item():.4f}, lr {scheduler.get_last_lr()[0]:.2e}")
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break
```

· 每 50 步打印一次。
· 到 500 步就停。

```python
    torch.save(model.state_dict(), "tiny_vla.pt")
```

· 保存模型参数到文件。

```python
if __name__ == "__main__":
    main()
```

· 直接运行时执行 main。

---

## 五、inference.py —— 推理

思路：加载训练好的模型，给一张图 + 指令，输出动作。

### 1.导入

```python
import torch
import numpy as np
from PIL import Image
from model import TinyVLA
from generate_data import render_frame
```

· 从训练脚本里导入 render_frame，用来造测试图。

### 2.预测函数

```python
@torch.inference_mode()
def predict(model, images, instruction, device="cuda"):
```

· @torch.inference_mode()：装饰器，整个函数不建计算图，省显存加速。
· 参数：模型、图像数组、指令、设备。

```python
    model.eval()
    model.to(device)
```

· eval()：评估模式，关 dropout。
· to(device)：确保在 GPU。

```python
    x = torch.from_numpy(images).float() / 255.0
    x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)
```

· 转张量、归一化。
· .permute(0, 3, 1, 2)：(T, H, W, 3) → (T, 3, H, W)。
· .unsqueeze(0)：在开头加一维，变 (1, T, 3, H, W)，因为模型要 batch 维。
· .to(device)：搬到 GPU。

```python
    out = model(x, [instruction])
    return out["pred_actions"].squeeze(0).cpu().numpy()
```

· [instruction]：包成列表，因为模型期望一批指令。
· .squeeze(0)：去掉 batch 维，(1, 6) → (6,)。
· .cpu().numpy()：搬到 CPU 转 numpy。

### 3.主函数

```python
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVLA().to(device)
    model.load_state_dict(torch.load("tiny_vla.pt", map_location=device))
```

· 创建模型，加载训练好的参数。

```python
    cube = np.array([20, 20], dtype=np.float32)
    ee = np.array([40, 40], dtype=np.float32)
    frames = []
    for _ in range(3):
        frames.append(render_frame(cube, ee))
        ee = ee + (cube - ee) * 0.3
    frames = np.stack(frames)
```

· 造一个测试场景：方块在 (20,20)，末端在 (40,40)。
· 模拟末端朝方块移动 3 步，生成 3 帧历史。

```python
    action = predict(model, frames, "move to the red cube", device)
    print("预测动作:", action)
    print("期望方向: 朝方块，即 (-6, -6) 左右")
```

· 预测，打印。
· 期望动作指向方块方向。

```python
if __name__ == "__main__":
    main()
```

---

## 六、整体思路总结

```text
1. generate_data.py
   画图 → 存 h5
   目的：造出 (图像, 指令, 动作) 数据

2. dataset.py
   读 h5 → 取 T 帧 → 归一化 → 转张量
   目的：把数据整理成模型能吃的格式

3. model.py
   图像 → CNN → 视觉特征
   指令 → 词袋 → 语言特征
   拼接 → MLP → 动作
   目的：定义"看图+听话→动作"的函数

4. train.py
   反复前向 → 算损失 → 反向传播 → 更新参数
   目的：让模型预测越来越准

5. inference.py
   加载参数 → 输入图+指令 → 输出动作
   目的：用训练好的模型做预测
```

---

## 七、几个反复出现的概念

张量（Tensor）：多维数组，PyTorch 的基本数据结构。
(B, T, 3, 64, 64) 就是一个 5 维张量。

形状（shape）：每个维度的大小。
模型对输入形状有要求，形状不对就报错。

batch：一次处理多条样本。
batch_size=16 就是一次 16 条。

损失（loss）：预测和真实的差距。
训练就是让 loss 变小。

梯度（gradient）：loss 对参数的变化率。
反向传播算梯度，优化器用梯度更新参数。

前向 / 反向：

· 前向：输入 → 输出。
· 反向：loss → 梯度。

---

## 八、为什么这么设计

为什么不直接写一个函数算动作：

因为"看图 + 听话 → 动作"这个映射太复杂，手写规则写不出来。
所以让模型从数据里自己学。

为什么用 h5 存数据：

· 二进制，读写快。
· 支持压缩。
· 支持大数据集，不用一次加载到内存。

为什么归一化：

神经网络对输入尺度敏感。
像素 0~255 太大，归一化到 0~1 训练更稳。

为什么用 mask：

有些动作维度在某些时刻无效，mask 让它们不参与损失。

为什么梯度裁剪：

防止梯度爆炸，训练更稳。

为什么用余弦退火：

前期学习率高快速收敛，后期低稳定微调。

---

## 九、可以怎么改

1. 把 CNN 换成 ViT：更接近真实 VLA。
2. 把词袋换成 Transformer：用 HuggingFace 的 BERT。
3. 加 LoRA：只微调少量参数。
4. 换成真实数据：用 Open X-Embodiment。
5. 换成 MuJoCo：更真实的仿真环境。

---

## 十、一句话总结

这套代码做了一件事：
用最简单的组件（CNN + 词袋 + MLP），走通 VLA 的完整流程（数据 → 模型 → 训练 → 推理）。

真实 VLA 只是把每个组件换成更强的大模型：

· CNN → ViT / SigLIP
· 词袋 → LLaMA
· MLP 融合 → 统一 Transformer
· 从头训 → LoRA 微调



---



## 一、数据层面

### 1.为什么要造假数据，不用真实数据？

表面原因：你没有真实机械臂数据。

深层原因：学 VLA 的难点不是数据本身，是理解数据长什么样。

真实数据（比如 Open X-Embodiment）也是这个结构：

```text
图像序列 + 指令 + 动作序列 + mask
```

你造的数据只是把"真实相机画面"换成"白底红方块"，但数据结构完全一样。

类比：学做菜先练切土豆丝，不用真的做满汉全席。切土豆丝和切松露的手法是一样的。

所以：造数据是为了让你专注理解流程，而不是被数据采集、清洗、标注这些工程问题淹没。

---

### 2.为什么存成 h5，不用 csv 或 json？

csv/json 的问题：

· 存数字数组很慢，要转成文本再转回来。
· 不支持压缩。
· 文件一大就卡。
· 不能存多维数组，只能存表格。

h5 的好处：

· 二进制存储，读写快。
· 支持压缩（compression="gzip"）。
· 能存任意维度的数组。
· 支持"部分读取"——只读第 100 帧，不用加载整个文件。

为什么机器人数据特别需要 h5：

一段演示可能有几千帧，每帧 (224, 224, 3) 的图像。
100 段演示就是几十 GB。
用 csv 根本存不下，用 h5 可以压缩 + 分块读。

类比：h5 像一个"带索引的档案柜"，csv 像"一本流水账"。数据少时都行，数据多时档案柜完胜。

---

### 3.为什么图像要归一化到 [0, 1]？

原始像素是 0~255。

问题：

· 神经网络里有很多乘法和加法。
· 如果输入是 255，经过几层后数值会爆炸成几百万。
· 梯度也会跟着爆炸，训练不稳定。

归一化后：

· 输入变成 0~1。
· 每层输出保持在合理范围。
· 梯度稳定，训练收敛快。

类比：用米还是毫米量身高都行，但用米时数字是 1.75，用毫米是 1750。
神经网络更喜欢小数字，就像你报身高会说 1.75 米而不是 1750 毫米。

为什么是除以 255 不是减均值除方差：

· 除以 255 是最简单的归一化，把范围压到 [0,1]。
· 更严格的做法是 (x - mean) / std，用数据集统计的均值和标准差。
· 教学代码用简单版，真实项目用严格版。

---

### 4.为什么要 .permute(0, 3, 1, 2) 换维度顺序？

原始形状：(T, H, W, 3) = (时间, 高, 宽, 通道)。

PyTorch 卷积要求：(T, 3, H, W) = (时间, 通道, 高, 宽)。

为什么 PyTorch 要通道在前：

· 这是历史约定，来自早期 GPU 加速库的设计。
· PyTorch 内部按这个顺序优化了内存布局。
· 不换的话，卷积层会报错或算错。

类比：同样是"地址"，中国是"省市区街道"，美国是"街道城市州邮编"。
格式不对，快递就送不到。数据格式不对，模型就读不懂。

为什么 h5 里存的是 (T, H, W, 3)：

· 因为图像处理库（PIL、OpenCV）默认是"高宽通道"。
· 存的时候用自然格式，用的时候再转。

---

### 5.为什么要取 T 帧历史，不只给当前帧？

单帧的问题：

· 一张图看不出物体是在往左还是往右移动。
· 看不出速度。
· 看不出之前发生了什么。

多帧的好处：

· 模型能推断运动方向。
· 能估计速度。
· 能做更平滑的决策。

为什么 T=3：

· 太少（T=1）看不到运动。
· 太多（T=10）计算量大，且很久以前的帧对当前决策帮助有限。
· T=3 或 T=4 是常见折中。

类比：开车时你不会只看当前一帧，你会回忆"刚才那辆车是不是在减速"。
T 帧就是给模型这个"回忆"。

---

### 6.为什么历史帧不够时要重复第一帧？

问题：episode 开头，t=0 时只有 1 帧，没有历史。

三种处理方式：

1. 补零：用全黑图填充。
2. 重复第一帧：用最早的帧填充。
3. 丢弃开头样本：只训练中间部分。

为什么选重复第一帧：

· 补零会引入"黑色"这个不存在的视觉信息，模型会困惑。
· 丢样本浪费数据。
· 重复第一帧表示"没有变化"，最自然。

类比：录像刚开始时，你会假设"画面和第一帧一样"，而不是假设"画面是黑的"。

---

### 7.为什么要有 action_mask？

问题：6 维动作里，有些维度在某些时刻可能无效。

例子：

· 只移动不平移旋转时，旋转 3 维无效。
· 某些关节不参与某个任务。

如果不加 mask：

· 无效维度的误差也会算进 loss。
· 模型被迫去学"预测无效维度"，浪费容量。
· 可能把有效维度的学习也带偏。

mask 的作用：

```python
loss = (|pred - action| * mask).sum() / mask.sum()
```

· mask=1 的维度参与 loss。
· mask=0 的维度被乘成 0，不影响。
· 除以 mask.sum() 做平均，避免除以 0。

类比：考试时有些题是选做，评分时不该算进总分。mask 就是"哪些题要算分"的标记。

---

## 二、模型层面

### 8.为什么要分视觉塔、语言塔、动作头？

因为这是三种不同的信息：

· 视觉：看到什么（图像 → 特征）。
· 语言：要做什么（文本 → 特征）。
· 动作：怎么做（融合 → 输出）。

为什么不直接一个网络从图像到动作：

· 语言指令就浪费了。
· 同一个视觉场景，不同指令要输出不同动作。
· 不分开的话，模型学不到"语言控制行为"这件事。

类比：人做事也是分开的——眼睛看、耳朵听、大脑决策、手执行。
分开是为了各司其职，最后再协调。

---

### 9.为什么视觉塔用 CNN（虽然真实用 ViT）？

教学原因：

· CNN 概念简单，卷积、池化大家熟悉。
· 代码短，几行就能跑。
· CPU 也能训，不需要 GPU。

真实原因：

· ViT 需要大数据预训练才有优势。
· 你造的数据只有 200 条，ViT 训不起来。
· CNN 在小数据上更容易收敛。

类比：教小孩算数先用苹果，不用微积分。
苹果和微积分都能算数，但苹果更直观。

等你理解了流程，再换成 ViT，这时你已经知道"视觉塔是干嘛的"，换起来就顺了。

---

### 10.为什么对 T 帧取平均，不用注意力？

平均的缺点：

· 所有帧权重一样。
· 丢掉了时间信息。

为什么教学用平均：

· 一行代码 .mean(dim=1) 就搞定。
· 不需要额外参数。
· 对简单任务够用。

真实做法：

· 用 Transformer 处理 T 帧序列。
· 或用一个专门的时序模块（LSTM、Temporal Transformer）。
· 让模型自己学"哪帧重要"。

类比：平均像"三个人投票一人一票"，注意力像"专家投票权重更高"。
简单任务一人一票够用，复杂任务需要加权。

---

### 11.为什么用词袋，不用 BERT？

词袋的缺点：

· 丢词序。
· 丢语法。
· 一词多义分不清。

为什么教学用词袋：

· 不需要下载预训练模型。
· 代码 10 行搞定。
· 不需要 GPU。
· 对你造的简单指令（"move to the red cube"）够用。

真实做法：

· 用预训练语言模型（BERT、LLaMA）。
· 每个词有上下文相关的向量。
· 能理解复杂指令。

类比：词袋像"把句子拆成关键词"，BERT 像"真正读懂句子"。
简单指令关键词够了，复杂指令需要真理解。

---

### 12.为什么用 hash 把词映射成 ID？

问题：模型只能处理数字，不能处理字符串。
所以每个词要映射成一个整数 ID。

正规做法：

· 建一个词表：{"move": 0, "to": 1, "the": 2, ...}。
· 查表得到 ID。

为什么教学用 hash：

· 不用提前建词表。
· 一行代码搞定。
· 对少量指令够用。

hash 的问题：

· 不同词可能 hash 到同一个 ID（冲突）。
· 每次运行 hash 结果可能不同（Python 的 hash 有随机性）。

真实项目：一定用 tokenizer，不用 hash。

---

### 13.为什么融合用拼接，不用交叉注意力？

拼接：

```python
h = torch.cat([visual, language], dim=-1)
```

· 视觉 32 维 + 语言 32 维 = 64 维。
· 简单，一行代码。
· 但视觉和语言在拼接前没有任何交互。

交叉注意力：

· 视觉特征可以"查询"语言特征。
· 语言特征可以"查询"视觉特征。
· 表达力强，但代码复杂。

为什么教学用拼接：

· 先理解"融合"这个概念。
· 拼接是最直观的融合。
· 后面再学交叉注意力。

类比：拼接像"两个人各写一半报告，最后装订在一起"。
交叉注意力像"两个人反复讨论，互相修改，最后写出一份报告"。
后者更好，但前者更容易理解。

---

### 14.为什么输出 6 维动作？

机械臂末端位姿的标准表示：

· 位置 3 维：(x, y, z)。
· 姿态 3 维：(roll, pitch, yaw)，即绕三个轴的旋转。

为什么用欧拉角：

· 直观，3 个数字就描述姿态。
· 缺点是有"万向锁"问题，真实系统常用四元数。
· 教学用欧拉角，简单。

为什么不是 7 维：

· 7 维通常是位置 3 + 四元数 4。
· 四元数没有万向锁，但多一维。
· 教学用 6 维欧拉角，够用。

---

## 三、训练层面

### 15.为什么用 L1 损失，不用 MSE？

MSE：(pred - action)²，对大误差惩罚重。
L1：|pred - action|，对所有误差一视同仁。

为什么机器人动作常用 L1：

· 动作有异常值时，MSE 会被异常值主导。
· L1 更鲁棒，对异常值不敏感。
· L1 的梯度是常数，训练稳定。

举例：

· 真实动作是 0.1，预测 0.2，误差 0.1。
· 另一个样本真实 10，预测 20，误差 10。
· MSE 里第二个样本的 loss 是第一个的 10000 倍。
· L1 里第二个是第一个的 100 倍。
· L1 不会被一个异常样本带偏。

类比：MSE 像"对犯大错的人重罚"，L1 像"对所有错误同等对待"。

---

### 16.为什么要梯度裁剪？

问题：反向传播时，梯度可能变得非常大（梯度爆炸）。

后果：

· 参数一步更新太多，直接跳到坏区域。
· loss 变成 NaN，训练崩溃。

梯度裁剪：

```python
clip_grad_norm_(params, 1.0)
```

· 如果梯度范数超过 1.0，就按比例缩小到 1.0。
· 保证每步更新不会太猛。

为什么动作头特别容易爆炸：

· 动作是连续值，范围可能很大。
· 动作头的输出直接是数值，不像分类有 softmax 约束。
· 所以动作头梯度容易失控。

类比：开车下坡时踩刹车，防止速度失控。梯度裁剪就是训练的"刹车"。

---

### 17.为什么用 AdamW，不用 SGD？

SGD：随机梯度下降，最基础的优化器。
AdamW：自适应学习率 + 权重衰减。

AdamW 的好处：

· 每个参数有自己的学习率，自动调整。
· 对学习率不敏感，1e-3 到 1e-4 都能跑。
· 收敛快，适合 Transformer。

为什么 Transformer 微调几乎都用 AdamW：

· Transformer 参数多，不同层梯度尺度差异大。
· Adam 的自适应学习率能处理这种差异。
· AdamW 修正了 Adam 的权重衰减实现，更稳定。

类比：SGD 像"所有人用同一个速度跑步"，AdamW 像"每个人根据自己的体力调整速度"。

---

### 18.为什么用余弦退火调度学习率？

固定学习率的问题：

· 太大：后期在最优解附近震荡，收敛不了。
· 太小：前期收敛慢。

余弦退火：

· 前期学习率大，快速靠近最优解。
· 后期学习率小，精细调整。
· 按余弦曲线平滑下降。

```text
lr
 |
 |\\
 | \\
 |  \\
 |   \\___
 |_______\\___→ step
```

为什么用余弦而不是线性：

· 余弦前期下降慢，后期下降快。
· 更符合"先粗调后精调"的直觉。
· 实际效果通常更好。

类比：找钥匙时，先在房间大范围找（大学习率），找到大概位置后再仔细翻（小学习率）。

---

### 19.为什么要 optimizer.zero_grad()？

PyTorch 的梯度默认是累加的。

如果不清零：

· 第 1 步梯度是 g1。
· 第 2 步梯度变成 g1 + g2。
· 第 3 步变成 g1 + g2 + g3。
· 越训越乱。

所以每次更新前要清零：

```python
optimizer.zero_grad()   # 清零
loss.backward()         # 算新梯度
optimizer.step()        # 更新
```

为什么 PyTorch 默认累加：

· 有时你想手动累加多个 batch 的梯度（梯度累积）。
· 所以默认不清零，给你灵活性。
· 但普通训练必须手动清零。

类比：记账时每笔新账要另起一页，不能在上次的基础上继续写。

---

### 20.为什么要 model.train() 和 model.eval()？

有些层在训练和推理时行为不同：

· Dropout：训练时随机丢神经元，推理时全用。
· BatchNorm：训练时用当前 batch 统计，推理时用全局统计。

model.train()：切到训练模式，启用 Dropout 等。
model.eval()：切到推理模式，关闭 Dropout。

如果不切换：

· 推理时 Dropout 还开着，每次输出都不一样，动作抖动。
· 训练时 BatchNorm 用错统计，训练不稳定。

类比：考试时不能还像平时练习那样"随机翻书查答案"（Dropout）。
考试要闭卷，用你真正学会的东西。

---

### 21.为什么要 torch.inference_mode()？

训练时需要：

· 保存中间结果（用于反向传播）。
· 构建计算图。

推理时不需要：

· 不反向传播。
· 不需要计算图。

inference_mode() 做的事：

· 不构建计算图。
· 不保存中间结果。
· 省显存、加速。

和 no_grad() 的区别：

· no_grad()：不建计算图，但张量还能用于训练。
· inference_mode()：更彻底，连版本计数都不记，更快。

推理首选 inference_mode()。

---

### 22.为什么要 clip_grad_norm_ 在 step 之前？

顺序不能乱：

1. zero_grad()：清空旧梯度。
2. backward()：算新梯度。
3. clip_grad_norm_()：裁剪梯度。
4. step()：用裁剪后的梯度更新。

如果裁剪在 step 之后：

· step 已经用原始梯度更新了参数。
· 裁剪没有意义。

类比：先检查刹车再开车，不能开完再检查。

---

## 四、工程层面

### 23.为什么要 DataLoader，不自己写循环？

DataLoader 帮你做：

· 分批（batch）。
· 打乱（shuffle）。
· 多进程加载（num_workers）。
· 自动调用 collate_fn。

自己写的问题：

· 要手动管理索引、分批、打乱。
· 多进程加载很难写对。
· 代码长且容易出 bug。

类比：DataLoader 像"自动喂饭机"，你只管吃，不用自己盛饭、端碗、洗碗。

---

### 24.为什么要 Dataset 类，不直接读文件？

Dataset 的好处：

· 统一接口：dataset[i] 取第 i 条。
· 和 DataLoader 配合，自动分批。
· 支持懒加载：用到才读，不占内存。

不用的后果：

· 要自己管理所有数据。
· 数据多时内存爆。
· 和 PyTorch 生态脱节。

类比：Dataset 像"图书馆索引系统"，你要第 5 本书，它帮你找。
不用的话，你得自己翻遍所有书架。

---

### 25.为什么要 collate_fn？

默认 collate 的问题：

· 只能处理数字张量。
· 遇到字符串列表会报错。
· 不同长度的数据不会处理。

自定义 collate_fn：

· 图像 stack 成 batch。
· 字符串放列表。
· 长度不一致的做 padding。

为什么必须有：

· 我们的 batch 里有图像（张量）和指令（字符串）。
· 默认 collate 处理不了混合类型。
· 所以必须自定义。

---

### 26.为什么要固定随机种子？

```python
torch.manual_seed(42)
```

不固定的后果：

· 每次运行初始化不同。
· 数据打乱顺序不同。
· 结果不可复现。

固定的好处：

· 调参时能对比"只改了一个变量"的效果。
· 出 bug 能复现。
· 论文实验可复现。

为什么是 42：

· 这是《银河系漫游指南》里的梗，程序员喜欢用。
· 其实任何数字都行。

---

### 27.为什么要保存 state_dict() 而不是整个模型？

保存整个模型的问题：

· 依赖模型类的定义。
· 换代码后可能加载失败。
· 文件大。

只保存参数：

```python
torch.save(model.state_dict(), "model.pt")
```

· 只存参数数值。
· 加载时先建模型再加载参数。
· 更灵活、更小。

类比：存文档内容而不是存整个 Word 软件。
换电脑后装个 Word 就能打开，不用把软件也拷过去。

---

## 五、总结：所有"为什么"背后的三个原则

### 原则 1：先让它跑起来，再让它跑得好

· 用 CNN 不用 ViT：因为 CNN 简单，能快速跑通。
· 用词袋不用 BERT：因为词袋代码短，不用下载模型。
· 用平均不用注意力：因为平均一行代码。

教学代码的第一目标是"能跑"，不是"最好"。

### 原则 2：每个设计都在"简单"和"有效"之间权衡

· 词袋简单但丢词序 → 简单任务够用。
· 拼接简单但交互弱 → 简单任务够用。
· L1 简单但不如 Huber → 简单任务够用。

真实项目会选更复杂的方案，但你要先理解简单版。

### 原则 3：所有"奇怪"的设计都是为了解决具体问题

· mask：解决无效动作维度污染 loss。
· 梯度裁剪：解决梯度爆炸。
· 归一化：解决数值范围过大。
· 零梯度：解决梯度累加。
· eval 模式：解决 Dropout 在推理时不该生效。

没有无缘无故的设计，每个都是被某个问题逼出来的。

---

## 六、该怎么学

第一步：把上面的代码跑一遍，看它真的能输出动作。

第二步：改一个地方，看结果怎么变。比如：

· 把 T 从 3 改成 1，看 loss 怎么变。
· 把 mask 全设成 0，看会发生什么。
· 把学习率调大 100 倍，看会不会爆炸。

第三步：每次改之前，先猜结果，再验证。猜对了说明理解了。

第四步：把某个模块换成更复杂的版本（CNN → ViT），看效果提升多少。