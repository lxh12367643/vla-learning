# 代码过程中遇到的一些问题

## 1.draw.py

- draw.rectangle后跟（），如果是【】会被认为成数组而报错
- 画图函数返回值转换成uint8（符合张量0-255）
- 正常dtype（float>uint/int>bool），可以小转大，但是不能大转小，会丢失精度，这是不被允许的
- 注意当T=3时，delta=（obs_xy-mac_xy)/3,最终并不能走到预计位置，在循环中它表示的是走剩下路程的三分之一，正确应为delta=（obs_xy-mac_xy）/（T-t）,可以到达预计位置
- 字符串不能np.stack
- h5py里面，一个字符串会被当做标量字符串数据集，因为是标量，不能chunks或者compression

---

## 2.vla_dataset

- 在根据dataset定义类的时候，注意里面的def名称除了____init____，另外两个只能叫____len____,____getitem____,否则dateset不认（双下划线命名在python中是写死的，不能改）
- 图片取一段历史，动作和掩码只要当前这一步，序列建模经典设定：输入是过去，标签是现在  所以images用的是idx ，action和mask只取第t步
- images转到张量之后不能再np.stack否则会转成numpy
- torch.stack里面要传列表不能是生成器
- key的命名要前后一致，也要和h5文件中的保持一致
- 注意重复帧的索引