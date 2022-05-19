# 简单图像去雨（使用pytorch框架和ResNet模型）
## 1.参考介绍
基于pytorch的图像去雨,参考文献为2017CVPR文章，数据集约800m，共1000张图片
- 参考文献**Removing rain from single images via a deep detail network**.       
- 使用数据集**Rain1000**

原始论文，数据集，代码**下载地址**：https://xueyangfu.github.io/projects/cvpr2017.html

使用模型为ResNet + Neg-mapping


<img width="104" alt="model_image" src="https://user-images.githubusercontent.com/105830075/169250649-2bf4ed35-324f-48f6-b212-fb0da1f34a3b.png">



## 2.简单实现
先配置环境，或者在已有环境中打开test.py

提供模型已经经过40个epoch的训练，可以直接运行获取测试结果

测试数据放置在test_data文件夹中，附带50张人工增雨图片。
- 可以自行寻找图片进行替换，但是**不建议使用真实降雨图片**（*因为效果很差*）
- 建议使用真实无雨图片，经过修图之后得到的人工降雨图片。

测试结果在output文件夹中

## 3.继续训练
train.py可以加载原模型数据并继续训练，可以从参考部分提供的链接下载数据集。

原数据集groundtruth图片数量仅为images数量的1/14，通过copy_images.py复制label图片后才能进行训练。

