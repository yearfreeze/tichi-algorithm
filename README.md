# tichi-algorithm
ALibaba TIANCHI global local climate zone(LCZ) image classifier competition

任务是在全球的某几个城市中进行当地气候区 对包含来自 Sentinel-1(sar)和 Sentinell-2(高 光谱 )patch 图像进行17分类。
对多达60GB的图像数据进行初步清洗，并对原始18通道图像进行特征提取 。
针对训练集中出现的样本类间不平衡问题，采用欠、重采样smote方法 做平衡处理
使用第二步得到的图像集 lenet -5、resnet18 、resnet50 esnet50 esnet50 、resnet101 、densenet50网络 。
用 IamgeNet数据 集预训的大型网络如 vgg19cifarnet等作为对照。 
为了克服训练时间的急剧增加，修改程序使得训练模型在两块 2080ti并行进。
为了平衡 CPU 和 GPU 的时间耗费，绕过 GIL 全局锁  用多进程方式 完成生产者消费模型 ，提高整个管线效率。
采用简单加权和 boosting的方式进行模型融合.
acc=86.4% top3%

ResNet

Tensorflow implementation of

    ResNet-18-34-50-101.
    ResNet-V2-18-34-50-101.

Prerequisites

    tensorflow 1.9
    python 2.7 or 3.6

