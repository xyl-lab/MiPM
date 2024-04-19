# MiPM

## 背景

针对桥梁沉井下沉的施工场景，通过传感器收集施工现场的GPS数据以及沉井底部结构应力的数据，然后通过GPS数据计算沉井的姿态指标：下沉量、横/纵向倾斜度、横/纵向顶口偏位、横/纵向底口偏位，共七个指标，利用姿态指标序列以及结构应力序列预测下一时刻的沉井姿态，即多元时序数据预测问题及其应用

## 模型结构

模型介绍：对沉井下沉姿态的预测本质上是多元时序数据预测问题，传统的统计模型以及机器学习模型都不适用于时序数据的预测。本文，基于卷积神经网络以及图神经网络建立深度学习模型，提取时序数据的时空关系，预测沉井的下沉姿态。

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/model_architecture.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">模型的总体架构图</div> </center>





<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/graph_learning.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">图学习层架构图</div> </center>



![tcmodule](./pictures/graph_learning.png)

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/tcmodule.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">时间卷积模块架构图</div> </center>

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/mix-pop.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">图卷积模块架构图</div> </center>

# 实验结果

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/study_results.jpg">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">各模型的预测结果</div> </center>

# 基线模型

以下基线模型，本文根据原网络结构，基于本文的数据集，进行调参。其余的基线模型由本文实现

## DARNN

@article{qin2017dual,
  title={A dual-stage attention-based recurrent neural network for time series prediction},
  author={Qin, Yao and Song, Dongjin and Chen, Haifeng and Cheng, Wei and Jiang, Guofei and Cottrell, Garrison},
  journal={arXiv preprint arXiv:1704.02971},
  year={2017}
}
https://github.com/sunfanyunn/DARNN
参数设置: hidden_size:128, dropout: 0.1

## LSTNet

@inproceedings{lai2018modeling,
  title={Modeling long-and short-term temporal patterns with deep neural networks},
  author={Lai, Guokun and Chang, Wei-Cheng and Yang, Yiming and Liu, Hanxiao},
  booktitle={The 41st international ACM SIGIR conference on research \& development in information retrieval},
  pages={95--104},
  year={2018}
}
https://github.com/laiguokun/LSTNet
参数设置: hidden_size: 64, d_model: 160, filter_size: 4, dropout: 0.30000000000000004, highway_window: 16

## STGNN

 @inproceedings{wang2020traffic,
  title={Traffic flow prediction via spatial temporal graph neural network},
  author={Wang, Xiaoyang and Ma, Yao and Wang, Yiqi and Jin, Wei and Wang, Xin and Tang, Jiliang and Jia, Caiyan and Yu, Jian},
  booktitle={Proceedings of the web conference 2020},
  pages={1082--1092},
  year={2020}
}
https://github.com/LMissher/STGNN
参数设置: d_k:20, num_layers: 1

## MTGNN

@inproceedings{wu2020connecting,
  title={Connecting the dots: Multivariate time series forecasting with graph neural networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={753--763},
  year={2020}
}
https://github.com/nnzhan/MTGNN
参数设置: layers: 3, propalpha: 0.2, subgraph_size: 2, out_channels:64, gcn_depth:1

## Informer

@inproceedings{zhou2021informer,
  title={Informer: Beyond efficient transformer for long sequence time-series forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={12},
  pages={11106--11115},
  year={2021}
}
https://github.com/zhouhaoyi/Informer2020
参数设置: factor: 4, d_k:40, dropout: 0.1

## FEDformer

@inproceedings{zhou2022fedformer,
  title={Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={International conference on machine learning},
  pages={27268--27286},
  year={2022},
  organization={PMLR}
}
https://github.com/MAZiqing/FEDformer
参数设置: d_k:40, dropout: 0.1

## MrCAN

@article{zhang2023mrcan,
  title={MrCAN: Multi-relations aware convolutional attention network for multivariate time series forecasting},
  author={Zhang, Jing and Dai, Qun},
  journal={Information Sciences},
  volume={643},
  pages={119277},
  year={2023},
  publisher={Elsevier}
}
https://github.com/JZhangNA/MrCAN
参数设置: dropout:0.3, filter_size:4, d_k:20











