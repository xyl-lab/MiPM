

# MiPM

## 背景

沉井作为基础结构，在桥梁建造中，被广泛应用。在沉井建造过程中，实时准确的下沉姿态预测，有助于降低事故风险，提高工程质量。然而，常用的预测模型，如统计模型、机器学习模型，无法处理时序数据中的非线性时空特性，如结构应力，不适用于沉井下沉姿态的预测。另外，现有的针对沉井下沉姿态预测进行的工作，无法同时对沉井多个姿态指标进行预测。因此，本文提出了多指标预测模型MiPM。对沉井的姿态指标：下沉量、横/纵向倾斜度、横/纵向顶口偏位、横/纵向底口偏位，共七个指标进行预测。在沉井下沉过程中，沉井姿态的变化导致底部结构应力的变化，本文使用结构应力作为辅助数据，提高模型的预测精度。

## 模型结构

基于卷积神经网络以及图神经网络建立深度学习模型，提取沉井下沉姿态数据以及结构应力之间的时空特征，预测沉井的下沉姿态。该模型主要分为图学习层、时间卷积模块、图卷积模块、输入输出模块。

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/model_architecture.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">模型的总体架构图</div> </center>



图卷积神经网络需要图邻接矩阵，本文使用GRU以及自注意力机制动态建立沉井下沉姿态以及结构应力之间的图邻接矩阵，提高模型的预测精度

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/graph_learning.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">图学习层架构图</div> </center>

沉井下沉过程中，沉井的姿态以及底部结构应力都是随时间变化，上一时刻的值影响下一时刻，本文使用1D扩展卷积层捕获多元时序数据的时间特征，学习沉井下沉姿态以及结构应力的时间关系

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/tcmodule.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">时间卷积模块架构图</div> </center>

在沉井下沉过程中，沉井姿态的变化会导致沉井底部结构应力的变化，且沉井底部不同位置都布置了结构应力点位，不同点位之间也是相互影响的。本文使用图神经网络来提取多元时序数据之间的空间特征。

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/mix-pop.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">图卷积模块架构图</div> </center>

# Run code
`Python run.py`
# 实验结果

本文各模型对沉井的七个姿态指标进行预测，并使用相关系数R2、均方根误差RMSE、平均绝对百分比误差MAPE，作为评价指标。各模型的预测结果如下表。

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./pictures/study_results.jpg">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">各模型的预测结果</div> </center>

# 基线模型

针对以下基线模型，本文根据原网络结构记忆本文的数据集，进行调参，并给出参数设置。其余未出现在下面模型中的基线模型由本文实现

## DARNN

* @article{qin2017dual,
    title={A dual-stage attention-based recurrent neural network for time series prediction},
    author={Qin, Yao and Song, Dongjin and Chen, Haifeng and Cheng, Wei and Jiang, Guofei and Cottrell, Garrison},
    journal={arXiv preprint arXiv:1704.02971},
    year={2017}
  }
* https://github.com/sunfanyunn/DARNN
* 参数设置: hidden_size:128, dropout: 0.1

## LSTNet

* @inproceedings{lai2018modeling,
    title={Modeling long-and short-term temporal patterns with deep neural networks},
    author={Lai, Guokun and Chang, Wei-Cheng and Yang, Yiming and Liu, Hanxiao},
    booktitle={The 41st international ACM SIGIR conference on research \& development in information retrieval},
    pages={95--104},
    year={2018}
  }
* https://github.com/laiguokun/LSTNet
* 参数设置: hidden_size: 64, d_model: 160, filter_size: 4, dropout: 0.30000000000000004, highway_window: 16

## STGNN

* @inproceedings{wang2020traffic,
    title={Traffic flow prediction via spatial temporal graph neural network},
    author={Wang, Xiaoyang and Ma, Yao and Wang, Yiqi and Jin, Wei and Wang, Xin and Tang, Jiliang and Jia, Caiyan and Yu, Jian},
    booktitle={Proceedings of the web conference 2020},
    pages={1082--1092},
    year={2020}
  }
* https://github.com/LMissher/STGNN
* 参数设置: d_k:20, num_layers: 1

## MTGNN

* @inproceedings{wu2020connecting,
    title={Connecting the dots: Multivariate time series forecasting with graph neural networks},
    author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
    booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining},
    pages={753--763},
    year={2020}
  }
* https://github.com/nnzhan/MTGNN
* 参数设置: layers: 3, propalpha: 0.2, subgraph_size: 2, out_channels:64, gcn_depth:1

## Informer

* @inproceedings{zhou2021informer,
    title={Informer: Beyond efficient transformer for long sequence time-series forecasting},
    author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
    booktitle={Proceedings of the AAAI conference on artificial intelligence},
    volume={35},
    number={12},
    pages={11106--11115},
    year={2021}
  }
* https://github.com/zhouhaoyi/Informer2020
* 参数设置: factor: 4, d_k:40, dropout: 0.1

## FEDformer

* @inproceedings{zhou2022fedformer,
    title={Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting},
    author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
    booktitle={International conference on machine learning},
    pages={27268--27286},
    year={2022},
    organization={PMLR}
  }
* https://github.com/MAZiqing/FEDformer
* 参数设置: d_k:40, dropout: 0.1











