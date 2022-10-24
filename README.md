# Instance-Segmentation-with-SpatialEmbedding-CA

SpatialEmbedding + Detail-Branch + Feature Fusion with Cross-Attention

Pytorch implementation of spatial embdding based instance segmentation with A1 of CVPPP2017. It utilizes detail-branch of BiSeNetV2 to enhance the feature representation of encoder. Feature fusion is performed by cross-attention which is based on Restormer.

<br/>

## Usage (for CVPPP2017)

1. setting is done by `train_config_cvppp.py`

2. train with `train_cvppp.py`

3. get inference with `test_config_cvppp.py` 

4. evaluate the inference output with `evaluation.py`

<br/>

## Network Structure

![image](https://user-images.githubusercontent.com/44194558/197445082-d25771b4-e4e7-49a9-bef3-8f81e6f8c524.png)

<br/>

## Feature Fusion with Cross-Attention

![image](https://user-images.githubusercontent.com/44194558/197445229-ca8d0c57-8ca2-42d3-af04-1e9ab0f807dc.png)

<br/>

## Results

| Model | Mean SBD | Dic | 
| :----------- | :------------: | ------------: | 
|BranchedBiSeNetV2(detail-branch x) | 0.868 | 0.857 |  



## References

1. Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (https://arxiv.org/abs/1906.11109)

2. BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation (https://arxiv.org/abs/2004.02147)

3. Restormer: Efficient Transformer for High-Resolution Image Restoration (https://arxiv.org/pdf/2111.09881v2.pdf)

4. Multi-Modality Cross Attention Network for Image and Sentence Matching (https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)
