# Instance-Segmentation-with-SpatialEmbedding-CA

SpatialEmbedding + Detail-Branch + Feature Fusion with Cross-Attention. It includes code explanation about spatial embedding loss & network.

Pytorch implementation of spatial embdding based instance segmentation with A1 of CVPPP2017. It utilizes detail-branch of BiSeNetV2 to enhance the feature representation of encoder. Feature fusion is performed by cross-attention which is based on Restormer.

My paper 'A study on the Performance Improvement of Small Local Regions in Instance Segmentation' was submitted to 'Autumn Annual Conference of IEIE, 2022'(https://conf.theieie.org/2022f/).

<br/>

## Usage (for CVPPP2017)

1. setting is done by `train_config_cvppp.py`

2. train with `train_cvppp.py`

3. get inference with `test_config_cvppp.py` 

4. evaluate the inference output with `evaluation.py`

**If you want to know the Davy Neven's original spaitial embedding paper, see `criterions/my_loss_cvppp.py` & `utils/utils_cvppp.py`. I added some explanations about the code.**

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
|1.BranchedBiSeNetV2 (detail-branch x) | 0.868 | 0.857 |  
|2.BranchedBiSeNetV2 (detail-branch o & BGA) | 0.876 | 0.821 |
|3.BranchedBiSeNetV2_CA (detail-branch 0 & CA) | 0.884 | 0.643 |  

![image](https://user-images.githubusercontent.com/44194558/197446790-c94bbf6c-03fe-416c-8bd1-a520c354dd56.png)

<br/>


## References

1. Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (https://arxiv.org/abs/1906.11109)

2. BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation (https://arxiv.org/abs/2004.02147)

3. Restormer: Efficient Transformer for High-Resolution Image Restoration (https://arxiv.org/pdf/2111.09881v2.pdf)

4. Multi-Modality Cross Attention Network for Image and Sentence Matching (https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)
