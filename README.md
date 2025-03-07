# CPEHGN

This is the source code for the paper "Complementary Perspectives Enhancement via Hierarchical Graph Network for Multimodal Fake News Detection."

## Download data
If you want to download the `Weibo` dataset, you can access the following link: [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18/)

If you want to download the `GossipCop` dataset, you can access the following link: [https://github.com/shiivangii/SpotFakePlus]( https://github.com/shiivangii/SpotFakePlus/)

Then, you should put them into `./Data`

## Data pre-processing

Use `data_preprocess_weibo.py` to pre-process the `Weibo` dataset.

Use `data_preprocess_gossipcop.py` to pre-process the `GossipCop` dataset.

If you want to change dataset for training, you should revise
```python
import utils.data_preprocess_weibo as data_preprocess
```
```python
--dataset default='weibo'
```
## Setup

### Dependencies

- Python=3.12.4
- PyTorch=2.3.1
- Torchvision=0.18.1
- Transformers=4.6.0


### Run the code

run ```main.py ```

## Reference
Thanks to their great works
* [MINER-UVS](https://github.com/wangbing1416/MINER-UVS)
