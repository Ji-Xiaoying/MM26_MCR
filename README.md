# MCR

This is a PyTorch implementation for MCR.

## Dataset

We conducted experiments on three datasets: MSCOCO, Flickr30K, and CC120K. We followed [NPC](https://github.com/ZhangXu0963/NPC) to split image-text pairs in MSCOCO and FLickr30K into training, validation and testing sets. All datasets can be obtained in  [NPC](https://github.com/ZhangXu0963/NPC).

## Training

**For training MCR** You can train a new model via the following command. Before training, you can read the `params.py` carefully to check your parameter setting. The `--num_anns` should be set to 5 for MSCOCO and Flickr30K, and 1 for CC120K.

```python
python main_MCR.py  --experiments ${YOUR PATH} --noise_ratio ${NOISE RATIO}  --noise_file ${YOUR PATH} --dataset_root ${YOUR PATH}  --dataset f30k --batch_size 256 --epochs 5  --lr 2e-7 --seed 42  --test_frequency 200 --queue_len 100 --gamma_decay 0.9 --lambda_reg 0.05
```

## Evaluation

You can use the following code to evaluate your model:

```python
python main_test.py --eval --resume ${YOUR PATH}  --dataset_root ${YOUR PATH} --dataset f30k --experiments ${YOUR PATH}
```
