# Approximate Manifold Defense Against Multiple Adversarial Perturbations (updated version)
This version removes numba dependency to train the RBF layer. 
It allows to train the RBF layer using Tensorflow in a GPU.
Older version of code can be found [[here]](https://github.com/jayjaynandy/RBF-CNN).

A shorter version of this paper has been accepted in [NeurIPS 2019 Workshop on Machine Learning with Guarantees](https://sites.google.com/view/mlwithguarantees/accepted-papers) 
[[pdf]](https://drive.google.com/file/d/1I2WKHg-s7wJgG21apg3FhxaYzzFl4vgt/view), 
[[poster]](https://drive.google.com/file/d/1Wp-kKsc0927ZXo5lS8f2GPnmSpIWdRlN/view) and the full version of this paper is accepted at IJCNN-2020 [[Arxiv Link]](https://arxiv.org/abs/2004.02183).
The video presentation of our paper is provided in this [youtube link](https://www.youtube.com/watch?v=oKBu90fuTgI).

## Step 1: Training of the RBF layer
Train RBF layer using `rbf_training.py`. Dependency: Keras + Tensorflow
Follow the instruction provided inline.


## Step 2: Training of the CNN network in presence of the trainined RBF layer
Similar to the previous version.

Execute `train_rCNN.py` to train rCNN model.

Execute `train_rCNN+.py` to train rCNN+ model. Note that, it would require a single set of adversarial images corresponding to the training images. 
We recommend to execute `train_rCNN.py` model for 50 epochs followed by applying PGD attack to create the adversarial training images (at l_inf = 0.05).

## Citation
If our code or our results are useful in your reasearch, please consider citing:

```[bibtex]
@inproceedings{rbfcnn_ijcnn20,
  author={Jay Nandy and Wynne Hsu and Mong{-}Li Lee},
  title={Approximate Manifold Defense Against Multiple Adversarial Perturbations},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2020},
}
```
