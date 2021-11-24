# (Implicit)<sup>2</sup>: Implicit Layers for Implicit Representations

This repo contains the implementation of the (Implicit)<sup>2</sup> network, an implicit neural representation (INR) learning framework backboned by [Deep Equilibrium Model](https://arxiv.org/abs/1909.01377) (DEQ). By taking advantage of the full-batch training scheme commonly applied to INR learning on low-dimensional data (e.g. images and audios) as well as an approximated gradient, (Implicit)<sup>2</sup> networks operate on significantly less computation and memory budget than exisiting explicit models while perform competitively.

![Comparsion of explicit & implicit models](/assets/exp_vs_imp.png)

For more info and implementation details, please refer to [our paper](https://openreview.net/forum\?id=AcoMwAU5c0s).

## Data

Data used in this project is publicly available on Google Drive ([link](https://drive.google.com/drive/folders/1AVPQ_cqZTKedGWwJ0R39zSBQXw7LC6Pf?usp=sharing)).

To replicate our experiments, create a _data_ folder under the root directory and download the correponding datasets.

```
📦data 
┣ 📂image
┃ ┣ 📜celeba_128_tiny.npy
┃ ┣ 📜data_2d_text.npz
┃ ┗ 📜data_div2k.npz
┣ 📂3d_occupancy
┣ 📂audio
┣ 📂sdf
┗ 📂video
```

## Reproduction of paper results

To reproduce results on image representation and image generalization, run

```
python scripts/train_2d_image.py --config_file ./configs/<task>/config_<task>_<dataset>.yaml
```

For other experiments (audio, video, and 3d_occupancy), run 
```
python scripts/train_<task>.py --config_file ./configs/<task>/<model>.yaml --dataset <dataset>
```

Below is a list of available dataset options for each task (including some extra data we did not cover in the paper)
```
audio: ['bach', 'counting']
video: ['cat', 'bikes']
3d_occupancy: ['dragon', 'buddha', 'bunny', 'armadillo', 'lucy']
```

## Credits

- The set of experiments on image, video, and audio signals and the corresponding data largely follows [SIREN](https://arxiv.org/abs/2006.09661) and [Fourier Feature Networks](https://arxiv.org/abs/2006.10739).
- Models for the 3D occupancy experiments are directly retrieved from the [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)

## Citation
```
@inproceedings{huang2021impsq,
  author    = {Zhichun Huang and Shaojie Bai and J. Zico Kolter},
  title     = {(Implicit)^2: Implicit Layers for Implicit Representations},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
}
```
