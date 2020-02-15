# Tensor Directed Stylization
*BERTHET Vincent*

*QUERO Benoit*
## Introduction 
This project rely on "Tensors-directed simulation of strokes for image stylization with hatching and contours" 
*- David TSCHUMPERLE in September 2011*

Its describe an algorithm to generate sketches from color images directed by an image-dependent tensor-valued geometry


| ![](./output/tensors/pierre/pierre.gif) | ![](./output/tensors/olivier/olivier.gif) | ![](./output/tensors/lamarche/lamarche.gif)  |
|:-:|:-:|:-:|
| ![](./output/tensors/ewa/ewa.gif)  | ![](./output/tensors/cozot/cozot.gif) | ![](./output/tensors/eric/eric.gif)  |
## Environment configuration (Conda)
### Import environment
You can directly use exported environment in `./conda/` by running the following command `conda env create -f conda/win.yml` 

*(\*.yml prefix path value should be edit for your device)*
### Packages
The previous environment contains the following packages :


[OpenCV](https://anaconda.org/conda-forge/opencv)  
`conda install -c conda-forge opencv`

[SciPy](https://anaconda.org/conda-forge/scipy)  
`conda install -c conda-forge scipy `

[Progressbar2](https://anaconda.org/conda-forge/progressbar2)  
`conda install -c conda-forge progressbar2 `

[Blend modes](https://anaconda.org/conda-forge/blend_modes)  
`conda install -c conda-forge blend_modes `

## Scripts
[naive.py](./scripts/naive.py) naive approach of the algorithm

[tensors.py](./scripts/tensors.py) main script to run the algorithm and use tensors approach for an input image. Arguments that can be used are the following :
- **-i,--image :** path of input image
- **-s1,--sigma1 :** set sigma 1 for gaussian gaussian blur 1
- **-s2, --sigma2 :** set sigma 2 for gaussian blur 2
- **-p1, --power1 :** power 1 to change tensors
- **-p2, --power2 :** power 2 to change tensors
- **-n, --number :** number of strokes on the sketch
- **-e, --epsilon :** epsilon to draw strokes
- **-l, --length :** length of strokes
- **-c, --coefficient :** ratio to reduce the input image
- **-o, --output :** custom output under ./output/tensors directory

[tensorsTools.py](./scripts/tensorsTools.py) toolbox for rendering, class structure,  used by tensors script

[generate.py](./scripts/generate.py) use to generate a lot of image using tensors approach by setting some different parameters of tensors script

`python ./scripts/tensors.py -i ./sources/lena.png`
## Results
### Naive approach
| ![](./sources/desert.png) | ![](./output/naive/desert/img_res_3.jpg) | ![](./output/naive/desert/img_res_2.jpg) |
|:-:|:-:|:-:|
|Source| Gray| Mulitply|

| ![](./sources/lena.png) | ![](./output/naive/lena/img_res_3.jpg) | ![](./output/naive/lena/img_res_2.jpg) |
|:-:|:-:|:-:|
|Source| Gray| Mulitply and Gaussain blur|



### Tensors approach


| ![](./output/tensors/desert/img.jpg) | ![](./output/tensors/desert/sktech1_n_250000_e_1_l_15_p1_1.2_p2_0.5img_results.jpg) | ![](./output/tensors/desert/sketch_2_n_100000_e_2_l_30_p1_1.2_p2_0.5img_results.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/lena/img.jpg) | ![](./output/tensors/lena/img_results_0.jpg) | ![](./output/tensors/lena/img_results_1.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/ratatouille/img.jpg) | ![](./output/tensors/ratatouille/n_150000_e_1_l_40_p1_3.0_p2_0.5img_results.jpg) | ![](./output/tensors/ratatouille/res_1_n_50000_e_4_l_100_p1_1.2_p2_0.5img_results.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/obama/img.jpg) | ![](./output/tensors/obama/img_1_n_100000_e_4_l_100_p1_1.2_p2_0.5img_results.jpg) | ![](./output/tensors/obama/img_2_n_125000_e_3_l_125_p1_1.2_p2_0.5img_results.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/montagne/img.jpg) | ![](./output/tensors/montagne/img_results_4.jpg) | ![](./output/tensors/montagne/res_2_n_250000_e_1_l_15_p1_1.2_p2_0.5img_results.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/pyramide/img.jpg) | ![](./output/tensors/pyramide/res_2_n_50000_e_2_l_100_p1_1.2_p2_0.5img_results.jpg) | ![](./output/tensors/pyramide/img_results_6.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|


| ![](./output/tensors/matthias/img.jpg) | ![](./output/tensors/matthias/matthias_1.png) | ![](./output/tensors/matthias/img_2_n_150000_e_5_l_100_p1_1.2_p2_0.5img_results.jpg) |
|:-:|:-:|:-:|
|Source| Sketch 1 | Sketch 2|
