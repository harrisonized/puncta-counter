## Introduction

I wrote this library to extract features from the output of CellProfiler, which allows the Anguera lab to quantify Xist localization from images of RNA FISH. The image and data files were provided to me by **Sarah Pyfrom**. The algorithms were tested on noisy data (to ensure robustness), which will be added to some pytests later. Currently, the main script is `describe_puncta.py`, which expects the `nuclei.csv` and `puncta.csv` files. It then uses the `confidence_ellipse` and `min_vol_ellipse` algorithms to draw boundaries around puncta. These algorithms return the following parameters per puncta: `center_x`, `center_y`, `major_axis_length`, `minor_axis_length`, `orientation`. Since the goal is to export ellipse parameters rather than exact boundaries, it is lightweight and runs very quickly. Most of the runtime is spent on file IO while saving plots.



## Getting Started

Install the relevant libraries in your conda environment. A requirements file is coming, but for now, just bear with me.

In your command line, run:

```bash
conda activate <your_env>
cd puncta_counter
python describe_puncta.py -a 'confidence_ellipse' 'min_vol_ellipse'
```

Note that to save Bokeh figures, you must install geckodriver, and this is separate from your conda environment:  `sudo apt-get install firefox-geckodriver`. Depending on how difficult it is for people to use Bokeh, I might switch to matplotlib, but this is not currently a priority.



## To Do

1. Outlier detection
2. Doublet detection and filtering
3. Actual image quantification



## License

This project is licensed under the terms of the [MIT license](https://github.com/harrisonized/puncta-counter/blob/master/LICENSE). If you choose to use this as a building block for your own project, please fork this repository, clone from your own fork, and give me due credit in a CREDITS.md file ([example from my blog](https://github.com/harrisonized/harrisonized.github.io/blob/master/CREDITS.md)).



## Getting Help

Always feel free to email me at [harrison.c.wang@gmail.com](mailto: harrison.c.wang@gmail.com).