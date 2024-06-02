# 800t-emulation
This project provides tools to simulate the Cinestill 800t on RAW digital images from a Panasonic G85/G80/G81

Download our dataset from
https://drive.google.com/drive/folders/1o8s59JwpYBfTTskVF3c7DCLGVQAuGsFa?usp=drive_link

Checkout our demo in the [Demo Notebook](demo.ipynb)

Coded for Python 3.10+

The grain rendering part of the program is inspired by http://www.ipol.im/pub/art/2017/192/

Although our grain rendering implementation provides a cross platform solution, you may wish to run the original
cpp version for performance reasons.
To get the same execution at the one of the demo, run:

```
./film_grain_rendering_main input_image output_image -r 0.150 -sigmaR 0 -NmonteCarlo 100 -algorithmID 0 -color 1
```

We also built a test implementation of the grain simulation for cuda with pycuda. Checkout the
[grain detection pycuda test](https://github.com/cs-413-cinestill/800t-emulation/tree/grain-detection-pycuda-test)
branch.