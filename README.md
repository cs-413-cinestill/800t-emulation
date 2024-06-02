# 800t-emulation
This project provides tools to simulate the Cinestill 800t on RAW digital images from a Panasonic G85/G80/G81

Download our dataset from
https://drive.google.com/drive/folders/1o8s59JwpYBfTTskVF3c7DCLGVQAuGsFa?usp=drive_link

Checkout our demo in the [Demo Notebook](demo.ipynb)

Coded for Python 3.10+

The grain rendering part of the program is inspired by http://www.ipol.im/pub/art/2017/192/
To get the same execution at the one of the demo, do:
./film_grain_rendering_main input_image output_image -r 0.150 -sigmaR 0 -NmonteCarlo 100 -algorithmID 0 -color 1