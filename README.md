# Neural Style Transfer with OpenCV

This repo shows how to perform Neural Style Transfer using OpenCV. <br/>
This project uses the pretrained models by [Justin Johnson](https://github.com/jcjohnson/fast-neural-style)<br/>
The code closely follows the pyimagesearch blog by Adrian Rosebrock that you can find here: https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/

# Requirements
python3
wget
opencv-python

# How to use?
1) Clone or download the repository
```
git clone https://github.com/iArunava/Neural-Style-Transfer-with-OpenCV
```

2) in Master directory execute download script for download models.
```
sh models/download.sh
```

3) There are two modes to the project
- Image
- Video

For the image mode,
```
python3 init.py --image ./path/to/image.png
```

For the video mode
```
python3 init.py
```

- Press `n` to move to the next style
- Press `p` to move to the previous style

4) Use the help to get some involved usage arguments
```
python3 init.py --help
```

5) **Enjoy!**

# Results

## Image

![rawpixel](https://user-images.githubusercontent.com/26242097/45502148-d6ba8b00-b7a0-11e8-9841-ae2c27f7ff4e.jpg)

![sty1](https://user-images.githubusercontent.com/26242097/45502132-cacec900-b7a0-11e8-9e06-d903e5746d55.png)

_Image Credits: Photo by rawpixel on Unsplash_

# LICENSE

Feel free to fork and make changes to the code! Have fun!
