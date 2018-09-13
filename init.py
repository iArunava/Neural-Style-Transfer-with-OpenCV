import argparse
import time
import os
import cv2 as cv

FLAGS = None
VID = 'video'
IMG = 'image'

# Source for this function:
# https://github.com/jrosebr1/imutils/blob/4635e73e75965c6fef09347bead510f81142cf2e/imutils/convenience.py#L65
def resize_img(img, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(img, dim, interpolation=inter)
    return resized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path',
                type=str,
                default='./models/instance_norm/',
                help='The model directory.')

    parser.add_argument('-i', '--image',
                type=str,
                help='Path to the image.')

    FLAGS, unparsed = parser.parse_known_args()

    # Set the mode image/video based on the argparse
    if FLAGS.image == '':
        mode =  VID
    else:
        mode = IMG

    # Check if there are models to be loaded and list them
    models = []
    for f in sorted(os.listdir(FLAGS.model_path)):
        if f.endswith('.t7'):
            models.append(f)

    if len(models) == 0:
        raise Exception('The model path doesn\'t contain models')

    # Load the neural style transfer model
    path = FLAGS.model_path + ('' if FLAGS.model_path.endswith('/') else '/')
    print (path + models[0])
    print ('Loading the model...')
    net = cv.dnn.readNetFromTorch(path + models[0])

    # Loading the image depending on the type
    if mode == VID:
        pass
    elif mode == IMG:
        img = cv.imread(FLAGS.image)
        img = resize_img(img)
        h, w  = img.shape[:2]
        blob = cv.dnn.blobFromImage(img, 1.0, (w, h),
            (103.939, 116.779, 123.680), swapRB=False, crop=False)

        net.setInput(blob)

        input('Shall I?')
        start = time.time()
        out = net.forward()
        end = time.time()
        print ('dfd')


        # Reshape the output tensor and add back in the mean subtraction, and
        # then swap the channel ordering
        out = out.reshape((3, out.shape[2], out.shape[3]))
        out[0] += 103.939
        out[1] += 116.779
        out[2] += 123.680
        out /= 255.0
        out = out.transpose(1, 2, 0)

        # Printing the inference time
        print ('The model ran in {:.4f} seconds'.format(end-start))

        # show the image
        cv.imshow('Input Image', img)
        cv.imshow('Stylized image', out)
        cv.waitKey(0)
