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

    parser.add_argument('-md', '--model',
                type=str,
                help='The file path to the direct model.\
                 If this is specified, the model-path argument is \
                 not considered.')

    parser.add_argument('--show-original-image',
                type=bool,
                default=False,
                help='Whether or not to show the original image')

    parser.add_argument('--save-img-with-name',
                type=str,
                default='stylizedimage.png',
                help='The path to save the generated stylized image \
                       only when in image mode.')

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
    print ('[INFO] Loading the model...')

    if FLAGS.model is not None:
        model_to_load = FLAGS.model
    else:
        model_to_load = path + models[0]
    net = cv.dnn.readNetFromTorch(model_to_load)

    print ('[INFO] Model Loaded successfully!')

    # Loading the image depending on the type
    if mode == VID:
        pass
    elif mode == IMG:
        print ('[INFO] Reading the image')
        img = cv.imread(FLAGS.image)
        print ('[INFO] Image Loaded successfully!')

        img = resize_img(img, width=600)
        h, w  = img.shape[:2]
        blob = cv.dnn.blobFromImage(img, 1.0, (w, h),
            (103.939, 116.779, 123.680), swapRB=False, crop=False)

        print ('[INFO] Setting the input to the model')
        net.setInput(blob)

        print ('[INFO] Starting Inference!')
        start = time.time()
        out = net.forward()
        end = time.time()
        print ('[INFO] Inference Completed successfully!')

        # Reshape the output tensor and add back in the mean subtraction, and
        # then swap the channel ordering
        out = out.reshape((3, out.shape[2], out.shape[3]))
        out[0] += 103.939
        out[1] += 116.779
        out[2] += 123.680
        out /= 255.0
        out = out.transpose(1, 2, 0)

        # Printing the inference time
        print ('[INFO] The model ran in {:.4f} seconds'.format(end-start))

        # show the image
        if FLAGS.show_original_image:
            cv.imshow('Input Image', img)
        cv.imshow('Stylized image', out)
        print ('[INFO] Hit Esc to close!')
        cv.waitKey(0)

        if FLAGS.save_image_with_name is not None:
            cv.imwrite(out, FLAGS.save_image_with_name)
