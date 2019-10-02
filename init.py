import argparse
import time
import os
import subprocess
import cv2 as cv

FLAGS = None
VID = 'video'
IMG = 'image'

def predict(img, h, w):
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
    if FLAGS.print_inference_time:
        print ('[INFO] The model ran in {:.4f} seconds'.format(end-start))

    return out

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

    parser.add_argument('--save-image-with-name',
                type=str,
                default='stylizedimage.png',
                help='The path to save the generated stylized image \
                       only when in image mode.')

    parser.add_argument('--download-models',
                type=bool,
                default=False,
                help='If set to true all the pretrained models are downloaded, \
                    using the script in the downloads directory.')

    parser.add_argument('--print-inference-time',
                type=bool,
                default=False,
                help='If set to True, then the time taken for the model is output \
                    to the console.')

    FLAGS, unparsed = parser.parse_known_args()

    # download models if needed
    if FLAGS.download_models:
        subprocess.call(['./models/download.sh'])

    # Set the mode image/video based on the argparse
    if FLAGS.image is None:
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

    model_loaded_i = -1
    total_models = len(os.listdir(FLAGS.model_path))

    if FLAGS.model is not None:
        model_to_load = FLAGS.model
    else:
        model_loaded_i = 0
        model_to_load = path + models[model_loaded_i]
    net = cv.dnn.readNetFromTorch(model_to_load)

    print ('[INFO] Model Loaded successfully!')

    # Loading the image depending on the type
    if mode == VID:
        pass
        vid = cv.VideoCapture(0)
        while True:
            _, frame = vid.read()
            img = resize_img(frame, width=600)
            h, w  = img.shape[:2]
            out = predict(img, h, w)

            cv.imshow('Stylizing Real-time Video', out)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and FLAGS.model is None:
                model_loaded_i = (model_loaded_i + 1) % total_models
                model_to_load = path + models[model_loaded_i]
                net = cv.dnn.readNetFromTorch(model_to_load)
            elif key == ord('p') and FLAGS.model is None:
                model_loaded_i = (model_loaded_i - 1) % total_models
                model_to_load = path + models[model_loaded_i]
                net = cv.dnn.readNetFromTorch(model_to_load)

        vid.release()
        cv.destroyAllWindows()
    elif mode == IMG:
        print ('[INFO] Reading the image')
        img = cv.imread(FLAGS.image)
        print ('[INFO] Image Loaded successfully!')

        img = resize_img(img, width=600)
        h, w  = img.shape[:2]

        # Get the output from the pretrained model
        out = predict(img, h, w)

        # show the image
        if FLAGS.show_original_image:
            cv.imshow('Input Image', img)
        cv.imshow('Stylized image', out)
        print ('[INFO] Hit Esc to close!')
        cv.waitKey(0)

        if FLAGS.save_image_with_name is not None:
            out = cv.normalize(out, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            out *= 255
            out = out.astype("uint8")
            cv.imwrite(FLAGS.save_image_with_name, out)
