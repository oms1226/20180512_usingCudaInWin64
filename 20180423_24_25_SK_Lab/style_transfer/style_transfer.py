import sys

import matplotlib
from tensorflow.contrib.distributions.python.ops.bijectors import inline

#sys.path.insert(0,'drive/Colab Notebooks/20180424_SK_Lab2/20180423_24_25_SK_Lab')

# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
from sys import stderr

import tensorflow as tf
import numpy as np
import scipy.misc

import math
from argparse import ArgumentParser

from PIL import Image

from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['axes.grid'] = False

import vgg

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 300
VGG_PATH = 'imagenet-vgg-verydeep-19.mat' #pretraining 된 vgg weight를 가져온다!
POOLING = 'max'

CONTENT_LAYERS = ('relu4_2', 'relu5_2') #컨텐츠는 큰 형태를 유지하겠다.
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')# fine grain(relu1_1) 부터 큰 것까지 스타일을 골고루 적용하기 위함

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
            content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            learning_rate, beta1, beta2, epsilon, pooling,
            print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'
    with g.as_default(), g.device('/cpu:0'), tf.Session(config=config) as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session(config=config) as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (
                        1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            '''
            Compute the content loss

            Variables:
            content_weight: scalar constant we multiply the content_loss by.
            net[content_layer]: features of the current image, Tensor with shape [1, height, width, channels] --> 이 값이 interation 되면서 계속 바뀌는 것이다.
            content_features[content_layer]: features of the content image, Tensor with shape [1, height, width, channels] --> original 이미지에서 vgg를 돌려서 가지게되는 constant이다.
            '''
            l_content = content_weight * tf.reduce_sum((net[content_layer] - content_features[content_layer]) ** 2)

            content_losses.append(content_layers_weights[content_layer] * l_content)
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        for i in range(len(styles)):  # 스타일이 하나니깐 현재 로직에서는 무시해도 된다.
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, channels = map(lambda i: i.value, layer.get_shape())
                size = height * width * channels

                '''
                Compute the Gram matrix of the layer

                Variables:
                layer: features of the current image at style_layer, Tensor with shape [1, height, width, channels]
                gram: computed gram matrix with shape [channels, channels]
                '''
                feats = tf.reshape(layer, (-1, channels))  # 2 dimension으로 만드는 거란다!
                gram = tf.matmul(tf.transpose(feats), feats)  # outter product가 되고,
                gram /= size  # nomalization을 해주는 거란다.

                '''
                Compute the style loss

                Variables:
                style_layers_weights[style_layer]: scalar constant we multiply the content_loss by.
                  gram: computed gram matrix with shape [channels, channels]
                style_gram: computed gram matrix of the style image at style_layer with shape [channels, channels]
                '''
                style_gram = style_features[i][style_layer]
                l_style = style_layers_weights[style_layer] * tf.reduce_sum((gram - style_gram) ** 2)

                style_losses.append(l_style)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        '''
        Compute the TV loss

        Variables:
        tv_weight: scalar giving the weight to use for the TV loss.
        image: tensor of shape (1, H, W, 3) holding current image.
        '''
        tv_loss = tv_weight * (tf.reduce_sum((image[:, 1:, :, :] - image[:, :-1, :, :]) ** 2) + tf.reduce_sum(
            (image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2))  # X방향과 Y방향으로 차를 구해서 제곱하는 방법이다. 훌륭하다 다른데 활용해 보자

        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(
                            Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                    yield (
                        (None if last_step else i),
                        img_out
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest='content', help='content image',
                        metavar='CONTENT', required=True)
    parser.add_argument('--styles',
                        dest='styles',
                        nargs='+', help='one or more style images',
                        metavar='STYLE', required=True)
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='statistics printing frequency',
                        metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
                        dest='width', help='output width',
                        metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
                        dest='style_scales',
                        nargs='+', help='one or more style scales',
                        metavar='STYLE_SCALE')
    parser.add_argument('--network',
                        dest='network', help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
                        dest='content_weight_blend',
                        help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
                        metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
                        dest='style_layer_weight_exp',
                        help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
                        metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
                        dest='style_blend_weights', help='style blending weights',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
                        dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
                        metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
                        dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
                        metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
                        dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
                        metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
                        dest='initial', help='initial image',
                        metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
                        dest='initial_noiseblend',
                        help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
                        metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
                        dest='preserve_colors',
                        help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
                        dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
                        metavar='POOLING', default=POOLING)
    return parser


def style_transfer(content, style, args=None):
    parser = build_parser()
    args_list = ['--content', content, '--styles', style]
    if args is not None:
        args_list += [args]
    options = parser.parse_args(args_list)

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(options.content)
    style_images = [imread(style) for style in options.styles]

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                                    content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                                              target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight
                               for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image

    for iteration, image in stylize(
            network=options.network,
            initial=initial,
            initial_noiseblend=options.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=options.preserve_colors,
            iterations=options.iterations,
            content_weight=options.content_weight,
            content_weight_blend=options.content_weight_blend,
            style_weight=options.style_weight,
            style_layer_weight_exp=options.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            beta1=options.beta1,
            beta2=options.beta2,
            epsilon=options.epsilon,
            pooling=options.pooling,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations
    ):
        combined_rgb = image

    plt.figure()
    plt.imshow(content_image / 255)
    plt.xlabel('CONTENT IMAGE')
    plt.show()
    for i in range(len(style_images)):
        plt.figure()
        plt.imshow(style_images[i] / 255)
        plt.xlabel('STYLE IMAGE {}'.format(i))
    plt.show()
    plt.figure()
    plt.imshow(np.clip(combined_rgb, 0, 255) / 255)
    plt.xlabel('OUTPUT')
    plt.show()


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img

style_transfer('examples/1-content.jpg',
               'examples/2-style1.jpg')