#!/usr/bin/env python3

import abc
import math
import functools

import cv2
import numpy as np
import numpy.random as random


def clip(img):
    return np.clip(img, 0, 1)


def random_log_uniform(variance):
    min_value = np.log(1 - variance)
    max_value = -np.log(1 - variance)
    return np.exp(random.uniform(min_value, max_value))


class Transform(abc.ABC):
    def __init__(self, prob=1.0, **kw):
        self.prob = prob

        for k, v in kw.items():
            setattr(self, k, v)

    @staticmethod
    def get_types(args):
        return [a.dtype for a in args]

    @staticmethod
    def get_shapes_sizes(args):
        return [len(a.shape) for a in args]

    def __call__(self, *args, return_backform=False, return_backform_coord=False):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
            return_always_list = True
        else:
            return_always_list = False

        if any(isinstance(arg, (list, tuple)) for arg in args):
            raise TransformError(self.name(), 'args should not contain a list or a tuple')

        params = list(args[0].shape[:2])
        params.extend(self.get_params(*params))

        apply_transform = random.random() < self.prob
        if apply_transform:
            try:
                result = [self.transform(a, *params) for a in args]
            except Exception as e:
                raise e
        else:
            result = list(args)

        if return_backform:
            result.append(functools.partial(self.backform, *params))
        if return_backform_coord:
            result.append(functools.partial(self.backform_coord, *params))

        return result if return_always_list or len(result) > 1 else result[0]

    def get_params(self, height, width):
        return ()

    @abc.abstractmethod
    def transform(self, img):
        pass

    def backform(self, params, img):
        raise NotImplementedError

    def backform_coord(self, params, point):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__


class Mirror(Transform):
    def get_params(self, height, width):
        return random.random() < 0.5, random.random() < 0.5

    def transform(self, img, height, width, hmirror, vmirror):
        if hmirror:
            img = np.fliplr(img)
        return img


class CropMinSize(Transform):
    def getlr(self, width, s):
        left = random.randint(0, width - s)
        right = random.randint(left + s, width)
        if random.random() < 0.5:
            left, right = width - right, width - left
        assert left <= right
        return left, right

    def get_params(self, height, width):
        top, bottom = self.getlr(height, self.min_size)
        left, right = self.getlr(width, self.min_size)
        return top, bottom, left, right

    def transform(self, img, height, width, top, bottom, left, right):
        return img[top:bottom, left:right]


class CropFixedSize(Transform):
    """
    Crop by fixed sizes
    * fixed_sizes: tuple(height, width)
    * align: 'random' as default, 'center'
    """

    def __init__(self, align='random', padding=None, padding_modes=('constant', 'reflect', ), **kw):
        self.align = align
        self.padding = padding
        self.padding_modes = padding_modes
        super(CropFixedSize, self).__init__(**kw)

    def getlr(self, width, size):
        if width < size:
            if self.padding == 'center':
                left = -(size - width) // 2
            elif self.padding == 'random':
                left = -random.randint(size - width + 1)
            else:
                raise Exception('Incorrect set padding = "%s"' % self.padding)
            padded = True
        else:
            if self.align == 'center':
                left = (width - size) // 2
            elif self.align == 'random':
                left = random.randint(width - size + 1)
            else:
                raise Exception('Incorrect set align = "%s"' % self.align)
            padded = False
        right = left + size
        assert left <= right
        assert (left <= 0 and width <= right) or (0 <= left and right <= width)
        return left, right, padded

    def get_params(self, height, width):
        top, bottom, h_padded = self.getlr(height, self.fixed_sizes[0])
        left, right, v_padded = self.getlr(width, self.fixed_sizes[1])
        padding_mode = random.choice(self.padding_modes) if h_padded or v_padded else None
        return top, bottom, left, right, padding_mode

    @staticmethod
    def get_bordered_(left, right, width):
        return max(left, 0), min(right, width)

    @staticmethod
    def get_pad_width_(left, right, width):
        return max(-left, 0), max(right - width, 0)

    def transform(self, img, height, width, top, bottom, left, right, padding_mode):
        top_, bottom_ = self.get_bordered_(top, bottom, height)
        left_, right_ = self.get_bordered_(left, right, width)
        ret = img[top_:bottom_, left_:right_]

        if padding_mode:
            pad_width = [
                self.get_pad_width_(top, bottom, height),
                self.get_pad_width_(left, right, width),
            ]
            if len(img.shape) == 3:
                pad_width.append((0, 0))
            ret = np.pad(ret, pad_width=tuple(pad_width), mode=padding_mode)
        return ret

    def backform(self, height, width, top, bottom, left, right, padding_mode, img):
        shape = list(img.shape)
        shape[0] = height
        shape[1] = width
        result = np.zeros(shape, dtype=img.dtype)

        top_, bottom_ = self.get_bordered_(top, bottom, height)
        left_, right_ = self.get_bordered_(left, right, width)
        result[top_:bottom_, left_:right_] = img
        return result

    def backform_coord(self, height, width, top, bottom, left, right, padding_mode, p):
        return p + np.array((left, top))


class Scale(Transform):
    """
    Represent resize not adjusted to image aspect ratio.
    """

    def __init__(self, base_size=None, min_scale=1.0, max_scale=1.0, interpolation_method=cv2.INTER_LINEAR):
        """
        :param base_size: base image size (height, width) produced when scale = 1.
               If None passed scale will be relative to original image size.
        :param min_scale: min value of scale. default=1.0
        :param max_scale: max value of scale. default=1.0
        :param interpolation_method: interpolation method for cv2.resize
        """
        super().__init__()
        self.base_size = base_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation_method = interpolation_method

        assert len(base_size) == 2, 'base image size should have 2 dimensions, got {}'.format(len(base_size))

    def get_params(self, height, width):
        scale = random.uniform(self.min_scale, self.max_scale)

        return [scale]

    def transform(self, img, height, width, scale):
        base_height, base_width = self.base_size or (height, width)
        base_height = int(np.round(base_height * scale))
        base_width = int(np.round(base_width * scale))
        img = cv2.resize(img, (base_width, base_height), interpolation=self.interpolation_method)
        return clip(img)


class Blur(Transform):
    def get_params(self, height, width):
        return np.random.uniform(self.min_sigma, self.max_sigma), self.kernel_size

    def transform(self, img, height, width, sigma, kernel_size):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
        return clip(img)


class ScaleBasedSizes(Transform):
    """
    Rescale to multiply on base size
    * based_sizes: tuple (height, width)
    * min_scale
    * max_scale
    * interpolation: scaling method
    """

    def __init__(self, interpolation=cv2.INTER_CUBIC, **kw):
        self.interpolation = interpolation
        super().__init__(**kw)

    def get_params(self, height, width):
        baseh, basew = self.based_sizes
        min_scale = max(baseh * self.min_scale / height, basew * self.min_scale / width)
        max_scale = max(baseh * self.max_scale / height, basew * self.max_scale / width)

        # make upscaling and downscaling symmetrically
        log_scale = random.uniform(np.log(min_scale), np.log(max_scale))
        scale = np.exp(log_scale)

        return scale,

    def transform(self, img, height, width, scale):
        new_h = int(np.round(height * scale))
        new_w = int(np.round(width * scale))
        ret = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
        if img.shape[-1] == 1:
            ret = np.expand_dims(ret, axis=2)
        return clip(ret)

    def backform(self, height, width, scale, img):
        ret = cv2.resize(img, (width, height), interpolation=self.interpolation)
        if img.shape[-1] == 1:
            ret = np.expand_dims(ret, axis=2)
        return clip(ret)

    def backform_coord(self, height, width, scale, p):
        return p / scale


class Rotate(Transform):
    def __init__(self, **kw):
        kw.setdefault('crop', True)
        super().__init__(**kw)

    def get_params(self, height, width):
        return random.uniform(self.min_angle, self.max_angle),

    def transform(self, img, height, width, angle):
        transform_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        img = cv2.warpAffine(img, transform_matrix, (width, height))
        if self.crop:
            rad = math.fabs(angle / 180. * math.pi)
            size = int(min(width, height) // (math.cos(rad) + math.sin(rad)))
            img = img[(height - size) // 2:(height + size) // 2, (width - size) // 2:(width + size) // 2]
        return clip(img)


class Blend(Transform):
    def transform(self, img, height, width):
        return clip(img * self.alpha + (1 - self.alpha) * self.blend)


class Grayscale(Transform):
    def transform(self, img, height, width):
        original_dtype = img.dtype
        if img.dtype != np.float32:
            img = img.astype('float32')
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if gs.dtype != original_dtype:
            gs = gs.astype(original_dtype)
        return np.repeat(np.expand_dims(gs, 2), 3, 2)


class Saturation(Transform):
    def __init__(self, var, **kw):
        super().__init__(**kw)
        self.var = var
        assert 0 <= var < 1, 'variance should be in [0, 1] range.'

    def get_params(self, height, width):
        return random_log_uniform(self.var),

    def transform(self, img, height, width, alpha):
        gs = Grayscale()(img)
        return Blend(alpha=alpha, blend=gs)(img)


class Brightness(Transform):
    def __init__(self, var, **kw):
        super().__init__(**kw)
        self.var = var
        assert 0 <= var < 1, 'variance should be in [0, 1] range.'

    def get_params(self, height, width):
        return random_log_uniform(self.var),

    def transform(self, img, height, width, alpha):
        return Blend(alpha=alpha, blend=np.zeros_like(img))(img)


class Contrast(Transform):
    def __init__(self, var, **kw):
        super().__init__(**kw)
        self.var = var
        assert 0 <= var < 1, 'variance should be in [0, 1] range.'

    def get_params(self, height, width):
        return random_log_uniform(self.var),

    def transform(self, img, height, width, alpha):
        gs = Grayscale()(img)
        gs.fill(gs.mean())
        return Blend(alpha=alpha, blend=gs)(img)


class ColorJitter(Transform):
    """
    * jit_type_prob
    * saturation
    * brightness
    * contrast
    """

    def get_params(self, height, width):
        perm = []
        if self.saturation and random.random() < self.jit_type_prob:
            perm.append(Saturation(var=self.saturation))
        if self.brightness and random.random() < self.jit_type_prob:
            perm.append(Brightness(var=self.brightness))
        if self.contrast and random.random() < self.jit_type_prob:
            perm.append(Contrast(var=self.contrast))
        random.shuffle(perm)
        return perm,

    def transform(self, img, height, width, perm):
        for t in perm:
            img = t(img)
        return img


class Add(Transform):
    def get_params(self, height, width):
        return random.uniform(-self.delta, self.delta),

    def transform(self, img, height, width, delta):
        return clip(img + delta)


class AddChannelWise(Transform):
    def get_params(self, height, width):
        return np.random.uniform(-self.delta, self.delta, size=3),

    def transform(self, img, height, width, delta):
        transformed = img.copy()
        for i in range(3):
            transformed[:, :, i] = img[:, :, i] + delta[i]
        return clip(transformed)


class Multiply(Transform):
    def __init__(self, mult, **kw):
        super().__init__(**kw)
        self.mult = mult
        assert 0 <= mult <= 1, 'mult should be in [0, 1] range.'

    def get_params(self, height, width):
        return random_log_uniform(self.mult),

    def transform(self, img, height, width, mult):
        return clip(img * mult)


class AddNoise(Transform):

    def __init__(self, max_normal_std, speckle_scale, **kw):
        super().__init__(**kw)
        self.max_normal_std = max_normal_std
        self.speckle_scale = speckle_scale

    def get_params(self, height, width):
        return np.random.choice(['normal', 'speckle']),

    def transform(self, img, height, width, mode):
        dtype = img.dtype
        n_dims = len(img.shape)

        if n_dims == 2:
            img = np.expand_dims(img, -1).copy()

        channels = img.shape[-1]
        noise_mean = tuple(np.zeros(channels))
        if mode == 'normal':
            noise = np.zeros_like(img, dtype='uint8')
            stds = np.random.randint(self.max_normal_std, size=channels)
            cv2.randn(noise, noise_mean, stds)
            ret = img + noise / 255
        elif mode == 'speckle':
            noise = np.random.normal(loc=1., scale=self.speckle_scale, size=img.shape)
            ret = img * noise

        if len(ret.shape) != n_dims:
            ret = ret[..., 0]

        return clip(ret).astype(dtype)


class Sharpen(Transform):
    def get_params(self, height, width):
        return random.uniform(0, self.alpha), random.uniform(0, self.lightness)

    def transform(self, img, height, width, alpha, lightness):
        kernel = np.array([[-2, -2, -2], [-2, 32 + lightness, -2], [-2, -2, -2]]) / 16
        sharpened = cv2.filter2D(img, -1, kernel)
        result = sharpened * alpha + (1 - alpha) * img
        return clip(result)


class MotionBlur(Transform):
    def get_params(self, height, width):
        # random.uniform(0, 360 * 2) gives expected rotation 0
        return random.uniform(0, 360 * 2), random.randint(self.min_size, self.max_size)

    def transform(self, img, height, width, angle, size):
        img_dtype = img.dtype
        img = (img * 255).astype('uint8')
        line = np.zeros((2, size))
        line[0] = np.arange(-int(size / 2), math.ceil(size / 2))
        if size % 2 == 0:
            line[0, size // 2:] += 1
        angle = angle * math.pi / 180
        sin = math.sin(angle)
        cos = math.cos(angle)
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        indexes = (rotation_matrix.dot(line) + size / 2.0).astype(int)
        kernel = np.zeros((size, size))
        kernel[indexes.tolist()] = 1.0 / size
        result = cv2.filter2D(img, -1, kernel)
        return clip(result / 255).astype(img_dtype)


class ProjectiveTransform(Transform):
    def get_params(self, height, width):
        a = random.uniform(0.9, 1.1)
        b = random.uniform(-0.1, 0.1)
        c = random.uniform(-0.15, 0.30) * width

        d = random.uniform(-0.1, 0.1)
        e = random.uniform(0.9, 1.1)
        f = random.uniform(-0.15, 0.30) * height

        g = random.uniform(-0.1, 0.1) / width
        h = random.uniform(-0.1, 0.1) / height
        i = random.uniform(1.1, 1.6)
        return np.array([[a, b, c], [d, e, f], [g, h, i]]),

    def transform(self, img, height, width, transform_matrix):
        return clip(cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]), borderMode=0))


class Elastic(Transform):
    def get_params(self, height, width):
        a0 = [
            width / np.random.uniform(self.ampl_divider_min, self.ampl_divider_max) / self.order
            for _ in range(self.order)
        ]
        a1 = [
            height / np.random.uniform(self.ampl_divider_min, self.ampl_divider_max) / self.order
            for _ in range(self.order)
        ]
        w0 = [np.random.uniform(self.freq_mult_min, self.freq_mult_max) / height for _ in range(self.order)]
        w1 = [np.random.uniform(self.freq_mult_min, self.freq_mult_max) / width for _ in range(self.order)]
        return a0, w0, a1, w1, self.order

    def transform(self, img, height, width, a0, w0, a1, w1, order):
        img = self.axis_transform(img, a0, w0, order)
        img = np.rot90(img)
        img = self.axis_transform(img, a1, w1, order)
        img = np.rot90(img, -1)
        return img

    @staticmethod
    def axis_transform(img, a, width, order):
        def shift(x):
            return np.sum([a[i] * np.sin(2.0 * np.pi * x * width[i]) for i in range(order)])

        for i in range(img.shape[0]):
            img[i, :] = np.roll(img[i, :, ], int(shift(i)), axis=0)
        return img


class Shadow(Transform):
    def get_params(self, height, width):
        zones_count = np.random.randint(self.max_zones) + 1
        zones_params = []
        for _ in range(zones_count):
            brightness_inv = 1 - np.random.uniform(self.brightness_min, self.brightness_max)
            borders = (np.array([height, width]) * self.offset_part).astype('int32')
            lr = np.random.randint(-borders[1], borders[1])
            ud = np.random.randint(-borders[0], borders[0])
            angle = np.random.randint(-90, 90)
            rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            offset_matrix = np.array([[1, 0, lr], [0, 1, ud]]).astype('float32')
            ksize = np.random.choice(np.arange(3, 15, 2))
            sigma = np.random.randint(3, 30)
            zones_params.append((rotation_matrix, offset_matrix, ksize, sigma, borders, brightness_inv))
        return zones_params,

    def transform(self, img, height, width, zones_params):
        img_dtype = img.dtype
        for rotation_matrix, offset_matrix, ksize, sigma, borders, brightness_inv in zones_params:
            mask = np.zeros((height, width))
            mask[borders[0]:mask.shape[0] - borders[0], borders[1]:mask.shape[1] - borders[1]] = brightness_inv
            mask = (mask * 255).astype('uint8')
            mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
            mask = cv2.warpAffine(mask, offset_matrix, (width, height))
            mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            mask = 1 - mask / 255
            img = img * mask[:, :, np.newaxis]
        return img.astype(img_dtype)


class Roll(Transform):

    def __init__(self, max_shift, axes, mode=None, **kwargs):
        """
        :param max_shift: maximal absoulte value for shift.
        :param axes: axes for rolling. Can be int or tuple of ints.
        :param mode: When multiple axes specified determine how to perform roll.
                     'single' - perform roll on a single randomly selected axis
                     'all' - perform roll on each of specified axis.
                     Pass None in case of single axis. Ignored when single axis is specified.
        """
        super().__init__(**kwargs)
        assert mode is not None or type(axes) is int, 'Mode should be specified for multiple axis.'
        assert mode in [None, 'all', 'single'], "Available modes is 'all', 'single' or None."

        self.max_shift = max_shift
        self.axes = axes
        self.mode = mode

    def get_params(self, height, width):
        if type(self.axes) is int:
            shift = np.random.randint(-self.max_shift, self.max_shift)
            return [[(self.axes, shift)]]

        if self.mode == 'all':
            selected_axes = self.axes
        elif self.mode == 'single':
            selected_axes = [np.random.choice(self.axes)]
        else:
            raise NotImplementedError

        shift_range = np.repeat(self.max_shift, len(selected_axes)) if type(self.max_shift) == int else self.max_shift

        axes_shifts = [(axis, np.random.randint(-max_shift, max_shift)) for axis, max_shift in
                       zip(selected_axes, shift_range)]

        return [axes_shifts]

    def transform(self, img, height, width, axes_shifts):
        for axis, shift in axes_shifts:
            img = np.roll(img, shift, axis=axis)

        return img


class Rot90(Transform):
    """
    Rotates image on 0, 90, 180, 270 degree randomly.
    """

    def get_params(self, height, width):
        turns = np.random.randint(0, 4)
        return turns,

    def transform(self, img, height, width, turns):
        return np.rot90(img, turns)
