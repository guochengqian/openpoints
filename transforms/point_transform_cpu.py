import numpy as np
import torch
from .point_transformer_gpu import DataTransforms


@DataTransforms.register_module()
class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):  
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not torch.is_tensor(data[key]):
                data[key] = torch.from_numpy(np.array(data[key]))
            # if key in ['y', 'label', 'labels', 'ind','indices', 'cls']: 
            #     data[key] = data[key].to(torch.long)
            # else:
            #     data[key] = data[key].to(torch.float32)
        return data


@DataTransforms.register_module()
class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1], **kwargs):
        self.angle = angle

    def __call__(self, data):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        data['pos'] = np.dot(data['pos'], np.transpose(R))
        return data


@DataTransforms.register_module()
class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False, **kwargs):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        data['pos'] *= scale
        return data


@DataTransforms.register_module()
class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
        self.shift = shift

    def __call__(self, data):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        data['pos'] += [shift_x, shift_y, shift_z]
        return data


@DataTransforms.register_module()
class RandomScaleAndTranslate(object):
    def __init__(self,
                 scale=[0.9, 1.1],
                 shift=[0.2, 0.2, 0],
                 scale_xyz=[1, 1, 1],
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.shift = shift

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        scale *= self.scale_xyz

        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        data['pos'] = np.add(np.multiply(data['pos'], scale), [shift_x, shift_y, shift_z])
        
        return data


@DataTransforms.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['pos'][:, 0] = -data['pos'][:, 0]
        if np.random.rand() < self.p:
            data['pos'][:, 1] = -data['pos'][:, 1]
        return data


@DataTransforms.register_module()
class RandomJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.sigma = jitter_sigma
        self.clip = jitter_clip

    def __call__(self, data):
        jitter = np.clip(self.sigma * np.random.randn(data['pos'].shape[0], 3), -1 * self.clip, self.clip)
        data['pos'] += jitter
        return data


@DataTransforms.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None, **kwargs):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.rand() < self.p:
            lo = np.min(data['x'][:, :3], 0, keepdims=True)
            hi = np.max(data['x'][:, :3], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data['x'][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data['x'][:, :3] = (1 - blend_factor) * data['x'][:, :3] + blend_factor * contrast_feat
            """vis
            from openpoints.dataset import vis_points
            vis_points(data['pos'], data['x']/255.)
            """
        return data


@DataTransforms.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05, **kwargs):
        self.p = p
        self.ratio = ratio

    def __call__(self, data):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data['x'][:, :3] = np.clip(tr + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005, **kwargs):
        self.p = p
        self.std = std

    def __call__(self, data):
        if np.random.rand() < self.p:
            noise = np.random.randn(data['x'].shape[0], 3)
            noise *= self.std * 255
            data['x'][:, :3] = np.clip(noise + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, **kwargs):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(data['x'][:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        data['x'][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data


@DataTransforms.register_module()
class RandomDropColor(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.p = color_drop

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['x'][:, :3] = 0
        return data
