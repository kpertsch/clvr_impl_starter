import numpy as np
import torch


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


def get_padding(seq, replace_dim, size, val=0.0):
    """Returns padding tensor of same shape as seq, but with the target dimension replaced to 'size'.
       All values in returned array are set to 'val'."""
    seq_shape = seq.shape
    if isinstance(seq, torch.Tensor):
        return val * torch.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim+1:], device=seq.device)
    else:
        return val * np.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim + 1:])


def stack_with_separator(tensors, dim, sep_width=2, sep_val=0.0):
    """Stacks list of tensors along given dimension, adds separator, brings to range [0...1]."""
    tensors = [(t + 1) / 2 if t.min() < 0.0 else t for t in tensors]
    stack_tensors = tensors[:1]
    if len(tensors) > 1:
        for tensor in tensors[1:]:
            assert tensor.shape == tensors[0].shape  # all stacked tensors must have same shape!
        separator = get_padding(stack_tensors[0], replace_dim=dim, size=sep_width, val=sep_val)
        for tensor in tensors[1:]:
            stack_tensors.extend([separator, tensor])
        stack_tensors = [np.concatenate(stack_tensors, axis=dim)]
    return stack_tensors[0]


def make_image_seq_strip(imgs, n_logged_samples=5, sep_val=0.0):
    """Creates image strip where each row contains full rollout of sequence [each element of list makes one row]."""
    plot_imgs = stack_with_separator(imgs, dim=3, sep_val=sep_val)[:n_logged_samples]
    return stack_with_separator([t[:, 0] for t in np.split(plot_imgs, int(plot_imgs.shape[1] / 1), 1)],
                                dim=3, sep_val=sep_val)


