import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def denorm(img_tensors, norm_schema):
    return img_tensors * norm_schema[1][0] + norm_schema[0][0]


def show_images(images, norm_schema, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], norm_schema), nrow=8).permute(1, 2, 0))


def show_batch(dl, norm_schema, nmax=64):
    for images, _ in dl:
        show_images(images, norm_schema, nmax)
        break


def accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)
