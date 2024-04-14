import matplotlib.pyplot as plt
import PIL
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid



def visualize_cifar10(dataset, samples_per_class, classes=None, seed=123):
    classes_full = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck']
    if classes is None:
        classes = classes_full

    # get image dimensions
    x, y = dataset[0]
    img_width = x.shape[1]
    img_height = x.shape[2]

    # calculate plot size in pixels
    img_padding = 2 # padding used in make_grid
    img_multiplier = 2 # used to increase image size
    px = 1/plt.rcParams['figure.dpi']  # pixel to inches conversion
    width = (img_width + img_padding) * img_multiplier * samples_per_class
    height = (img_height + img_padding) * img_multiplier * len(classes)

    # collect all samples (used to ensure same number of samples per class)
    x_data = torch.tensor(dataset.data)
    y_data = torch.tensor(dataset.targets)

    fig, ax = plt.subplots(figsize=(width*px, height*px))

    samples = []
    gen = torch.Generator().manual_seed(seed)
    # get samples for each class
    for i, cls in enumerate(classes):
        # plot class number
        cls_idx = classes_full.index(cls)
        ax.text(-4, 34 * i + 18, cls, ha='right', fontsize=16)

        # randomly sample from instances
        class_idxs = (y_data == cls_idx).nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(class_idxs), generator=gen)
        indices = indices[:samples_per_class]
        indices = class_idxs[indices]
        samples.extend([x_data[idx].permute(2, 0, 1) for idx in indices])

    # generate a single image containing all instances
    img = make_grid(samples, nrow=samples_per_class, padding=img_padding)
    img = img.permute(1, 2, 0)

    ax.imshow(img)
    ax.axis('off')
    return fig


def visualize_dataset(dataset, num_instances=25, max_col=5, img_size=128,
        seed=123):
    # select random indices
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)
    indices = indices[:num_instances]

    # get samples
    images = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)

    fig = visualize_images(images, img_size=img_size, max_col=max_col)
    return fig


def visualize_images(images, img_size, max_col=5):
    num_instances = len(images)
    num_rows, num_cols = _calculate_samples_grid(num_instances, max_col)

    # calculate plot size in pixels
    px = 1/plt.rcParams['figure.dpi']  # pixel to inches conversion
    w = img_size * num_cols * px
    h = img_size * num_rows * px

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(w, h))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= num_instances:
            continue
        img = images[i]

        # min-max normalization
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)

        img = ToPILImage()(img)
        img = img.resize((img_size, img_size), PIL.Image.LANCZOS)
        ax.imshow(img)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def _calculate_samples_grid(num_samples, max_col):
    remainder = 1 if (num_samples % max_col) > 0 else 0
    num_rows = (num_samples // max_col) + remainder
    num_cols = min(max_col, num_samples)
    return num_rows, num_cols


def visualize_templates(w, dim_data, max_col=10, img_size=128):
    # normalize each image individually to [0, 255]
    # get min and max values for each image
    w_min = w.min(dim=1)[0].unsqueeze(1)
    w_max = w.max(dim=1)[0].unsqueeze(1)
    # normalize images
    w_imgs = (w - w_min) / (w_max - w_min) * 255 # [w_min, w_max] -> [0, 255]

    # reshape to image
    w_imgs = w_imgs.view(w_imgs.shape[0], *dim_data) # [D, 3072] -> [D, C, H, W]
    # w_imgs = w_imgs.permute(0, 2, 3, 1) # [D, C, H, W] -> [D, H, W, C]
    # w_imgs = w_imgs.to(torch.uint8) # float32 to uint8

    num_rows, num_cols = _calculate_samples_grid(w_imgs.shape[0], max_col)
    fig_width = min(10, num_cols)
    fig_height = min(5, num_rows)

    fig = visualize_images(w_imgs, img_size, max_col)
    return fig

    # hacky way to adjust figure size according to the number of images
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    img_scaler = 128
    fig_width = num_cols*img_scaler*px
    fig_height = (1+num_rows)*img_scaler*px

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
            figsize=(fig_width, fig_height))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= num_imgs:
            continue
        ax.imshow(w_imgs[i])
        if num_imgs == 10:
            ax.set_title(cifar_classes[i])

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.suptitle('Learned Templates for CIFAR10', fontsize='xx-large')
    return fig


def visualize_losses(losses, losses_smooth):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(range(len(losses)), losses, c='steelblue')
    ax.plot(range(len(losses_smooth)), losses_smooth, c='crimson')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cross-Entropy Loss')
    fig.suptitle('Training Loss', fontsize='xx-large')
    return fig


def visualize_evaluation(accuracy_overall, accuracy_cls):
    (cls_names, cls_values) = zip(*accuracy_cls.items())

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.barh(cls_names, cls_values, facecolor='steelblue')
    ax.axvline(x=accuracy_overall, c='crimson', label='Overall Accuracy')
    ax.text(x=accuracy_overall, y=len(cls_names), weight='bold',
            s=str(f'{accuracy_overall:.2f}')+'%')

    for i, v in enumerate(cls_values):
        ax.text(v, i, str(f'{v:.2f}')+'%')

    fig.suptitle('Classifier Accuracy', fontsize='xx-large')
    fig.legend()
    return fig


def visualize_neighbors(samples, dataset, model, batch_size=64, k=10):
    # create loader to manage forward passes
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # generate image containing original samples
    img_test = make_grid(samples, nrow=1, normalize=True, scale_each=True)
    img_test = img_test.permute(1, 2, 0)

    # collect features for samples
    device = next(model.parameters()).device
    samples = samples.to(device)
    with torch.no_grad():
        samples_features = model(samples)

    # collect features for dataset
    x_data_features = []
    for (x, _) in loader:
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
        x_data_features.append(out)

    x_data_features = torch.cat(x_data_features, dim=0)

    # collect nearest neighbors at feature level
    indices_per_sample = []
    for sample in samples_features:
        sample = sample.unsqueeze(0).view(1, -1)
        dists = torch.norm(x_data_features - sample, dim=1, p=None)
        values, indices = dists.topk(k, largest=False)
        indices_per_sample.append(indices)

    indices_per_sample = torch.stack(indices_per_sample).flatten()

    # generate image containing nearest neighbors
    test = [dataset[i][0] for i in indices_per_sample]
    test = torch.stack(test)
    # test = dataset[indices_per_sample]

    img_neighbors = make_grid(test, nrow=k, normalize=True, scale_each=True)
    img_neighbors = img_neighbors.permute(1, 2, 0)

    # finish plotting
    # we duplicate test image info to add a ghost axes (blank space)
    heights = [img_test.shape[0], img_test.shape[0], img_neighbors.shape[0]]
    widths = [img_test.shape[1], img_test.shape[1], img_neighbors.shape[1]]

    fig_width = 10
    fig_height = fig_width * sum(heights) / sum(widths)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height),
            gridspec_kw={'width_ratios': widths})

    axes[0].imshow(img_test)
    axes[0].axis('off')
    # setting up ghost axes (spacing between test images and neighbors)
    axes[1].set_box_aspect(img_test.shape[0]/img_test.shape[1])
    axes[1].patch.set_alpha(0)
    axes[1].axis('off')
    axes[2].imshow(img_neighbors)
    axes[2].axis('off')

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # old way of adding spacing between test images and neighbors
    # w = (1 + k) / 2.0
    # fig.subplots_adjust(wspace=1/w, hspace=0, left=0, right=1, bottom=0, top=1)
    return fig


def visualize_cnn_filters(w, max_col=10):
    # normalize images
    w_min = torch.min(w.view(w.shape[0], -1), dim=1)[0]
    w_min = w_min.view(w_min.shape[0], 1, 1, 1)
    w_max = torch.max(w.view(w.shape[0], -1), dim=1)[0]
    w_max = w_max.view(w_max.shape[0], 1, 1, 1)
    w_imgs = (w - w_min) / (w_max - w_min) * 255 # [w_min, w_max] -> [0, 255]

    # reshape to image
    w_imgs = w_imgs.permute(0, 2, 3, 1) # [B, 3, H, W] -> [B, H, W, 3]
    w_imgs = w_imgs.to(torch.uint8) # float32 to uint8

    # plot templates for each kernel
    # calculating number of rows and columns to plot nicely
    num_filters = w.shape[0]
    remainder = 1 if (num_filters % max_col) > 0 else 0
    num_cols = min(max_col, num_filters)
    num_rows = (num_filters // max_col) + remainder

    fig_width = min(10, num_cols)
    fig_height = min(5, num_rows)

    # hacky way to adjust figure size according to the number of images
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    img_scaler = 64
    fig_width = num_cols*img_scaler*px
    fig_height = (1+num_rows)*img_scaler*px

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= num_filters:
            continue
        ax.imshow(w_imgs[i])

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig
