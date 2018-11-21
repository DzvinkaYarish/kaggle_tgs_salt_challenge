#IMPORTS

from jupyter_header import *


def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image, cmap="gray")
    axarr[1].imshow(mask, cmap="gray")
    axarr[0].grid(False)
    axarr[1].grid(False)
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')
    plt.show()

def plot_pictures_pairs_example(train_df):
    for i, idx in enumerate(train_df.index[:10]):
        img = train_df.loc[idx].images
        mask = train_df.loc[idx].masks
        plot2x2Array(img, mask)


def plot_imgs(train_df, max_images=60, grid_width=15, img_size_ori=101):
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for i, idx in enumerate(train_df.index[:max_images]):
        img = train_df.loc[idx].images
        mask = train_df.loc[idx].masks
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.text(1, img_size_ori-1, train_df.loc[idx].z, color="black")
        ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
        ax.text(1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")
    plt.show()


def predict_to_mask(raw_predictions, treashhold):
    return [np.round(prediction > treashhold) for prediction in raw_predictions]

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# TODO make tool for errors analysys