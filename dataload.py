import torch
import torchvision
import numpy as np
import PIL

def get_ansi_color_code(r, g, b):
    return (
        16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
    )

def grayscale_code(p):
    if p < 8:
        return 16
    if p > 248:
        return 231
    return round(((p - 8) / 247) * 24) + 232

def format_pixel(pix):
    return "\x1b[48;5;{}m \x1b[0m".format(pix)


def show_image(img: PIL.Image.Image, grayscale: bool = True):

    h = img.height // 2
    w = img.width

    img = img.resize((w, h), PIL.Image.Resampling.LANCZOS)
    img_arr = np.asarray(img)

    for x in range(h):
        for y in range(w):
            pix = img_arr[x][y]
            if grayscale:
                print(
                    format_pixel(grayscale_code(pix)),
                    sep="",
                    end="",
                ),
            else:
                print(format_pixel(get_ansi_color_code(pix[0], pix[1], pix[2])), sep="", end="")
        print()


def load_mnist():
    mnist_data = torchvision.datasets.MNIST("datasets", download=True)
    data_loader = torch.utils.data.DataLoader(
        mnist_data, batch_size=4, shuffle=True, num_workers=6
    )
    return data_loader


def make_predictions(data_loader: torch.utils.data.DataLoader):
    img, target = data_loader.dataset.__getitem__(0)
    # img = torchvision.io.decode_image(img.open())
    # img = img.open()
    # Convert grayscale to RGB by repeating the channel 3 times
    show_image(img)
    
    img = img.convert("RGB")
    # img.show()
    print("PIL Image:")
    print(type(img))
    print(img)

    # Step 1: Initialize model with the best available weights
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    # print(weights.meta["categories"])
    model = torchvision.models.vit_b_16(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    print(batch.shape)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    data_loader = load_mnist()
    make_predictions(data_loader)
