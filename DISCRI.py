# Discriminative_full_original_print_windows_safe.py
# ✅ Windows safe (num_workers>0) with __main__ + freeze_support
# ✅ Keeps your ORIGINAL accuracy display style: test acc = batch mean ± std
# ✅ Faster than your original: uses lists (not np.append), pin_memory, non_blocking
# ✅ Plots ONLY ONCE at the end (no image every 1000 steps)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# -----------------------
# helper
# -----------------------
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# CIFAR-100 class names
class_names = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle',
    'bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel',
    'can','castle','caterpillar','cattle','chair','chimpanzee','clock',
    'cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster',
    'house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard',
    'lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree',
    'pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine',
    'possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal',
    'shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel',
    'streetcar','sunflower','sweet_pepper','table','tank','telephone',
    'television','tiger','tractor','train','trout','tulip','turtle',
    'wardrobe','whale','willow_tree','wolf','woman','worm'
]

# -----------------------
# normalization (same as yours)
# -----------------------
MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

def unnormalize(img_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)

# -----------------------
# Model (your code)
# -----------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block
    - expand_ratio: e.g. 2 (expand channels = in_ch * 2)
    - stride: 1 or 2
    - use residual when stride==1 and in_ch==out_ch
    """
    def __init__(self, in_ch, out_ch, stride=2, expand_ratio=2.5):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1x1 expand (only if expand_ratio != 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
        else:
            hidden = in_ch

        # 3x3 depthwise
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]

        # 1x1 project (linear)
        layers += [
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        n_classes = params["n_classes"]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        self.ds_conv2 = DepthwiseSeparableConv(32, 64, stride=2)

        self.ir_conv3 = InvertedResidual(64, 128, stride=2, expand_ratio=2.5)
        
        self.ds_conv4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.drop = nn.Dropout(p=0.05)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ds_conv2(x)
        x = self.ir_conv3(x)
        x = self.ds_conv4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

# -----------------------
# Prediction visualization (same interface as yours)
# -----------------------
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img_t = torch.tensor(img)
    img_show = unnormalize(img_t).permute(1, 2, 0).numpy()
    plt.imshow(img_show)

    predicted_label = int(np.argmax(predictions_array))
    color = "green" if predicted_label == int(true_label) else "red"

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[int(true_label)]
    ), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], int(true_label[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(100), predictions_array)
    plt.ylim([0, 1])
    predicted_label = int(np.argmax(predictions_array))
    thisplot[predicted_label].set_alpha(0.9)
    thisplot[true_label].set_alpha(0.9)

# -----------------------
# Main (Windows-safe)
# -----------------------
def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("gpu count:", torch.cuda.device_count())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # -----------------------
    # Data augmentation + normalization
    # -----------------------
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=1, magnitude=4),     # optional
        T.ToTensor(),
        T.Normalize(MEAN, STD), 
        #T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # optional
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # -----------------------
    # DataLoaders (faster + GPU-friendly)
    # -----------------------
    NUM_WORKERS = 4  # if error/slow, try 2; if still error, use 0
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100("data", train=True, download=True, transform=train_tf),
        batch_size=512, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100("data", train=False, download=True, transform=test_tf),
        batch_size=512, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    train_iterator = iter(cycle(train_loader))
    test_iterator  = iter(cycle(test_loader))

    print(f"> Size of training dataset {len(train_loader.dataset)}")
    print(f"> Size of test dataset {len(test_loader.dataset)}")

    # -----------------------
    # Model
    # -----------------------
    params = {"n_channels": 3, "n_classes": 100}
    N = Classifier(params).to(device)

    n_params = len(torch.nn.utils.parameters_to_vector(N.parameters()))
    print(f"> Number of parameters {n_params}")

    if n_params > 100000:
        print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

    # -----------------------
    # Optimiser + scheduler
    # -----------------------
    max_steps  = 10000
    eval_every = 1000

    optimiser = torch.optim.AdamW(N.parameters(), lr=8e-3,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_steps)

    plot_data = []
    steps = 0

    # -----------------------
    # Train loop
    # -----------------------
    while steps < max_steps:
        chunk = min(eval_every, max_steps - steps)

        # Faster than np.append in loop
        train_loss_list = []
        train_acc_list  = []
        test_acc_list   = []

        # Train chunk
        for _ in range(chunk):
            x, t = next(train_iterator)
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            p = N(x)
            pred = p.argmax(dim=1, keepdim=True)

            loss = F.cross_entropy(p, t, label_smoothing=0.05)
            loss.backward()
            optimiser.step()
            scheduler.step()
            steps += 1

            train_loss_list.append(loss.detach().item())
            train_acc_list.append(pred.eq(t.view_as(pred)).float().mean().item())

        # Test (ORIGINAL style: batch accuracy mean ± std)
        N.eval()
        with torch.no_grad():
            for x, t in test_loader:
                x = x.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                p = N(x)
                pred = p.argmax(dim=1, keepdim=True)
                test_acc_list.append(pred.eq(t.view_as(pred)).float().mean().item())
        N.train()

        # Print in your original format
        train_loss = float(np.mean(train_loss_list))
        train_acc_mean = float(np.mean(train_acc_list))
        train_acc_std  = float(np.std(train_acc_list))
        test_acc_mean  = float(np.mean(test_acc_list))
        test_acc_std   = float(np.std(test_acc_list))

        print(
            "steps: {:.0f}, train loss: {:.3f}, train acc: {:.3f}±{:.3f}, test acc: {:.3f}±{:.3f}".format(
                steps,
                train_loss,
                train_acc_mean,
                train_acc_std,
                test_acc_mean,
                test_acc_std
            )
        )

        plot_data.append([steps, train_acc_mean, train_acc_std, test_acc_mean, test_acc_std])

    print(f"\nDone. Total steps: {steps}")
    print(f"Total parameters: {n_params}")

    # Plot ONCE at the end (no image spam)
    plt.figure()
    plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], "-", label="Train accuracy")
    plt.fill_between(
        [x[0] for x in plot_data],
        [x[1] - x[2] for x in plot_data],
        [x[1] + x[2] for x in plot_data],
        alpha=0.2
    )
    plt.plot([x[0] for x in plot_data], [x[3] for x in plot_data], "-", label="Test accuracy")
    plt.fill_between(
        [x[0] for x in plot_data],
        [x[3] - x[4] for x in plot_data],
        [x[3] + x[4] for x in plot_data],
        alpha=0.2
    )
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.show()

    # -----------------------
    # Visualize predictions (same as your original; optional)
    # -----------------------
    test_images, test_labels = next(test_iterator)
    test_images, test_labels = test_images.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)

    N.eval()
    with torch.no_grad():
        test_preds = torch.softmax(N(test_images), dim=1).data.cpu().numpy()
    N.train()

    num_rows = 8
    num_cols = 4
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, test_preds, test_labels.cpu().numpy(), test_images.cpu().numpy())
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, test_preds, test_labels.cpu().numpy())
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ✅ required on Windows for DataLoader multiprocessing
    torch.multiprocessing.freeze_support()
    main()
