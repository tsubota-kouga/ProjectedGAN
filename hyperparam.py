import torch


class HyperParam:
    dataset = "afhq-dog"
    dataroot = f"/mnt/My Files2/datasets/{dataset}/preprocess/"
    dataset_path = {
        "celeba-hq": "/mnt/My Files/celeba-hq",
        "ffhq": "/mnt/My Files/ffhq",
        "afhq-dog": "/mnt/My Files/afhq/*/dog/",
        "afhq-cat": "/mnt/My Files/afhq/*/cat/",
        "afhq-wild": "/mnt/My Files/afhq/*/wild/",
        "afhq": "/mnt/My Files/afhq/*/*/",
        }
    log_dir = "./log"
    profile_dir = "./profile"
    model_dir = "/mnt/My Files/projected-gan-model/"
    batch_sizeD = 4
    batch_sizeG = 4
    gradient_accumulation = 5
    use_fp16 = False
    reload = False

    class AdaptivePeudoAugment:
        start_epoch = 0.  #float("nan") #not used
        initial_prob = 0.
        threshold = 0.95
        speed = 1e-6
        iteration_per = 4
        max_prob = 0.3

    dlr = 2.0e-4
    glr = 2.0e-4
    move_average_rate = 0.999
    truncation_trick_rate = 0.7

    latent_dim = 256

    generator_channels_info = [
        (1024, 1024),  # 8x8
        (1024, 512),  # 16x16
        (512, 256),  # 32x32
        (256, 128),  # 64x64
        (128, 64),  # 128x128
        (64, 32),  # 256x256
        # (32, 3),  # 512x512
        # (3, 3),  # 1024x1024
        ]

    feature_channels = [24, 40, 112, 320]
    projected_channels = [64, 128, 512, 1024]
    # projected_channels = [128, 512, 1024, 1024]

    discriminator_channels_info = [
        (64, 128),
        (128, 256),
        (256, 512),
        ]

    betas = (0., 0.99)
    eps = 1e-8
    device = torch.device("cuda")
    num_epoch = 5000
    weight_decay = 0
    max_level = len(generator_channels_info) + 1
    resolution = 2 ** (max_level + 1)
