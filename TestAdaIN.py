import argparse
import PIL
import torch

from Augmentation import get_train_transforms, get_test_transforms
from Data import data_path_to_dataset
from IO import *
from TrainIMLE import get_model_optimizer
import Misc
from Utils import *
from Models import de_normalize

def test_adain_outputs(model, latent_spec,  args):
    """Shows information about the outputs of the model's AdaIN layer."""
    model = de_dataparallel(model)
    data_val = data_path_to_dataset(args.data_val,
        transform=get_test_transforms(args))
    x = data_val[0][0].unsqueeze(0).to(device)
    z = sample_latent_dict(latent_spec, bs=1)

    Misc.set_seed(args.other_seed) 
    z["latents"] = torch.randn(1, 1, 512, device=device, requires_grad=True)

    z_orig = z["latents"].detach().cpu().numpy()

    # optimizer = torch.optim.SGD([z["latents"]] + list(model.idx2block["11"].ip_method.parameters()), lr=.1)
    optimizer = torch.optim.SGD(model.decoder_blocks[0].norm1.parameters(), lr=.1)

    images = []
    for idx in range(1000):
        loss, pred, mask = model(x, z, mask_ratio=args.mask_ratio, return_all=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if idx == 0:
            images.append(pred)

        print(idx, loss.item())
    
    x = de_normalize(x)
    print("\n\n")
    print(torch.linalg.norm(torch.as_tensor(z_orig).to(device) - z["latents"]))

    image = torch.cat([x, images[0], pred, mask], dim=0)
    image = images_to_pil_image(image)
    image.show()

def get_args(args=None):
    """Returns parsed arguments, either from [args] or standard input."""
    P = argparse.ArgumentParser()
    P = add_hardware_args(P)
    P = add_train_imle_args(P)
    P = add_util_args(P)
    P = add_eval_imle_args(P)
    P = add_train_imle_debugging_args(P)

    P.add_argument("--adain_name", required=True,
        help="Name of AdaIN module")
    P.add_argument("--adain_block", required=True,
        help="Name of AdaIN module")
    P.add_argument("--other_seed", required=True, type=int,
        help="Name of AdaIN module")

    args = P.parse_args() if args is None else P.parse_args(args)
    args.save_folder = args.save_folder.strip("/")
    return args

if __name__ == "__main__":
    args = get_args()
    Misc.set_seed(args.seed)
    model, _ = get_model_optimizer(args)
    model = de_dataparallel(model)

    latent_spec = model.get_latent_spec(
        mask_ratio=args.mask_ratio,
        input_size=args.input_size)
        
    Misc.pretty_print_args(args)

    print(model)
    # assert 0
    
    tqdm.write(f"LATENT_SPEC\n{latent_spec}")

    _ = test_adain_outputs(model, latent_spec, args)
