from monai.networks import convert_to_trt
from monai.data import save_net_with_metadata
from generator import define_generator
import torch

def initialize_generator(weights="/localhome/local-vennw/code/gan_training_data/checkpoints/200_net_G.pth"):
    generator = define_generator(
        input_nc=1,
        output_nc=1,
        ngf=64,
    )

    state_dict = torch.load(weights)
    generator.load_state_dict(state_dict)

    return generator

model = initialize_generator()

torchscript_model = convert_to_trt(
    model=model,
    precision="fp16",
    input_shape=[1, 1, 256, 256],
    use_trace=False,
    verify=True,
    device=0,
    rtol=1,
    atol=1,
)

save_net_with_metadata(torchscript_model, "tcia_rt_to_us")

model = torch.jit.load("tcia_rt_to_us.ts")
print("model loaded")
