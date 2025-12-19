from model.TeD import TeD
from torchsummary import summary
from utils.util import parse_arguments
import torch
import contextlib
import io

def get_total_params(model, input_size):
    # Redirect stdout to capture summary output
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        summary(model, input_size, batch_size=1)
        output = buf.getvalue()

    # Find and extract "Total params" line
    for line in output.splitlines():
        if "Total params" in line:
            total_params = int(line.split(":")[1].strip().replace(",", ""))
            return total_params

if __name__ == '__main__':
    opt = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TeD(img_size=(opt.image_size[1], opt.image_size[2]),
                patch_size=opt.patch_size,
                in_channels=opt.input_frames,
                out_channels=opt.output_frames,
                window_size=opt.window_size, depths=opt.rstb_depths,
                embed_dim=opt.embed_dim,
                num_heads=opt.num_heads,
                attn_drop_rate=opt.attn_drop_rate)

    model.to(device)
    total_params = get_total_params(model, [(opt.input_frames, opt.image_size[1], opt.image_size[2]), (opt.input_frames, opt.image_size[1], opt.image_size[2])])
    print(f"{total_params/1e6:.2f}")
