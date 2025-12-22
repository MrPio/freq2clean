import torch
import torch.nn as nn
import torch.nn.functional as F

class Projection_Loss(nn.Module):
    def __init__(self):
        super(Projection_Loss, self).__init__()

    def forward(self, predicted_tensor, target_tensor):
        _, t, w, h = predicted_tensor.size()
        crop_size = 20
        total_loss = 0
        for i in range(10):

            top = torch.randint(0, w - crop_size + 1, (1,))
            left = torch.randint(0, h - crop_size + 1, (1,))

            cropped_predicted_tensor = predicted_tensor[:, :, top:(top + crop_size), left:(left + crop_size)]
            cropped_target_tensor = target_tensor[:, :, top:(top + crop_size), left:(left + crop_size)]


            mean_predicted_w = cropped_predicted_tensor.mean(dim = (2, 3))
            mean_target_w = cropped_target_tensor.mean(dim = (2, 3))
            loss_w = F.l1_loss(mean_predicted_w, mean_target_w)
            total_loss += loss_w

        return total_loss



