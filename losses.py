import torch
import torch.nn.functional as F

def inter_segment_sim_loss(fusion, segment_flag_pos):

    fusion = F.normalize(fusion, dim=-1)

    loss_b = 0
    n_exist_loss = 0

    for i in range(0, fusion.size(0)):
        loss_i = 0
        segment_flag_pos_i = segment_flag_pos[i, :]
        num_i = 0
        for j in range(0, 9):
            if segment_flag_pos_i[j] == 0:
                pass
            elif (segment_flag_pos_i[j] == 1) and (segment_flag_pos_i[j + 1] == 1):
                num_i = num_i + 1
                sim_j = torch.mm(fusion[i, j, :].unsqueeze(0), fusion[i, j + 1, :].unsqueeze(1)).squeeze()
                sim_j = - torch.log(sim_j + 1e-12)
                loss_i = loss_i + sim_j
            else:
                pass

        if num_i == 0:
            pass
        else:
            n_exist_loss = n_exist_loss + 1
            loss_b = loss_b + loss_i / num_i

    loss = loss_b / n_exist_loss

    return loss






