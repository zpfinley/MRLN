import torch
from torch import nn
import torch.nn.functional as F
from .models import Mix_Encoder, EncoderLayer, Encoder
from .Rs_GCN import AC_GAT

from .cmran import cmran_model
# from .cmbs import cmbs_model
# from .dmin import dmin_model

class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, hidden_dim=128, seg_num=10):
        super(LSTM_A_V, self).__init__()

        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_fea):
        bs, seg_num, a_dim = a_fea.shape
        # hidden_a = (torch.zeros(2, bs, a_dim), torch.zeros(2, bs, a_dim))
        # hidden_v = (torch.zeros(2, bs, a_dim), torch.zeros(2, bs, a_dim))
        hidden_a = (torch.zeros(2, bs, a_dim).double().cuda(), torch.zeros(2, bs, a_dim).double().cuda())
        return hidden_a

    def forward(self, a_fea):
        # a_fea, v_fea: [bs, 10, 128]
        hidden_a = self.init_hidden(a_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_audio.flatten_parameters()

        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a)

        return lstm_audio

class Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()

        # spatial attention
        self.affine_video_2 = nn.Linear(256, 128)
        self.affine_audio_2 = nn.Linear(256, 128)
        self.affine_v_s_att = nn.Linear(256, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feature, audio):
        '''
        :param visual_feature: [batch, 10, 36, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, o, v_dim = visual_feature.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = visual_feature.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat)) # [640, 36, 128]
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2) # [640, 1, 128]
        audio_query_2 = audio_query_2.repeat(1, o, 1)
        audio_video_query_2 = torch.cat([c_att_visual_query, audio_query_2], dim=-1)
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat


# attmodel = Audio_Guided_Attention()
# loc_vis = torch.rand(64, 10, 36, 256)
# audio = torch.rand(64, 10, 256)
# out = attmodel(loc_vis, audio)
# print(out.shape)


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        # self.att_pool = AttFlat(dim=256)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fused_content):

        # fused_content = F.dropout(fused_content, 0.1)
        # fused_content [10, bz, dim]
        # max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        max_fused_content = fused_content.transpose(1, 0).mean(1)
        # max_fused_content = self.att_pool(fused_content.transpose(1, 0))
        # print(max_fused_content.shape)

        logits = self.classifier(fused_content)

        class_logits = self.event_classifier(max_fused_content)

        return logits, class_logits

# class Adapter_Gate(nn.Module):
#     def __init__(self):
#         super(Adapter_Gate, self).__init__()
#
#         # self.sim_w = nn.Linear(256, 256)
#         self.fc = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
#
#     def forward(self, fusion):
#
#         t, bz, dim = fusion.shape
#
#         zero_pad = torch.zeros(1, bz, dim).cuda()
#         fusion_pad = torch.cat([fusion[1:t, :, :], zero_pad], dim=0)
#         sub_fusion = torch.sub(fusion_pad, fusion) # 后一秒的特征减掉前一秒的特征
#         sub_fusion = sub_fusion[0:(t-1), :, :] # 舍弃0填充
#         # sim = torch.pow(sub_fusion, 2)
#         # sim = F.normalize(self.sim_w(sim), dim=-1)
#         sim = F.normalize(sub_fusion, dim=-1)
#         g = self.fc(sim).squeeze()
#         g = g.transpose(1, 0) # [64, 9]
#
#         return g

class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.spatial_att = Audio_Guided_Attention()

        self.video_input_dim = 2048
        self.video_fc_dim = 512
        self.d_model = 256

        self.audio_fc = nn.Sequential(nn.Linear(2048, 256),
                                      nn.LeakyReLU(),
                                      # nn.Linear(256, 256),
                                      # nn.Dropout(0.1)
                                      )

        self.visual_fc = nn.Sequential(nn.Linear(2048, 256),
                                      nn.LeakyReLU(),
                                      # nn.Linear(256, 256),
                                      # nn.Dropout(0.1)
                                       )

        self.gcn1 = AC_GAT(512, 256)
        # self.gcn2 = AC_GAT(512, 256)
        # self.gcn3 = AC_GAT(512, 256)

        self.mixencoder = Mix_Encoder(d_model=256, num_layers=2, nhead=4)
        # params = sum(p.numel() for p in self.mixencoder.parameters() if p.requires_grad)
        # print("Total Parameter: \t%2.1fM" % params)

        # self.mixencoder = cmran_model()
        # self.mixencoder = cmbs_model()
        # self.mixencoder = dmin_model()
        # self.mixencoderA = Mix_Encoder(d_model=256, num_layers=2, nhead=4)
        # self.mixencoderV = Mix_Encoder(d_model=256, num_layers=2, nhead=4)

        # self.mixencoder = LSTM_A_V(a_dim=256, hidden_dim=256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.v_fc = nn.Linear(256, 256, bias=False)
        self.a_fc = nn.Linear(256, 256, bias=False)
        self.layer_norm = nn.LayerNorm(256, eps=1e-6)

        self.localize_module = SupvLocalizeModule(self.d_model)

        # self.adapter_g = Adapter_Gate()


    def forward(self, visual_feature, audio):

        visual_feature = self.visual_fc(visual_feature)
        # visual_feature = torch.cat([visual_feature, box_feat], dim=-1)
        audio = self.audio_fc(audio)

        bz, t, o, dim = visual_feature.size()
        visual_feature = visual_feature.view(bz*t, o, dim)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = self.gcn1(visual_feature, audio)
        # visual_feature = self.gcn2(visual_feature, audio)
        # visual_feature = self.gcn3(visual_feature, audio)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = visual_feature.view(bz, t, o, dim)

        visual_feature = self.spatial_att(visual_feature, audio)
        # visual_feature = visual_feature.mean(2).squeeze()

        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()

        visual_feature, audio = self.mixencoder(visual_feature, audio)
        # video_query_output = self.mixencoder(visual_feature, audio)

        # visual_feature = self.mixencoder(visual_feature)
        # audio = self.mixencoder(audio)

        # visual_feature = self.mixencoderV(visual_feature)
        # audio = self.mixencoderA(audio)

        visual = self.dropout(self.relu(self.v_fc(visual_feature)))
        audio = self.dropout(self.relu(self.a_fc(audio)))
        visual = self.layer_norm(visual)
        audio = self.layer_norm(audio)
        video_query_output = torch.mul(visual + audio, 0.5)

        # adapter_g = self.adapter_g(video_query_output)

        # video_query_output = video_query_output.transpose(1, 0)

        scores = self.localize_module(video_query_output)

        return scores[0], scores[1], video_query_output


if __name__ == '__main__':

    net_model = supv_main_model()
    def count_parameters(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1000000
    num_params = count_parameters(net_model)
    print(num_params)
    print("Total Parameter: \t%2.1fM" % num_params)

    from thop import profile
    from thop import clever_format

    model = net_model.double().to('cuda')
    input1 = torch.randn(64, 10, 36, 2048).double().to('cuda')
    input2 = torch.randn(64, 10, 2048).double().to('cuda')
    flops, params = profile(model, inputs=(input1, input2))
    print("profile: ", flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print("clever: ", flops, params)

    # model = net_model.double().to('cuda')
    # input1 = torch.randn(64, 10, 36, 2048).double().to('cuda')
    # input2 = torch.randn(64, 10, 2048).double().to('cuda')
    # out = model(input1, input2)
