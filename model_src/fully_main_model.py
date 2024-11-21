import torch
from torch import nn
from .models import CMRE_Encoder
from .ac_gat import AC_GAT

class Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()

        self.affine_video_2 = nn.Linear(256, 128)
        self.affine_audio_2 = nn.Linear(256, 128)
        self.affine_v_s_att = nn.Linear(256, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feature, audio):

        audio = audio.transpose(1, 0)
        batch, t_size, o, v_dim = visual_feature.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)

        # ============================== Spatial Attention =====================================
        c_att_visual_feat = visual_feature.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_query_2 = audio_query_2.repeat(1, o, 1)
        audio_video_query_2 = torch.cat([c_att_visual_query, audio_query_2], dim=-1)
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)
        self.event_classifier = nn.Linear(d_model, 28)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fused_content):

        mean_fused_content = fused_content.transpose(1, 0).mean(1)

        logits = self.classifier(fused_content)

        class_logits = self.event_classifier(mean_fused_content)

        return logits, class_logits



class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.spatial_att = Audio_Guided_Attention()

        self.video_input_dim = 2048
        self.video_fc_dim = 512
        self.d_model = 256

        self.audio_fc = nn.Sequential(nn.Linear(2048, 256),
                                      nn.LeakyReLU(),)

        self.visual_fc = nn.Sequential(nn.Linear(2048, 256),
                                      nn.LeakyReLU(),)

        self.ac_gat = AC_GAT(512, 256)

        self.cmre = CMRE_Encoder(d_model=256, num_layers=2, nhead=4)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.v_fc = nn.Linear(256, 256, bias=False)
        self.a_fc = nn.Linear(256, 256, bias=False)
        self.layer_norm = nn.LayerNorm(256, eps=1e-6)

        self.localize_module = SupvLocalizeModule(self.d_model)


    def forward(self, visual_feature, audio):

        visual_feature = self.visual_fc(visual_feature)
        audio = self.audio_fc(audio)

        bz, t, o, dim = visual_feature.size()
        visual_feature = visual_feature.view(bz*t, o, dim)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = self.ac_gat(visual_feature, audio)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = visual_feature.view(bz, t, o, dim)

        visual_feature = self.spatial_att(visual_feature, audio)

        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()

        visual_feature, audio = self.cmre(visual_feature, audio)

        visual = self.dropout(self.relu(self.v_fc(visual_feature)))
        audio = self.dropout(self.relu(self.a_fc(audio)))
        visual = self.layer_norm(visual)
        audio = self.layer_norm(audio)
        video_query_output = torch.mul(visual + audio, 0.5)

        scores = self.localize_module(video_query_output)

        return scores[0], scores[1], video_query_output


if __name__ == '__main__':

    net_model = supv_main_model()

    model = net_model.double().to('cuda')
    input1 = torch.randn(64, 10, 36, 2048).double().to('cuda')
    input2 = torch.randn(64, 10, 2048).double().to('cuda')
    o1, o2, o3 = model(input1, input2)

    print(o1.shape, o2.shape, o3.shape)

