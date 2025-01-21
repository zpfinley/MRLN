import torch
from torch import nn
import torch.nn.functional as F

from models import Encoder_2, DecoderLayer, Decoder
from torch.nn import MultiheadAttention


class New_Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(New_Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # print(audio_feature.shape, visual_feature.shape, raw_visual_feature.shape)
        # torch.Size([20, 128]) torch.Size([2, 10, 49, 512]) torch.Size([2, 10, 49, 512])
        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2) # [20, 1, 512]
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch * t_size, h * w, -1) # [20, 49, 512]
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        # print(channel_att_maps.shape) # [2, 10, 1, 512]
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))
        # print(c_att_visual_feat.shape) # torch.Size([2, 10, 49, 512])

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, dim_feedforward=2048):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder = Encoder_2(d_model=d_model, num_layers=2, nhead=4, dim_feedforward=dim_feedforward)
        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature

class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, dim_feedforward):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=dim_feedforward)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):

        query_feature = self.affine_matrix(query_feature)
        output = self.decoder_layer(query_feature, memory_feature)

        return output


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits

        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        class_scores = class_logits

        return logits, class_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class cmran_model(nn.Module):
    def __init__(self):
        super(cmran_model, self).__init__()

        self.video_encoder = InternalTemporalRelationModule(input_dim=256, d_model=256, dim_feedforward=2048)
        self.video_decoder = CrossModalRelationAttModule(input_dim=256, d_model=256, dim_feedforward=2048)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=256, d_model=256, dim_feedforward=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=256, d_model=256, dim_feedforward=2048)

    def forward(self, visual_feature, audio_feature):

        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        return audio_query_output, video_query_output


class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.spatial_channel_att = New_Audio_Guided_Attention()
        self.video_input_dim = 512
        self.video_fc_dim = 512
        self.d_model = 256
        self.v_fc = nn.Linear(512, self.video_fc_dim)
        # self.a_fc = nn.Linear(2048, 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=512, d_model=256, dim_feedforward=2048)
        self.video_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256, dim_feedforward=2048)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=128, d_model=256, dim_feedforward=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=128, d_model=256, dim_feedforward=2048)

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.1)
        self.localize_module = SupvLocalizeModule(self.d_model)

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # audio_feature = self.a_fc(audio_feature)
        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)

        # audio-guided needed
        visual_feature = visual_feature.transpose(1, 0).contiguous()

        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        scores = self.localize_module(video_query_output)

        return scores

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

    model = net_model.float().to('cuda')
    input1 = torch.randn(1, 10, 7, 7, 512).float().to('cuda')
    input2 = torch.randn(1, 10, 128).float().to('cuda')
    flops, params = profile(model, inputs=(input1, input2))
    print("profile: ", flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print("clever: ", flops, params)