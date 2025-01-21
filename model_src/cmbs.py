import torch
from torch import nn
import torch.nn.functional as F
from models import Encoder, DecoderLayer
from torch.nn import MultiheadAttention

class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, d_model, num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

class RNNEncoder2(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder2, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

class New_Audio_Guided_Attention(nn.Module):
    r"""
    This implementation is slightly different from what we described in the paper, which we later found it to be more efficient.

    """

    def __init__(self, beta):
        super(New_Audio_Guided_Attention, self).__init__()

        self.beta = beta
        self.relu = nn.ReLU()
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.hidden_dim = 256
        # channel attention
        self.affine_video_1 = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.affine_audio_1 = nn.Linear(self.audio_input_dim, self.video_input_dim)
        self.affine_bottleneck = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_v_c_att = nn.Linear(self.hidden_dim, self.video_input_dim)
        # spatial attention
        self.affine_video_2 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_audio_2 = nn.Linear(self.audio_input_dim, self.hidden_dim)
        self.affine_v_s_att = nn.Linear(self.hidden_dim, 1)

        self.latent_dim = 4
        self.video_query = nn.Linear(self.video_input_dim, self.video_input_dim // self.latent_dim)
        self.video_key = nn.Linear(self.video_input_dim, self.video_input_dim // self.latent_dim)
        self.video_value = nn.Linear(self.video_input_dim, self.video_input_dim)

        self.affine_video_ave = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_video_3 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.ave_bottleneck = nn.Linear(512, 256)
        self.ave_v_att = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.video_input_dim)

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

        # ============================== Self Attention =======================================
        # visual_feature = c_att_visual_feat
        video_query_feature = self.video_query(visual_feature).reshape(batch * t_size, h * w, -1)  # [B, h*w, C]
        video_key_feature = self.video_key(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value(visual_feature).reshape(batch * t_size, h * w, -1)
        output = torch.matmul(attention, video_value_feature)
        output = self.norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output))
        # c_att_visual_feat = output
        visual_feature = output
        # ============================== Video Spatial Attention ====================================
        video_average = visual_feature.sum(dim=1) / (h * w)
        video_average = video_average.reshape(batch * t_size, v_dim)
        video_average = self.relu(self.affine_video_ave(video_average)).unsqueeze(-2)
        self_video_att_feat = visual_feature.reshape(batch * t_size, -1, v_dim)
        self_video_att_query = self.relu(self.affine_video_3(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query)).transpose(2, 1))
        self_att_feat = torch.bmm(self_spatial_att_maps, visual_feature).squeeze().reshape(batch, t_size, v_dim)

        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch * t_size, h * w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))

        # ==============================Audio Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        # audio_query_2 = self.relu(self.audio_ff(audio_query_1))
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        c_s_att_visual_feat = c_s_att_visual_feat + self.beta * self_att_feat.sigmoid() * c_s_att_visual_feat

        return c_s_att_visual_feat

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, dim_feedforward=2048):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder = Encoder(d_model=d_model, num_layers=2, nhead=4, dim_feedforward=dim_feedforward)
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



class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):

        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


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


class weak_main_model(nn.Module):
    def __init__(self, config):
        super(weak_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config["video_inputdim"]
        self.video_fc_dim = self.config["video_inputdim"]
        self.d_model = self.config["d_model"]
        self.audio_input_dim = self.config["audio_inputdim"]
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=2048)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=1024)
        #self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=256, num_layers=1)
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        #self.localize_module = WeaklyLocalizationModule(self.d_model)
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),

            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.CAS_model = CAS_Module(d_model=self.d_model, num_class=28)
        self.classifier = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        #audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        #visual_rnn_input = visual_feature


        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(audio_key_value_feature)

        av_gate = (audio_gate + video_gate) / 2
        av_gate = av_gate.permute(1, 0, 2)

        video_query_output = (1 - self.alpha)*video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = (1 - self.alpha)*audio_query_output + video_gate * audio_query_output * self.alpha

        video_cas = self.video_cas(video_query_output)
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        #
        # video_cas_gate = (video_cas_gate > 0.01).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.01).float()*audio_cas_gate

        # video_cas = audio_cas_gate.unsqueeze(1) * video_cas
        # audio_cas = video_cas_gate.unsqueeze(1) * audio_cas
        #
        # sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        # topk_scores_video = sorted_scores_video[:, :4, :]
        # score_video = torch.mean(topk_scores_video, dim=1)
        # sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        # topk_scores_audio = sorted_scores_audio[:, :4, :]
        # score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 29]
        #
        # video_cas_gate = score_video.sigmoid()
        # audio_cas_gate = score_audio.sigmoid()
        # video_cas_gate = (video_cas_gate > 0.5).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.5).float()*audio_cas_gate

        #
        # av_score = (score_video + score_audio) / 2


        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        #scores = self.localize_module((video_query_output+audio_query_output)/2)


        fused_content = (video_query_output+audio_query_output)/2
       # fused_content = video_query_output
        fused_content = fused_content.transpose(0, 1)
        #is_event_scores = self.classifier(fused_content)

        cas_score = self.CAS_model(fused_content)
        #cas_score = cas_score + 0.2*video_cas_gate.unsqueeze(1)*cas_score + 0.2*audio_cas_gate.unsqueeze(1)*cas_score
        cas_score = self.gamma*video_cas_gate*cas_score + self.gamma*audio_cas_gate*cas_score
        #cas_score = cas_score*2
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :]
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]       #[32, 29]

        #fused_logits = is_event_scores.sigmoid() * raw_logits
        fused_logits = av_gate * raw_logits
        # fused_scores, _ = fused_logits.sort(descending=True, dim=1)
        # topk_scores = fused_scores[:, :3, :]
        # logits = torch.mean(topk_scores, dim=1)
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        event_scores = event_scores

        return av_gate.squeeze(), raw_logits.squeeze(), event_scores


class cmbs_model(nn.Module):
    def __init__(self):
        super(cmbs_model, self).__init__()

        self.alpha = 0.2

        # feedforward_dim=1024
        self.video_encoder = InternalTemporalRelationModule(input_dim=256, d_model=256, dim_feedforward=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=256, d_model=256, dim_feedforward=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=256, d_model=256, dim_feedforward=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=256, d_model=256, dim_feedforward=1024)

        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=128, d_model=128, num_layers=1)

        self.audio_gated = nn.Sequential(
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )


        self.fa = nn.Linear(256, 128, bias=False)

        self.fv = nn.Linear(256, 128, bias=False)

    def forward(self, visual_feature, audio_feature):


        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio_feature = audio_feature.transpose(1, 0).contiguous()

        audio_rnn_input = self.fa(audio_feature)
        visual_rnn_input = self.fv(visual_feature)

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)

        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]


        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)

        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        # audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha

        return video_query_output, audio_query_output

class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.beta = 0.2
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.video_fc_dim = 512
        self.d_model = 256

        self.v_fc = nn.Linear(512, self.video_fc_dim)
        # self.a_fc = nn.Linear(2048, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model, dim_feedforward=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model, dim_feedforward=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, dim_feedforward=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, dim_feedforward=1024)
        self.audio_visual_rnn_layer = RNNEncoder2(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.video_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = 0.2
        self.gamma = 0.2


    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # audio_feature = self.a_fc(audio_feature)
        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]


        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)


        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha


        video_cas = self.video_cas(video_query_output)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        is_event_scores, event_scores = self.localize_module((video_query_output + audio_query_output)/2)
        event_scores = event_scores + self.gamma*av_score
        #event_scores = event_scores + self.gamma * (event_visual_gate * event_audio_gate) * event_scores


        return is_event_scores, event_scores, audio_visual_gate, av_score

if __name__ == '__main__':

    net_model = supv_main_model()
    def count_parameters(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1000000
    num_params = count_parameters(net_model)
    print(num_params)
    print("Total Parameter: \t%2.1fM" % num_params)
