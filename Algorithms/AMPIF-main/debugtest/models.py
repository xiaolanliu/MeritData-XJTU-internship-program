import torch
import math
import torch.nn as nn

from Encoder import Season_Encoder, Trend_Encoder
from ts_clustering import get_clustering
from utils import divide_data, trend_sim, log_Normal_standard, log_Normal_diag, fft_sim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, low_f_num, high_f_num, is_low_noise=False):
        super(series_decomp, self).__init__()
        self.low_f_num = low_f_num
        self.high_f_num = high_f_num
        self.is_low_noise = is_low_noise

    def forward(self, x, is_label=False):
        x = x.permute(0, 2, 1)
        B, N, L = x.shape
        t_fft = torch.fft.rfft(x, dim=-1)
        mask_low = torch.ones_like(t_fft).to(device)
        mask_low[:, :, self.low_f_num:] = 0
        mask_high = torch.ones_like(t_fft).to(device)
        mask_high[:, :, :self.low_f_num] = 0
        t_fft_low = mask_low * t_fft
        t_fft_high = mask_high * t_fft
        if not is_label:
            t_q = torch.abs(t_fft_high)
            idx = torch.topk(t_q, self.high_f_num)[1]
            flag_h = torch.zeros_like(t_fft).to(device)
            flag_h = flag_h.reshape(B * N, -1)
            idx = idx.reshape(B * N, -1)
            for i in range(B * N):
                flag_h[i][idx[i]] = 1
            flag_h = flag_h.reshape(B, N, -1)
            t_fft_high = t_fft_high * flag_h

        trend = torch.fft.irfft(t_fft_low, dim=-1).permute(0, 2, 1)
        seasonal = torch.fft.irfft(t_fft_high, dim=-1).permute(0, 2, 1)

        return trend, seasonal


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(
            d_model=d_model, embed_type='TimeF', freq='h')
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class CriticFunc(nn.Module):
    def __init__(self, args):
        super(CriticFunc, self).__init__()

        self.fc_y = nn.Linear(args.seq_len + args.pred_len, args.emb_dim)
        self.critic = nn.Sequential(
            nn.Linear(2 * args.emb_dim, args.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(args.emb_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, y, fea):
        cat = torch.cat([self.fc_y(y), fea], dim=-1)
        return self.critic(cat)



class GRL(nn.Module):
    def __init__(self, args):
        super(GRL, self).__init__()
        self.batch_size = args.batch_size
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.emb_dim = args.emb_dim
        self.num_samples = args.num_samples

        self.trend_conv_mean = torch.nn.Conv1d(1, args.conv_dim, args.kernel_size, stride=1)
        self.trend_conv_var = torch.nn.Conv1d(1, args.conv_dim, args.kernel_size, stride=1)
        self.t_fc_mean = nn.Linear(args.fc_dim, args.emb_dim)
        self.t_fc_var = nn.Linear(args.fc_dim, args.emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(args.conv_dim)
        self.season_conv_mean = torch.nn.Conv1d(1, args.conv_dim, args.kernel_size, stride=1)
        self.season_conv_var = torch.nn.Conv1d(1, args.conv_dim, args.kernel_size, stride=1)
        self.s_fc_mean = nn.Linear(args.fc_dim, args.emb_dim)
        self.s_fc_var = nn.Linear(args.fc_dim, args.emb_dim)

        self.s_fc_rec = nn.Linear(args.emb_dim, args.pred_len + args.seq_len)
        self.t_fc_rec = nn.Linear(args.emb_dim, args.pred_len + args.seq_len)

        self.critic_xz = CriticFunc(args)

    def compute_KL(self, z_q, z_q_mean, z_q_var):
        log_p_z = log_Normal_standard(z_q, dim=1)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_var, dim=1)
        KL = -(log_p_z - log_q_z)

        return KL.mean()

    def compute_MLBO(self, x, z_q, method="our"):
        idx = torch.randperm(z_q.shape[0])
        z_q_shuffle = z_q[idx].view(z_q.size())
        if method == "MINE":
            mlbo = self.critic_xz(x, z_q).mean() - torch.log(
                torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean(dim=-1)).mean()
        else:
            point = 1 / torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean()
            point = point.detach()

            if len(x.shape) == 3:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle))  # + 1 + torch.log(point)
            else:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(self.critic_xz(x, z_q_shuffle))

        return mlbo.mean()


    def sample_normal(self, mu, log_variance):
        shape = [self.num_samples] + list(mu.size())
        eps = torch.rand(shape).to(device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))


    def Encoder(self, x, conv_mean, conv_var, fc_mean, fc_var):
        mean = conv_mean(x.transpose(-2, -1))
        mean = self.bn1(mean)
        mean = mean.view(mean.shape[0], -1)
        mean = fc_mean(mean)

        var = conv_var(x.transpose(-2, -1))
        var = self.bn1(var)
        var = var.view(var.shape[0], -1)
        var = torch.log(1 + torch.exp(fc_var(var)))

        samples = self.sample_normal(mean, var)
        fea = samples.mean(0)

        return fea, mean, var


    def forward(self, trend_x, season_x, trend_y, season_y):
        trend_y, season_y = trend_y.squeeze(-1), season_y.squeeze(-1)

        t_fea, t_mean, t_var = self.Encoder(trend_x, self.trend_conv_mean, self.trend_conv_var, self.t_fc_mean, self.t_fc_var)
        s_fea, s_mean, s_var = self.Encoder(season_x, self.season_conv_mean, self.season_conv_var, self.s_fc_mean, self.s_fc_var)

        tx_rec = self.t_fc_rec(t_fea)
        sx_rec = self.s_fc_rec(s_fea)

        elbo = - ((tx_rec - trend_y) ** 2).mean() - ((sx_rec - season_y) ** 2).mean() + fft_sim(sx_rec, season_y) \
               + trend_sim(tx_rec, trend_y) - self.compute_KL(t_fea, t_mean, t_var) - self.compute_KL(s_fea, s_mean, s_var)

        mlbo = self.compute_MLBO(trend_y, t_fea) + self.compute_MLBO(season_y, s_fea)

        return t_fea, s_fea, elbo + mlbo


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.batch_size = args.batch_size
        self.support_rate = args.support_rate
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.num_cluster = args.num_cluster
        self.num_samples = args.num_samples
        self.emb_dim = args.emb_dim
        self.hid_dim = args.hid_dim
        self.data_name = args.data_name

        # Decomp
        self.decomp = series_decomp(args.low_f_num, args.high_f_num, args.is_low_noise)
        # Encoder
        self.Season_Encoder = Season_Encoder(args)
        # self.Trend_Encoder = Trend_Encoder(args)
        self.Trend_Encoder = Season_Encoder(args)
        if args.data_name == 'Ashare':
            self.lstm = nn.LSTM(args.num_var - 1, args.emb_dim, num_layers=2, dropout=args.dropout)
            self.fc_factor_mean = nn.Linear(2 * args.hid_dim, args.emb_dim)
            self.fc_factor_var = nn.Linear(2 * args.hid_dim, args.emb_dim)
            self.factor_label_fc = nn.Linear(1, args.hid_dim)
            self.fc_output = nn.Linear(2 * (args.emb_dim + args.hid_dim), 1)
        else:
            self.fc_output_1 = nn.Linear(args.emb_dim + args.hid_dim, 1)
            self.fc_output_2 = nn.Linear(args.seq_len + args.pred_len, args.pred_len)


        self.s_label_fc = nn.Linear(1, args.hid_dim)
        self.t_label_fc = nn.Linear(1, args.hid_dim)


        self.season_embedding = DataEmbedding(1, args.emb_dim, args.dropout)
        self.trend_embedding = DataEmbedding(1, args.emb_dim, args.dropout)
        self.fc_hid_season = nn.Linear(args.emb_dim, args.hid_dim)
        self.fc_hid_trend = nn.Linear(args.emb_dim, args.hid_dim)
        self.fc_hid_factor = nn.Linear(args.emb_dim, args.hid_dim)
        self.fc_season_mean = nn.Linear(2 * args.hid_dim, args.emb_dim)
        self.fc_trend_mean = nn.Linear(2 * args.hid_dim, args.emb_dim)

        self.fc_season_var = nn.Linear(2 * args.hid_dim, args.emb_dim)
        self.fc_trend_var = nn.Linear(2 * args.hid_dim, args.emb_dim)

        self.GRL = GRL(args)




        self.criterion = nn.CrossEntropyLoss()



    def sample_normal(self, mu, log_variance):
        shape = [self.num_samples] + list(mu.size())
        eps = torch.rand(shape).to(device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))

    def get_feas(self, test_feas, adaptation_params):
        test_size = test_feas.shape[0]
        test_feas = test_feas.unsqueeze(0)
        test_feas = test_feas.repeat(self.num_samples, 1, 1, 1)
        adaptation_params = adaptation_params.repeat(1, test_size, 1, 1)
        test_output = torch.cat([test_feas, adaptation_params], dim=-1)
        test_output = test_output.mean(0)
        test_output = test_output.squeeze(-1)
        return test_output

    def mate_inference(self, fea, y, cluster, fc_hid, fc_mean, fc_var, fc_label, num_cluster):
        support_fea, support_label, query_fea, query_label, support_cluster, query_cluster\
            = divide_data(fc_hid(fea), y, cluster, self.support_rate)
        output_fea = torch.zeros([query_fea.shape[0], self.seq_len + self.pred_len, self.emb_dim + self.hid_dim],
                                 device=device)

        label_fea = fc_label(support_label.unsqueeze(-1))

        for c in range(num_cluster):
            c_support_mask = torch.eq(support_cluster, c)
            c_support_fea = support_fea[c_support_mask]
            c_support_label = label_fea[c_support_mask]
            c_support_fea = torch.cat([c_support_fea, c_support_label], dim=-1)
            c_s_fea_mean = c_support_fea.mean(0).unsqueeze(0)
            c_mu = fc_mean(c_s_fea_mean)
            c_log_variance = fc_var(c_s_fea_mean)
            c_psi_samples = self.sample_normal(c_mu, c_log_variance)

            c_query_mask = torch.eq(query_cluster, c)
            c_query_fea = query_fea[c_query_mask]
            c_query = self.get_feas(c_query_fea, c_psi_samples)
            output_fea[c_query_mask] = c_query

        return output_fea, query_label




    def forward(self, x, factor, y):
        # decomp init
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=device)
        if self.data_name == 'Ashare':
            zeros_factor = torch.zeros([factor.shape[0], self.pred_len, factor.shape[2]], device=device)
        trend_x, seasonal_x = self.decomp(x, is_label=False)
        trend_y, seasonal_y = self.decomp(y.unsqueeze(-1), is_label=True)

        t_fea, s_fea, loss_GRL = self.GRL(trend_x, seasonal_x, trend_y, seasonal_y)

        # cluster
        with torch.no_grad():
            season_cluster = get_clustering(s_fea.unsqueeze(-1).detach().cpu().numpy(), self.num_cluster, self.support_rate, mode='season')
            trend_cluster = get_clustering(t_fea.unsqueeze(-1).detach().cpu().numpy(), self.num_cluster, self.support_rate, mode='trend')
            if self.data_name == 'Ashare':
                factor_cluster = get_clustering(factor.detach().cpu().numpy(), self.num_cluster, self.support_rate, mode='factor')


        if self.data_name == 'Ashare':
            factor = torch.cat([factor, zeros_factor], dim=1)
            # factor Encoder
            factor_fea, _ = self.lstm(factor)
            factor_query_fea, query_y = \
                self.mate_inference(factor_fea, y, factor_cluster, self.fc_hid_factor, self.fc_factor_mean,
                                    self.fc_factor_var, self.factor_label_fc, self.num_cluster)



        trend_x = torch.cat([trend_x, mean], dim=1)
        seasonal_x = torch.cat([seasonal_x, zeros], dim=1)
        seasonal_fea = self.season_embedding(seasonal_x, factor)
        seasonal_fea = self.Season_Encoder(seasonal_fea)

        # trend Encoder
        trend_fea = self.trend_embedding(trend_x, factor)
        trend_fea = self.Trend_Encoder(trend_fea)

        season_query_fea, season_query_y = \
            self.mate_inference(seasonal_fea, seasonal_y.squeeze(-1), season_cluster, self.fc_hid_season, self.fc_season_mean,
                                self.fc_season_var, self.s_label_fc, self.num_cluster)
        trend_query_fea, trend_query_y = \
            self.mate_inference(trend_fea, trend_y.squeeze(-1), trend_cluster, self.fc_hid_trend, self.fc_trend_mean,
                                self.fc_trend_var, self.t_label_fc, self.num_cluster)
        query_y = season_query_y + trend_query_y

        if self.data_name == 'Ashare':
            query_fea = torch.cat([season_query_fea + trend_query_fea, factor_query_fea], dim=-1)
        else:
            query_fea = season_query_fea + trend_query_fea

        output = self.fc_output_1(query_fea).squeeze(-1)
        output, label = self.fc_output_2(output), query_y[:, self.seq_len:]
        return output, label, loss_GRL

