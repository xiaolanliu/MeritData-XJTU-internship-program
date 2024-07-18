import torch
import argparse
import numpy as np
import warnings
import os
from utils import load_ETT_dataset
from models import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

class model_init:
    def __init__(self, args):
        # 初始化相关信息
        self.batch_size = args.batch_size
        self.base_lr = args.base_lr
        self.epsilon = args.epsilon
        self.support_rate = args.support_rate
        self.epoch = args.epoch
        self.show_len = args.show_len
        self.epoch_begin = args.epoch_begin
        self.num_cluster = args.num_cluster
        self.opt = args.optimizer
        self.max_grad_norm = args.max_grad_norm
        self.output_name = args.output_name
        self.base_name = args.base_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.best_result = {'mae': 999, 'mse': 999}
        self.output_file = os.path.join(f"output/{args.data_name}_result/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/", self.output_name + '.txt')

        if not os.path.exists(f"output/{args.data_name}_save_model/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/"):
            os.makedirs(f"output/{args.data_name}_save_model/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/")
        if not os.path.exists(f"output/{args.data_name}_result/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/"):
            os.makedirs(f"output/{args.data_name}_result/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/")

        # 初始随机种子
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.data = load_ETT_dataset(args)
        args.num_var = self.data['train_loader'].num_var
        # 加载模型
        model = Model(args)

        if self.epoch_begin > 0:
            model.load_state_dict(
                torch.load(f'output/{args.data_name}_save_model/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/{self.output_name}.pth'
                           , map_location='cpu'))
        self.model = model.cuda() if torch.cuda.is_available() else model


    def eval(self, mode='test'):
        loader = self.data[f'{mode}_loader']
        num_batch = loader.num_batch
        self.model.support_rate = 0.5
        self.model.num_cluster = 2 * self.num_cluster
        self.model.eval()
        l_MSE = torch.nn.MSELoss()
        l_MAE = torch.nn.L1Loss()
        sum_mae, sum_mse = 0, 0
        with torch.no_grad():
            iterator = loader.get_iterator()
            for batch_idx, (x, factor, y) in enumerate(iterator):
                output, label, _ = self.model(x, factor, y)
                mae, mse = l_MAE(output, label), l_MSE(output, label)
                sum_mae += mae
                sum_mse += mse
        return sum_mae / num_batch, sum_mse / num_batch



    def train(self):
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, eps=self.epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                        lr=self.base_lr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, eps=self.epsilon)

        l_MSE = torch.nn.MSELoss()
        l_MAE = torch.nn.L1Loss()
        for epoch_num in range(self.epoch_begin, self.epoch):
            print("Num of epoch:", epoch_num)
            optimizer.zero_grad()
            self.model = self.model.train()
            train_iterator = self.data['train_loader'].get_iterator()
            num_batches = self.data['train_loader'].num_batch

            for batch_idx, (x, factor, y) in enumerate(train_iterator):
                self.model.support_rate = self.support_rate
                self.model.num_cluster = self.num_cluster
                output, label, loss_GRL = self.model(x, factor, y)
                loss = l_MAE(output, label) - loss_GRL
                mae, mse = l_MAE(output, label), l_MSE(output, label)
                if torch.isinf(loss_GRL):
                    continue
                loss.backward()
                optimizer.step()
                if batch_idx % self.show_len == 0:
                    self.print_result(mae, mse, epoch_num, batch_idx, num_batches, loss_GRL, mode='train')
            val_mae, val_mse = self.eval(mode='val')
            if val_mae + val_mse < self.best_result['mae'] + self.best_result['mse']:
                self.best_result['mae'] = val_mae
                self.best_result['mse'] = val_mse
                self._save_model()
            self.print_result(val_mae, val_mse, epoch_num, 0, num_batches, 0, mode='val')

    def print_result(self, mae, mse, epoch_num, batch_idx, num_batches, loss_GRL, mode='train'):
        if mode == 'train':
            output_data = 'Train_Epoch: {} .. batch: {}/{} .. : loss_GRL: {}.. : loss_mae: {}.. : loss_mse: {}..'.format(
                epoch_num, batch_idx, num_batches, loss_GRL, mae, mse)
            print(output_data)
            with open(self.output_file, "a+") as f:
                f.write(output_data + '\n')
                f.close()
        else:
            output1 = '+++++++{}_result_Train_Epoch: {} .. batch: {}/{}+++++++'.format(mode, epoch_num, batch_idx, num_batches)
            output2 = 'mae: {}.. mse: {}..'.format(mae, mse)
            output3 = 'best_mae: {}.. best_mse: {}..'.format(self.best_result['mae'], self.best_result['mse'])
            output4 = '================================================================'
            print(output1)
            print(output2)
            print(output3)
            print(output4)
            with open(self.output_file, "a+") as f:
                f.write(output1 + '\n')
                f.write(output2 + '\n')
                f.write(output3 + '\n')
                f.write(output4 + '\n')
                f.close()


    def _save_model(self):

        output_path = os.path.join(f'output/{args.data_name}_save_model/{self.base_name}/sl_{self.seq_len}_pl_{self.pred_len}/', self.output_name)
        torch.save(self.model.state_dict(), f"{output_path}.pth")




def main(args):
    exp = model_init(args=args)
    if args.only_test:
        exp.model.load_state_dict(
            torch.load(
                f'output/{args.data_name}_save_model/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/{args.output_name}.pth'
                , map_location='cpu'))
        exp.model = exp.model.cuda() if torch.cuda.is_available() else exp.model
    else:
        exp.train()
        exp.model.load_state_dict(
            torch.load(
                f'output/{args.data_name}_save_model/{args.base_name}/sl_{args.seq_len}_pl_{args.pred_len}/{args.output_name}.pth'
                , map_location='cpu'))
        exp.model = exp.model.cuda() if torch.cuda.is_available() else exp.model

    test_mae, test_mse = exp.eval(mode='test')
    exp.print_result(test_mae, test_mse, args.epoch, 0, 0, 0, mode='test')



def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='ETTh2', type=str,
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'], help='dataset type')
    parser.add_argument('--base_name', default='AMPIF', type=str, help='')
    parser.add_argument('--batch_size', default=64, type=float, help='batch size of train input data')
    parser.add_argument('--base_lr', default=0.000002, type=float, help='optimizer learning rate')
    parser.add_argument('--epoch', default=100, type=float, help='Number of training epoch')
    parser.add_argument('--show_len', default=300, type=float,
                        help='Output the results every how many batches during the training process')
    parser.add_argument('--epoch_begin', default=0, type=float, help='Starting epoch')

    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--seed', default=3407, type=float, help='Random seed setting')
    parser.add_argument('--max_grad_norm', default=5, type=float, help='Gradient cropping')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--epsilon', default=1.0e-4, type=float, help='optimizer epsilon')
    parser.add_argument('--emb_dim', default=128, type=float, help='Embedding Dimension')
    parser.add_argument('--hid_dim', default=128, type=float, help='Hidden Dimension')
    parser.add_argument('--conv_dim', default=8, type=float, help='Convolutional layer dimension')
    parser.add_argument('--fc_dim', default=696, type=float, help='Global mapping linear layer dimension')
    parser.add_argument('--kernel_size', default=10, type=float, help='kernel_size')
    parser.add_argument('--seq_len', default=96, type=float, help='input sequence length')
    parser.add_argument('--pred_len', default=24, type=float, help='prediction sequence length')

    parser.add_argument('--support_rate', default=0.5, type=float, help='Support rate')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--factor', type=int, default=1, help='')
    parser.add_argument('--num_cluster', default=20, type=float, help='')
    parser.add_argument('--num_samples', default=10, type=float, help='Number of task parameter samples')
    parser.add_argument('--low_f_num', default=3, type=float, help='')
    parser.add_argument('--high_f_num', default=8, type=float, help='')

    parser.add_argument('--is_low_noise', default=True, type=bool, help='')
    parser.add_argument('--shuffle', default=True, type=bool, help='')
    parser.add_argument('--only_test', default=False, type=bool, help='status')

    args = parser.parse_args()
    args.output_name = "dm_%s_bs_%d_lr_%f_dim_%d_nc_%d_ns_%d" % (args.data_name, args.batch_size, args.base_lr,
                                                                      args.emb_dim, args.num_cluster, args.num_samples)

    return args


if __name__ == '__main__':
    args = Args()
    main(args)



