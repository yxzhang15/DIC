import os
#import matplotlib.pyplot as plt
import json
import torch


class LossWriter():
    def __init__(self, save_dir):
        """
        初始化，loss_writer = LossWriter("xxx/")
        :param save_dir: 损失值保存路径
        """
        self.save_dir = save_dir

    def add(self, loss_name: str, loss, i: int):
        """
        将迭代次数和loss值写入txt
        :param loss_name: 损失函数名
        :param loss: utils.item()
        :param i: 迭代次数
        :return: None
        """
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


def write_metrics(path, epoch, ssim, psnr):
    with open(path, mode='a') as f:
        info = str(epoch) + " " + str(ssim) + " " + str(psnr) + "\n"
        f.write(info)
        f.close()
def write_loss(path, epoch, loss):
    with open(path, mode='a') as f:
        info = str(epoch) + " " + str(loss) + "\n"
        f.write(info)
        f.close()

def write_metrics_for_reg(path, epoch, train, r_rmse, r_mae, t_rmse, t_mae, rre, rte):
    with open(path, mode='a') as f:
        info = str(epoch) + " " + str(r_rmse) + " " + str(train) + " " + str(r_mae)+ " "+ str(t_rmse) + " " + str(t_mae) + " " + str(rre) + " " + str(rte) + " " + "\n"
        f.write(info)
        f.close()








def save_config_as_json(save_path, config):
    with open(save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def save_best_model(cur_psnr, cur_ssim, best_psnr, best_ssim, save_dir, network, model_name, dataset_name):
    if cur_psnr > best_psnr:
        best_psnr = cur_psnr
        torch.save(network.state_dict(), os.path.join(save_dir, "models", "best_psnr_" + model_name + "_" + dataset_name + ".pth"))
        # print("save best psnr")
    if cur_ssim > best_ssim:
        best_ssim = cur_ssim
        torch.save(network.state_dict(),  os.path.join(save_dir, "models", "best_ssim_" + model_name + "_" + dataset_name + ".pth"))
        # print("save best ssim")

    torch.save(network.state_dict(), os.path.join(save_dir, "models", "last_" + model_name + "_" + dataset_name + ".pth"))

    return best_ssim, best_psnr


def save_cur_model(save_dir, network, model_name, dataset_name, epochs):

    torch.save(network.state_dict(), os.path.join(save_dir, "models", str(epochs) + "_" + model_name + "_" + dataset_name + ".pth"))

    return None


if __name__ == "__main__":
    plot_all_losses(losses_path="../results/simvp_bs1_seq4/utils/")
