# normal
import matplotlib.pyplot as plt
import sys
import argparse

# pytouch
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

# torch_geometric
from torch_geometric.loader import DataLoader

# original module
import original_module as mod
import original_model

# argparse.ArgumentParserクラスをインスタンス化して、説明等を引数として渡す
parser = argparse.ArgumentParser(
    prog="estimate_mom",
    usage="python3 estimate_mom.py <input_csv_file_path>", # プログラムの利用方法
    description="training GNN-model script.", # ヘルプの前に表示
    epilog="end", # ヘルプの後に表示
    add_help=True, # -h/–-helpオプションの追加
)
parser.add_argument("input_csv_file_path", type=str, help="Input csv file path")
args = parser.parse_args()

# cpu, gpuの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input, output sizeの設定
input_dim  = 3 # num of edge feature (energy deposit)
output_dim = 1  # num of output size  (momentum)

# エポック数
num_epochs = 50

# データ読み込み
gen7208 = mod.DataManager(args.input_csv_file_path)
data = gen7208.load_data(isDebug=False)

# 学習データと検証データに分割
train_data, valid_data = mod.shuffle_list_data(data)

# dataloader の作成（ミニバッチ処理のため）
batch_size  = 64
num_workers = 8
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataloader,
    'val'  : valid_dataloader
}

# modelの作成
model = original_model.GNNmodel(input_dim, output_dim).to(device)

# 損失関数などの定義
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience=3, min_lr=0.0001)

# modelなどの読み込み
checkpoint = torch.load("model/20240312-012520/checkpoint.bin") 
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])

# 学習実行
dict_data = mod.learning( device, model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, scheduler )

# discordで通知( https://pypi.org/project/discordwebhook/ )
try:
    from discordwebhook import Discord
    import discord_url  # 自作関数でurlを単に格納しているだけ
    discord = discord_url.get_discord()
    discord.post(content="({}) finish estimate_mom.py".format(sys.platform))
except:
    pass

# 損失関数表示
plt.plot(dict_data["train"]["loss"])
plt.plot(dict_data["valid"]["loss"])
plt.show()
