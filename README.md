# ml-momentum

環境設定
```
conda create -n ml-mom python=3.10 -y
conda activate ml-mom
conda install -y numpy psutil pyyaml rich matplotlib tqdm statistics
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y pyg -c pyg
```


input用データの作成方法

geant4 data is here
```
/home/had/sryuta/public/geant4_E72/gen7208
```

root file -> csv fileにする必要があり、そのためのマクロが```root2csv.C```になる。コンパイルするときは
```
make NAME=root2csv.C
```
でできて、実行する際は
```
./root2csv rootfileのpath 使用するデータ数
```
のようにする。使用するデータ数を指定しない場合はすべてのデータが使われる（最大は500000イベント）。


機械学習の実行は```estimate_mom.py```を実行すればよく、読み込むデータは内部にpathを指定するところがあるはず。
