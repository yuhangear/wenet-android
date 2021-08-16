# Process for deploying WENet to Android



If you want to learn more about the WENET model, you can read the link. This article focuses on the detailed process of training and deployment.

+ https://zhuanlan.zhihu.com/c_1346070871304269824
+ http://placebokkk.github.io/wenet/2021/06/04/asr-wenet-nn-1.html

### Configure the WENET environment

+ Download wenet

  ```
  git clone https://github.com/wenet-e2e/wenet.git
  ```

+ install Conda: https://docs.conda.io/en/latest/miniconda.html

+ Create the Conda environment

  ```
  conda create -n wenet python=3.8
  conda activate wenet
  pip install -r requirements.txt
  conda install pytorch==1.6.0 cudatoolkit=10.1 torchaudio=0.6.0 -c pytorch
  
  # GPU 3090 is available ,Perform the following configuration
  conda create -n wenet python=3.8
  conda activate wenet
  pip install -r requirements.txt
  conda install pytorch torchvision torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
  ```

+ Pre-training preparation

  Based on the model of Librispeech training Online, there are two directories s0 and S1 in example/librispeech directory. Among them, S1 will use the Kaldi script to extract features in advance and save them. While the feature extraction of S0 is completed in CollateFunc (), the feature is extracted during the training process and will not be stored.

  ```
  cd example/librispeech/s0
  ```

  Setting the configuration file

  + train_conformer.yaml ：offline model
  + train_conformer_bidecoder_large.yaml ：full attention
  + train_u2++_conformer.yaml：U2++ attention exists in L2R and R2L decoder
  + train_unified_conformer.yaml ：online model

  When training the online model, change train_config=conf/train_unified_conformer.yaml in run.sh

  modify "run.sh" to ensure the PYTHONPATH of the conda environment

  + source $ANACONDA_ROOT/bin/activate your_wenet

  + export PYTHONPATH=./

  + #Delete during the training phase that does not rely on the Kaldi script

    <del>[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh</del>

  Modify run.sh based on the server environment

  + Modify stage and stop_stage to determine the training phase
  + modify : export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

  + If the server is not using cluster directly：./run.sh or nohup ./run.sh > log 2 1>&1 &
  + If the server has a cluster, run the run.sh command based on the service configuration.
    + cmd="./slurm.pl"
    + modify "wenet/bin/train.py " to "./slurm.pl --quiet --gpu 1 ./log.train python wenet/bin/train.py  --gpu 0"
    + Submit the task：sbatch -o ./log   ./run.sh

### Training model

+ Stage 0: Prepare Training data

  In this stage, `local/aishell_data_prep.sh` organizes the original aishell-1 data into two files:

  - **wav.scp** each line records two tab-separated columns : `wav_id` and `wav_path`
  - **text** each line records two tab-separated columns : `wav_id` and `text_label`

​       If you want to train using your customized data, just organize the data into two files `wav.scp` and    `text`, and start from `stage 1`.

+ Stage 1: Extract optinal cmvn features

  `example/aishell/s0` uses raw wav as input and and [TorchAudio](https://pytorch.org/audio/stable/index.html) to extract the features just-in-time in dataloader. So in this step we just copy the training wav.scp and text file into the `raw_wav/train/` dir.

  `tools/compute_cmvn_stats.py` is used to extract global cmvn(cepstral mean and variance normalization) statistics. These statistics will be used to normalize the acoustic features. Setting `cmvn=false` will skip this step.

+ Stage 2: Generate label token dictionary

  The dict is a map between label tokens (we use characters for Aishell-1) and the integer indices.

  An example dict is as follows

  ```
  <blank> 0
  <unk> 1
  
  ...
  
  <sos/eos> 4232
  ```

  - `<blank>` denotes the blank symbol for CTC.
  - `<unk>` denotes the unknown token, any out-of-vocabulary tokens will be mapped into it.
  - `<sos/eos>` denotes start-of-speech and end-of-speech symbols for attention based encoder decoder training, and they shares the same id.

+ Stage 3: Prepare WeNet data format

  This stage generates a single WeNet format file including all the input/output information needed by neural network training/evaluation.

  See the generated training feature file in `fbank_pitch/train/format.data`.

  In the WeNet format file , each line records a data sample of seven tab-separated columns. For example, a line is as follows (tab replaced with newline here):

  ```
  utt:BAC009S0764W0121
  feat:/export/data/asr-data/OpenSLR/33/data_aishell/wav/test/S0764/BAC009S0764W0121.wav
  feat_shape:4.2039375
  text:甚至出现交易几乎停滞的情况
  token:甚 至 出 现 交 易 几 乎 停 滞 的 情 况
  tokenid:2474 3116 331 2408 82 1684 321 47 235 2199 2553 1319 307
  token_shape:13,4233
  ```

  `feat_shape` is the duration(in seconds) of the wav.

+ Stage 4: Neural Network training

  **If the model does not converge well, we need to lower the learning rate**

  The config of neural network structure, optimization parameter, loss parameters, and dataset can be set in a YAML format file.

  In `conf/`, we provide several models like transformer and conformer. see `conf/train_conformer.yaml` for reference.

  - Use Tensorboard

  The training takes several hours. The actual time depends on the number and type of your GPU cards. In an 8-card 2080 Ti machine, it takes about less than one day for 50 epochs. You could use tensorboard to monitor the loss.

  ```
  tensorboard --logdir tensorboard/$your_exp_name/ --port 12598 --bind_all
  ```

+ Stage 5: Recognize wav using the trained model

+ This stage shows how to recognize a set of wavs into texts. It also shows how to do the model averaging.

  - Average model

  If `${average_checkpoint}` is set to `true`, the best `${average_num}` models on cross validation set will be averaged to generate a boosted model and used for recognition.

  - Decoding

  Recognition is also called decoding or inference. The function of the NN will be applied on the input acoustic feature sequence to output a sequence of text.

  Four decoding methods are provided in WeNet:

  - `ctc_greedy_search` : encoder + CTC greedy search
  - `ctc_prefix_beam_search` : encoder + CTC prefix beam search
  - `attention` : encoder + attention-based decoder decoding
  - `attention_rescoring` : rescoring the ctc candidates from ctc prefix beam search with encoder output on attention-based decoder.

  In general, attention_rescoring is the best method. Please see [U2 paper](https://arxiv.org/pdf/2012.05481.pdf) for the details of these algorithms.

  `--beam_size` is a tunable parameter, a large beam size may get better results but also cause higher computation cost.

  `--batch_size` can be greater than 1 for “ctc_greedy_search” and “attention” decoding mode, and must be 1 for “ctc_prefix_beam_search” and “attention_rescoring” decoding mode.

  - WER evaluation

+ Stage 6: Export the trained model

  `wenet/bin/export_jit.py` will export the trained model using Libtorch. The exported model files can be easily used for inference in other programming languages such as C++.

### Deploy the model to android phones

+ Install androidstudio in a window environment
  + https://www.runoob.com/android/android-studio-install.html
  + https://developer.android.google.cn/studio
+ Installing Android Studio requires dependency packages: Launch Android Studio, open preferences, and search for SDK. After opening the SDK Tools TAB, we need to install some build Tools:
  - `Android SDK Build-Tools`: 30.0.3
  - `NDK`: 22.0.7026061
  - `Android SDK Command-line Tools`: 4.0.0
  - `CMake`: 3.10.2.4988404
  - `Android Emulator`
  - `Android SDK Platform-tool`

![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/06.png)

+ Get your Android phone ready and turn on the USB debug option. When you're ready, connect your phone to your computer.
+ Prepare wenet Android project
  + Wenet Android project code download in the window environmenthttps://github.com/wenet-e2e/wenet/archive/refs/heads/main.zip
  + unzip
  + Need to train in the service area (Linux) model related files, download to the Window environment
    + exp/sp_spec_aug/final.zip
    + data/dict/words.txt（Note that there may be other names）
    + 将上诉文件放在window环境下wenet-android项目文件路径中
      + wenet-main\runtime\device\android\wenet\app\src\main\assets\
  + ` The contents of the directory(core) directly copied to ` device/android/wenet/app/SRC/main/CPP ` directory
  + ![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/04.png)
+ build and   run
  + Import the project in Android Studio
  + ![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/02.png)
  + ![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/03.png)
  + Click the run

![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/01.png)

![Alt text](https://github.com/yuhangear/wenet-android/tree/main/img/05.png)
