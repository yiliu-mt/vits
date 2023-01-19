config=exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json
test_file=exp/2023Q1/0112/test.txt
model=/nfs1/yi.liu/tts/vits/exp/2023Q1/0112/tuning_mita_bc_chatbot10/G_200000.pth
output_dir=exp/2023Q1/0112/audio_out

mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0 python inference_ms.py -c $config -t $test_file -m $model -o $output_dir