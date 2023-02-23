config=exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q1/0112/tuning_mita_bc_chatbot10/G_200000.pth
output_dir=exp/2023Q1/0112/audio_out/mt_test
mkdir -p $output_dir

test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_mita.txt
CUDA_VISIBLE_DEVICES=5 python inference_ms.py --warmup --repeat 10 -c $config -t $test_file -m $model -o $output_dir
