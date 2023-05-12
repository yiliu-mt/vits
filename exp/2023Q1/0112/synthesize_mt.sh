config=exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json
model=model/G_200000.pth
output_dir=exp/2023Q1/0112/audio_out/mt_test
mkdir -p $output_dir

# # To test the speed
# test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_mita.txt
# PVR_GPUIDX=4 python inference_ms.py --warmup --repeat 10 -c $config -t $test_file -m $model -o $output_dir

# To test the accuracy
test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test.txt
output_dir=exp/2023Q1/0112/audio_out/
export MUSA_VISIBLE_DEVICES=3
python inference_ms.py -c $config -t $test_file -m $model -o $output_dir

