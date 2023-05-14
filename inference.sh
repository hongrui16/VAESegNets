python inference.py --model_name VAEUNet --resume .\logs\VAEUNet\run_0\VAEUNet_model_best.pth.tar --with_background #False
python inference.py --model_name VAEUNet --resume .\logs\VAEUNet\run_2\VAEUNet_model_best.pth.tar 


python inference.py --model_name LikeUNet --resume .\logs\LikeUNet\run_1\LikeUNet_model_best.pth.tar 
python inference.py --model_name LikeUNet --resume .\logs\LikeUNet\run_1\LikeUNet_model_best.pth.tar --with_background #False

python inference.py --model_name bscGAN --resume .\logs\bscGAN\run_1\generator_checkpoint.pth.tar


