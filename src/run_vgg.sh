CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_wheat_imbal --phase train --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_maize_imbal --phase train --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_rice_imbal --phase train --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_sorg_imbal --phase train --data_path /opt/data1/dyw/GrainSet/sorg
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_wheat_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_maize_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_rice_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=1,3 python src/train.py --model_name vgg19 --save_name vgg19_sorg_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/sorg