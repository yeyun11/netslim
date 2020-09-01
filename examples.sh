# Train with sparsity
python train-cifar.py --arch vgg14 --sparsity 0.0001

# Prune & Fine-tune using NS
python train-cifar.py --arch vgg14 --prune-ratio 0.4 --epochs 40 --lr 0.001 --lr-decay-schedule 1.1 --resume-path output-cifar100-vgg14/bs_sp_0.0001_wd_0.0001/ckpt_last.pth --postfix bsp_0.0001 --test-at-0
# Prune & TFS using NS
python train-cifar.py --arch vgg14 --epochs 256 --prune-ratio 0.4 --resume-path output-cifar100-vgg14/bs_sp_0.0001_wd_0.0001/ckpt_last.pth --postfix bsp_0.0001 --tfs

# Train with a higher sparsity
python train-cifar.py --arch vgg14 --sparsity 0.0002

# Prune & Fine-tune using OT
python train-cifar.py --arch vgg14 --prune-ratio 1 --epochs 40 --lr 0.001 --lr-decay-schedule 1.1 --resume-path output-cifar100-vgg14/bs_sp_0.0002_wd_0.0001/ckpt_last.pth --postfix bsp_0.0002 --test-at-0
# Prune & TFS using OT
python train-cifar.py --arch vgg14 --epochs 256 --prune-ratio 1 --resume-path output-cifar100-vgg14/bs_sp_0.0002_wd_0.0001/ckpt_last.pth --postfix bsp_0.0002 --tfs

# Check the pruned models
#python check-pruned-model.py output-cifar100-vgg14/ns_pr_0.4_wd_0.0001_bsp_0.0001/ckpt_best.pth
#python check-pruned-model.py output-cifar100-vgg14/ot_wd_0.0001_bsp_0.0002/ckpt_best.pth

# Convert the pruned model to ONNX
#python convert2onnx.py output-cifar100-vgg14/ns_pr_0.4_wd_0.0001_bsp_0.0001/ckpt_best.pth
#python convert2onnx.py output-cifar100-vgg14/ot_wd_0.0001_bsp_0.0002/ckpt_best.pth 
