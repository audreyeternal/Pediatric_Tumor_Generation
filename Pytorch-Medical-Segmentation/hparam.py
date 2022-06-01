class hparams:

    Program_name = 'deeplab_augment'
    train_or_test = 'test'
    output_dir = 'logs/' + Program_name
    aug = None
    # latest_checkpoint_file = 'checkpoint_latest.pt'
    latest_checkpoint_file = 'checkpoint_0040.pt'
    total_epochs = 100
    epochs_per_checkpoint = 10
    batch_size = 4
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 256,256,1 # if 3D: 512,512,32
    patch_size = 128,128,1 # if 3D: 512,512,32 

    # for test
    patch_overlap = 4,4,0 # if 3D: 4,4,4

    fold_arch = '*.png'
    fold_arch_label = '*.png'
    save_arch = '.png'

    source_train_dir = '/host_project/SPADE/datasets/png_file/train_img_png_T1'
    label_train_dir = '/host_project/SPADE/datasets/png_file/train_label_png_T1'
    # source_train_dir = '/host_project/Pytorch-Medical-Segmentation/data/train/image'
    # label_train_dir = '/host_project/Pytorch-Medical-Segmentation/data/train/label'
    source_test_dir = '/host_project/SPADE/datasets/png_file/val_img_png_T1'
    label_test_dir = '/host_project/SPADE/datasets/png_file/val_label_png_T1'

    output_dir_test = 'results/' + Program_name