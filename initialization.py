import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument("--name", default='test', help="if you want train, please write train,else write test in this") 
    parser.add_argument("--snapshot_dir", default='logdir', help="path of snapshots") 
    parser.add_argument("--out_dir", default='output', help="path of train outputs")     
    parser.add_argument("--image_size", type=int, default=[256, 512], help="load image size") 
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed") 
    parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') 
    parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch') 
    parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") 
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') 
    parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") 
    parser.add_argument("--write_pred_every", type=int, default=101, help="times to write image.") 
    parser.add_argument("--save_pred_every", type=int, default=10000, help="times to save ckpt.") 
    parser.add_argument("--x_train_data_path", default='city/Cityspace/trainA', help="path of x training datas.") 
    parser.add_argument("--y_train_data_path", default='city/Cityspace/trainB', help="path of y training datas.")     
    
    parser.add_argument("--test_data_path", default='city/Cityspace/test', help="path of x test dataset") 
    parser.add_argument("--test_output", default='test_output',help="Output Folder") 
    args = parser.parse_args()
    
    return args