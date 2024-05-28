from Utils import ddpred
import argparse

def run():
    parser = argparse.ArgumentParser(description = 'SeqPINN')
    parser.add_argument('--mode', type=str, choices=['train', 'load', 'check'], default='check',
                        help='Mode to run the script: train or load')
    parser.add_argument('--lr', type = float, default = 0.001,
                       help = 'learning rate')
    parser.add_argument('--epochs', type = int, default = 300,
                       help = 'epochs')
    parser.add_argument('--patience', type = int, default = 5,
                       help = 'threshold for early stopping')
    parser.add_argument('--training_batch', type = int, default = 512,
                       help = 'batch size for data input')
    parser.add_argument('--resolution', type = int, default = 15,
                       help = 'time resolution')
    parser.add_argument('--enco', type = int, default = 24,
                       help = 'encoder length')
    parser.add_argument('--deco', type = int, default = 96,
                       help='decoder length')
    parser.add_argument('--trainday', type = int, default = 27,
                       help='training data size')
    parser.add_argument('--startday', type = int, default = 3,
                       help='training start day')
    parser.add_argument('--testday', type = int, default = 1,
                       help='testing data size')
    parser.add_argument('--check_terms', type=str, default='HVAC',
                        help='checking objective')
    parser.add_argument('--path', type = str, default = '../Dataset/Link.csv',
                       help='training data path')

    args = parser.parse_args()
    ddp = ddpred()
    ddp.data_ready(args=parser.parse_args())

    if args.mode == 'train':
        ddp.train()
        ddp.train_valid_loss_plot()
        ddp.test()
    elif args.mode == 'load':
        ddp.load()
        ddp.test()
        ddp.prediction_show()
    elif args.mode == 'check':
        ddp.load()
        ddp.test()
        ddp.check()
        ddp.check_show()


if __name__ == "__main__":
    run()


