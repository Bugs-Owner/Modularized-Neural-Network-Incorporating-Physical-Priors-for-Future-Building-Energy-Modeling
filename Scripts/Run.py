from Utils import ddpred
import argparse

def run():
    parser = argparse.ArgumentParser(description='SeqPINN')
    # Para for SeqPINN
    parser.add_argument('--mode', type=str, choices=['train', 'load', 'check'], default='train',
                        help='Mode to run the script: train or load')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='threshold for early stopping')
    parser.add_argument('--training_batch', type=int, default=512,
                        help='batch size for data input')
    parser.add_argument('--resolution', type=int, default=15,
                        help='time resolution')
    parser.add_argument('--enco', type=int, default=24,
                        help='encoder length')
    parser.add_argument('--deco', type=int, default=96,
                        help='decoder length')
    parser.add_argument('--trainday', type=int, default=60,
                        help='training data size')
    parser.add_argument('--startday', type=int, default=0,
                        help='training start day')
    parser.add_argument('--testday', type=int, default=1,
                        help='testing data size')
    parser.add_argument('--check_terms', type=str, default='HVAC',
                        help='checking objective')
    parser.add_argument('--path', type=str, default='../Dataset/Eplus_Generator/Eplus_train/SFH_Denver_current_TMY/SFH_Denver_current_TMY.csv',
                        help='training data path')
    parser.add_argument('--plott', type=str, default='all',
                        help='result: only plot prediction result; all: prediction results + checking results')
    parser.add_argument('--HVAC_module', type=str, default='linear',
                        help='select HVAC module')
    parser.add_argument('--modeltype', type=str, default='SeqPINN',
                        help='Baseline and SeqPINN')
    parser.add_argument('--scale', type=int, default=2,
                        help='epochs')
    # Para for Eplus
    parser.add_argument('--Eplusmdl', type=str, default='SFH',
                        help='DOE building type')
    parser.add_argument('--city', type=str, default='Denver',
                        help='City name')
    parser.add_argument('--term', type=str, default='current',
                        help='Time span, including current, mid and long')
    parser.add_argument('--sce', type=str, default='TMY',
                        help='Weather scenario, including TMY, EWY, ECY')
    parser.add_argument('--setpoint', type=int, default=24,
                        help='Temperature setpoint')
    parser.add_argument('--setback', type=int, default=8,
                        help='Temperature setback')
    # Para for DPC
    parser.add_argument('--dpclr', type=float, default=0.01,
                        help='DPC learning rate')
    parser.add_argument('--dpcepochs', type=int, default=200,
                        help='DPC epochs')

    args = parser.parse_args()
    SeqPINN = ddpred()
    SeqPINN.data_ready(args=args)
    if args.mode == 'train':
        SeqPINN.train()
        SeqPINN.train_valid_loss_plot()
        SeqPINN.test()
    elif args.mode == 'load':
        SeqPINN.load()
        SeqPINN.test()
        SeqPINN.prediction_show()
    elif args.mode == 'check':
        SeqPINN.load()
        SeqPINN.test()
        SeqPINN.check()
        SeqPINN.check_show()


if __name__ == "__main__":
    run()

