'''
Authors:     Ferran Carrascosa, Oscar Casanovas, Alessandro Pintaudi, Martí Pomés 
Project:    AIDL Music Group

'''
import os
from music_data_utils import MidiDataset, list_midi_files
import seq2seq
from torch.utils import data
import torch


def training(MIDI_SOURCE,QNSF=4,COD_TYPE=2,SEQ_LEN=128,BATCH_SIZE=16,EPOCHS=20,DATA_LEN_TRAIN=10,DATA_LEN_VAL=5,DATA_LEN_TEST=5,RNN_DIM=512,RNN_LAYERS=2,TEACHER_FORCING=0.5,ALPHA=0.65,GAMMA=2.0,LR=1e-4):
    
    '''
    QNSF=4 #  Quarter Note Sampling frequency for dividing time

    COD_TYPE = 2  #  1 is 88 notes, 2 is 176 where position 0 is starting note, 1 is continuation note 

    SEQ_LEN=8*4*QNSF  # input note sequence   (SEQ_LEN / QNSF) 

    EPOCHS=20

    DATA_LEN_TRAIN=5  #Number of songs used for Data Generator (-1 for all) for train
    DATA_LEN_VAL=5  #Number of songs used for Data Generator (-1 for all) for validation
    DATA_LEN_TEST=5   #Number of songs used for Data Generator (-1 for all) test

    #MIDI_SOURCE = "tempData/midi" #For Bach Choral a    
    '''
    BATCH_SIZE_TRAIN=BATCH_SIZE  # batch for the train 
    BATCH_SIZE_VAL=BATCH_SIZE
    BATCH_SIZE_TEST=BATCH_SIZEs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#DataLoader and DataGenerator 

    #Read Files (returns a 3-d tuple of list of files)
    midi_files  = list_midi_files(MIDI_SOURCE, DATA_LEN_TRAIN, DATA_LEN_VAL, DATA_LEN_TEST, randomSeq=True)

    #Init Midi Model
    dataset_train=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files[0])
    dataset_val=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files[1])
    dataset_test=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files[2])

    #Generate Data Loader
    training_generator = data.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    validating_generator = data.DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=True)
    testing_generator = data.DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=True)
    

#Init Model

    model = seq2seq.Seq2Seq(input_dim=88*COD_TYPE, rnn_dim=RNN_DIM, rnn_layers=RNN_LAYERS, thr=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


#Train Model

    model.training_network(training_generator,learning_rate=LR,epochs=EPOCHS, teacher_forcing_val=TEACHER_FORCING, tearcher_forcing_strat="fix", focal_alpha=ALPHA, focal_gamma=GAMMA)
    
    torch.save(model, 'models/model_def_{0}_{1}songs.pt'.format(EPOCHS,DATA_LEN_TRAIN))
    
    return model
    
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create and train a model')
    parser.add_argument('--midi_source', metavar='path', required=True,
                      help='the path to MIDI DATASET')
    parser.add_argument('--qnsf', type=int, required=False, default=4)
    parser.add_argument('--cod_type', type=int, required=False, default=2)
    parser.add_argument('--seq_len', type=int, required=False, default=128)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--epochs', type=int, required=False, default=50)
    parser.add_argument('--data_len_train', type=int, required=False,default=10)
    parser.add_argument('--data_len_val', type=int, required=False,default=5)
    parser.add_argument('--data_len_test', type=int, required=False,default=5)
    parser.add_argument('--rnn_dim', type=int, required=False,default=512)
    parser.add_argument('--rnn_layers', type=int, required=False,default=2)
    parser.add_argument('--teacher_forcing', type=float, required=False,default=0.5)
    parser.add_argument('--alpha', type=float, required=False,default=0.7)
    parser.add_argument('--gamma', type=float, required=False,default=2.0)
    parser.add_argument('--lr', type=float, required=False,default=1e-4)
    args = parser.parse_args()
    training(MIDI_SOURCE=args.midi_source,QNSF=args.qnsf,COD_TYPE=args.cod_type,SEQ_LEN=args.seq_len,BATCH_SIZE=args.batch_size,EPOCHS=args.epochs,DATA_LEN_TRAIN=args.data_len_train,DATA_LEN_VAL=args.data_len_val,DATA_LEN_TEST=args.data_len_test,RNN_DIM=args.rnn_dim,RNN_LAYERS=args.rnn_layers,TEACHER_FORCING=args.teacher_forcing,ALPHA=args.alpha,GAMMA=args.gamma,LR=args.lr)  
