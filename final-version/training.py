'''
Authors:     Ferran Carrascosa, Oscar Casanovas, Alessandro Pintaudi, Martí Pomés 
Project:    AIDL Music Group

'''
import os
from music_data_utils import MidiDataset, list_midi_files
import seq2seq
from torch.utils import data
import torch


def training(MIDI_SOURCE):
    QNSF=4 #  Quarter Note Sampling frequency for dividing time

    COD_TYPE = 2  #  1 is 88 notes, 2 is 176 where position 0 is starting note, 1 is continuation note 

    SEQ_LEN=8*4*QNSF  # input note sequence   (SEQ_LEN / QNSF) 

    BATCH_SIZE_TRAIN=16  # batch for the train 
    BATCH_SIZE_VAL=16  # batch for the validation
    BATCH_SIZE_TEST=16  # batch for the test
    EPOCHS=20

    DATA_LEN_TRAIN=5  #Number of songs used for Data Generator (-1 for all) for train
    DATA_LEN_VAL=5  #Number of songs used for Data Generator (-1 for all) for validation
    DATA_LEN_TEST=5   #Number of songs used for Data Generator (-1 for all) test

    #MIDI_SOURCE = "tempData/midi" #For Bach Choral a    

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

    model = seq2seq.Seq2Seq(input_dim=88*COD_TYPE, rnn_dim=512, rnn_layers=2, thr=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


#Train Model

    model.training_network(training_generator,learning_rate=1e-4,epochs=EPOCHS, teacher_forcing_val=0.5, tearcher_forcing_strat="fix", focal_alpha=0.75, focal_gamma=2.0)
        
    return 0
    
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create and train a model')
    parser.add_argument('--midi_source', metavar='path', required=True,
                      help='the path to MIDI DATASET')
    args = parser.parse_args()
    training(MIDI_SOURCE=args.midi_source)  
