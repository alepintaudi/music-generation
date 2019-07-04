    
'''
Authors:     Ferran Carrascosa, Oscar Casanovas, Alessandro Pintaudi, Martí Pomés 
Project:    AIDL Music Group
'''
import os
from music_data_utils import MidiDataset, list_midi_files, create_midi, cleanSeq
import seq2seq
from torch.utils import data
import torch

def generating(MODEL_PATH,MIDI_FILE,SEQ_LEN=200,HIDDEN_INIT="zeros"):
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	seq2seqMod = torch.load(MODEL_PATH)

	sequence=seq2seq.to(device).predict(seq_len=SEQ_LEN,hidden_init=HIDDEN_INIT)

	create_midi(sequence, qnsf = 4, cod_type=2, midiOutputFile=MIDI_FILE)

	return sequence

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Create and train a model')
	parser.add_argument('--model_path', metavar='path', required=True)
	parser.add_argument('--seq_len', type=int, required=False, default=200)
	parser.add_argument('--hidden_init', type=string, required=False, default="zeros")
	parser.add_argument('--midi_file', type=string, required=True)

	args = parser.parse_args()
	generating(MODEL_PATH=args.model_path,SEQ_LEN=args.seq_len,HIDDEN_INIT=args.hidden_init,MIDI_FILE=args.midi_file)
			   
	
	
