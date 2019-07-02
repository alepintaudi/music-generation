'''
functions and classes:
MidiDataset = Torch data loader. Transforms a list of midi files (midi_files) into a pytorch dataloader
list_midi_files = Given a source folder and the number of Train, Validation and test cases that you want, it will return a list of midis for each category

extra info about libraries:
music21 = Open source toolkit. Used to extract midi files into a Python understandable, event-based syntaxis. (https://web.mit.edu/music21/doc/)
'''

import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream

from os import walk
import random
from torch.utils import data

class MidiDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, qnsf, seq_len=25, cod_type=2, midi_files=[]):
        'Initialization'
        if cod_type !=1 and cod_type !=2:  
          raise TypeError("cod_type is not 1 (88 notes) or 2 (176 notes)")
        self.notes = self.get_notes(qnsf=qnsf, cod_type=cod_type, midi_files=midi_files)
        self.qnsf = qnsf
        self.seq_len = seq_len
        self.cod_type = cod_type
        self.midi_source = midi_files
        


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.notes) - self.seq_len*2

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        samples = self.notes[index:(index+self.seq_len*2)]
        x=np.asarray(samples).astype(np.float32)
        
        return (x[0:self.seq_len,:],x[self.seq_len:,:])
        #return  (samples[0:self.seq_len], samples[self.seq_len:])
      
  def get_notes(self, qnsf, cod_type,midi_files):
      """ Get all the notes and chords from the midi files in the ./midi_songs directory """
      notes = []

      for i,file in enumerate(midi_files):

          
          midi = converter.parse(file)

          print("Parsing %s" % file)
          
          
          try: # file has instrument parts
              s2 = instrument.partitionByInstrument(midi)
              notes_to_parse = s2.parts[0].recurse() 
          except: # file has notes in a flat structure
              notes_to_parse = midi.flat.notes

              #initialize piano roll
          length=int(notes_to_parse[-1].offset*qnsf)+1 #count number subdivisions
          notes_song=np.zeros((length,88*cod_type))

          for element in notes_to_parse:
              #print(element.offset, element.duration.quarterLength, element.pitch.midi-21)
              if isinstance(element, note.Note):
                  # cod_type is based on (0,1), (1,0), (0,0)
                  notes_song[int(element.offset*qnsf),cod_type*(element.pitch.midi-21)]=1.0    
                  notes_song[(int(element.offset*qnsf)+1):(int(element.offset*qnsf)+int(element.duration.quarterLength*qnsf)),cod_type*(element.pitch.midi-21)+1]=1.0   

              elif isinstance(element, chord.Chord):
                  for note_in_chord in element.pitches:
                    # cod_type is based on (0,1), (1,0), (0,0)
                    notes_song[int(element.offset*qnsf),cod_type*(note_in_chord.midi-21)]=1.0   
                    notes_song[(int(element.offset*qnsf)+1):(int(element.offset*qnsf)+int(element.duration.quarterLength*qnsf)),cod_type*(note_in_chord.midi-21)+1]=1.0   
              #print(notes_song.shape)
                   
          notes+=[list(i) for i in list(notes_song)]
      return notes


def list_midi_files(midi_source, data_len_train, data_len_val , data_len_test, randomSeq=True, seed = 666):
    """ Given a directory with midi files, return a tuple of 3 lists with the filenames for the train, validation and test sets 
        The funcion can apply some randomeness in the files selection order although it can be reproduced cause of the seed number
    """
    
  midi_files_all = []
  # iterate over all filenames in the midi_Source
  for (dirpath, dirnames, filenames) in walk(midi_source):
    midi_files_all.extend(filenames)
  midi_files_all = [ glob.glob(midi_source+"/"+fi)[0] for fi in midi_files_all if fi.endswith(".mid") ]
  
  # we apply som randomnes in the midi selection to prevent some systematic dependences between train, validation and test.
  if randomSeq:
    random.seed( seed )
    midi_files_sel = random.sample(midi_files_all, (data_len_train + data_len_val + data_len_test))
  else:
    midi_files_sel = midi_files_all
    
  
  return midi_files_sel[:data_len_train], midi_files_sel[data_len_train:(data_len_train + data_len_val)], midi_files_sel[(data_len_train + data_len_val):]

def cleanSeq(x, cod_type):
    """ Given a pianoroll x tith 3 dimensions: batches, timesteps and notes*cod_type, the function returns the
        corrected pianoroll adapted to the codificacion rules: a one in the odd position (continuation note),
        has to be precedded in the previous timestep by a one in the even or odd position. If exists, this issue,
        then we moove this one to the even position. 
        Another rule is that it can't exist a one in the even and odd position in the same time step. If exists, 
        then the one in the ood is removed.
    """
  if cod_type==2:
    # first row correction
    notePos = np.where(x[0,:]==1)[0]        # where are the ones
    notePosWrong = np.array([aa%2 for aa in notePos])  
    notePosWrongPos = np.where(notePosWrong==1)[0]     # we look if corespond to odd positions
    x[0,notePos[notePosWrongPos]]=0                    # if present, we correct to 0 the odd
    x[0,notePos[notePosWrongPos]-1]=1                  # if present, we correct to 0 the correspondent even position
    
    # other rows correction
    # when we have 1 in even and odd position, then remove the second one 
    x[:,[ii for ii in range(x.shape[1]) if ii%2==1 ] ] = x[:,[ii for ii in range(x.shape[1]) if ii%2==1 ] ] * (1-x)[:,[ii for ii in range(x.shape[1]) if ii%2==0 ] ]

    # now we want to compare with previous row
    x_0 = x[1:,1:]     #   base values to correct
    x_1 = x[0:-1,1:]   #   previous same column
    x_11 = x[0:-1,0:-1]#   previous even column
    x_sel = np.zeros(x_0.shape)  # odd columns f x
    x_sel[:, [ii for ii in range(x_sel.shape[1]) if ii%2==0 ] ] =1   # corespond to even columns of x_sel
    x_2 = ((x_1==0) & (x_11==0) & (x_0==1) & (x_sel==1))   # columns with one in odd that not have any in even or odd correspondece in previous row
    x[1:,1:][x_2] = 0    # correction in unpair row
    x[1:,:-1][x_2] = 1
    
  return(x)


def create_midi(prediction_output, qnsf = 4, cod_type=2, midiOutputFile='test_output.mid'):
    """ convert the output (as a numpy array) from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for i,pattern in enumerate(prediction_output):
      # pattern is a note
      notes = []
      noteIndex = np.where(pattern==1)[0]
      noteIndex = noteIndex[[ii % cod_type ==0 for ii in noteIndex]]
      if len(noteIndex)>0:  
        for current_note in noteIndex:
          new_note = note.Note(int(current_note/cod_type + 21))
          new_note.storedInstrument = instrument.Piano()
          new_note.offset = offset
          # duration.quarterLength in case cod_type==2
          if cod_type==2 :
            # sequence of duration equal to one in the odd position
            auxDuration = np.where(prediction_output[(i+1):,current_note + 1]==1)[0]
            # initialize duration
            minimum_value = 0
            if len(auxDuration)>0:
              # minimum position where we have a consecuive sequence at 0
              # we add one to include complete sequences
              minimum_value = np.array(range(len(auxDuration)+1))[~np.isin(range(len(auxDuration)+1),auxDuration)].min()

            # we calcuate the minimum number in the sequance :len(auxDuration) that is not 
            # in the sequence, add one and divide by QNSF
            new_note.duration.quarterLength = ( minimum_value + 1.0)/qnsf
            
          output_notes.append(new_note)

      offset += 1.0/qnsf
    
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=midiOutputFile)
    print("created midi file: ",midiOutputFile )
