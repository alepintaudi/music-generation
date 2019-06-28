QNSF = 4   #  Quarter Note Sampling frequency for dividing time 

COD_TYPE = 2  #  1 is 88 notes, 2 is 176 where position 0 is starting note, 1 is continuation note 

SEQ_LEN=12*4*QNSF  # secuencia de notas de entrada   (SEQ_LEN / QNSF)  son las notas de entrada

print("Input quarter notes:{} with code_type:{} ".format(SEQ_LEN / QNSF, COD_TYPE))

BATCH_SIZE_TRAIN=50  # batch for the train 
BATCH_SIZE_VAL=15  # batch for the validation
BATCH_SIZE_TEST=50  # batch for the test

EPOCHS=100

DATA_LEN_TRAIN=5  #Number of songs used for Data Generator (-1 for all) for train
DATA_LEN_VAL=3  #Number of songs used for Data Generator (-1 for all) for validation
DATA_LEN_TEST=3   #Number of songs used for Data Generator (-1 for all) test
# time 2:20
#DATA_LEN=34 #Number of songs used for Data Generator (-1 for all)
# time 5:10
#DATA_LEN=67 #Number of songs used for Data Generator (-1 for all)

MIDI_SOURCE = "tempData/midi"    #  directory with *.midi files
#MIDI_SOURCE = "/content/Classical-Piano-Composer/midi_songs"   #  direc
#MIDI_SOURCE = "TPDData/TPD/classical"
#MIDI_SOURCE = "TPDData/TPD/jazz"


SILENCE=1 #Encode Silence as a distinct pitch
