# music-generation
AIDL music generation

## Here the original file with testing results and graphs (not visible in this ReadMe)
https://www.notion.so/aidlmusic/ReadMe-014338557385464a9ede62b3dd04a29b#6ad6da2ba64445dfb43464bac25e5bce

# Task Definition

This project presents the implementation of a basic MIDI compositions generation system while we learn about the implementation of 3 different Neural Network models (LSTM Seq2One with Keras, LSTM Seq2One with PyTorch, LSTM Seq2Seq with PyTorch). We also learn on how to train a Network using Google Compute Engine and provide an overview of the current State of the Art on Music Generation using Deep Learning as well as other simpler but very interesting individual projects.

# Introduction

Throughout the past 4 years, we have seen impressive developments in the field of generative music and Artificial Intelligence thanks to the progress made on Deep Learning technologies.

The goal of this Postgraduate project is to get hands-on experience on building our own Model that is trained with a collection of MIDI files and then is modified to generate novel short composition snippets that are evaluated on their degree of musicality and closeness to passing a touring test on random non-musically trained subjects.

It is important to note that none of the four project participants had previous experience implementing Deep Learning models and that all the knowledge has been acquired during the past 5 months over the UPC School - Artificial Intelligence with Deep Learning Postgraduate Degree. We also want to thank our supervisor Carlos Segura Perales for his dedication, availability and valuable insights provided over the past months.

# Statement of the Problem

Recent projects such as Bachbot (Feynman Liang), Coconet (Google Magenta), Music Transformer (Google Magenta) and MuseNet (OpenAI) among others, have led to technology capable of achieving results like passing the Turing test on certain manually selected output novel piano music pieces and composer styles.

At the same time, several companies have been working on research and commercial applications of similar systems for certain use cases such as described in the Benchmark Section.

During the latest AI Music news boom, many journalists agree that if the current rate of yearly progress on quality and quantity of the musical output of such generative systems, we will start seeing more and more of these deployed commercially for several real world applications, such as royalty-free, low-cost, quickly personalized and on-demand tailored music. These use cases could have a strong “Product/market fit” for new automatic music composing systems able to provide professional quality music at a very affordable rate for companies, freelancers, individuals, music enthusiasts as well as general music listeners.

# Motivation

We started our research looking at the BachBot challenge that inspired us in conducting these experiments

BachBot

[http://bachbot.com/](http://bachbot.com/#/?_k=ult5ym)

Paper:

[http://www.mlmi.eng.cam.ac.uk/foswiki/pub/Main/ClassOf2016/Feynman_Liang_8224771_assignsubmission_file_LiangFeynmanThesis.pdf](http://www.mlmi.eng.cam.ac.uk/foswiki/pub/Main/ClassOf2016/Feynman_Liang_8224771_assignsubmission_file_LiangFeynmanThesis.pdf)

Github

[https://github.com/feynmanliang/bachbot](https://github.com/feynmanliang/bachbot)

# Benchmarks & SOTA

For reference, we believe it is also important to mention the following Deep Learning Music Generation projects that show the output quality of current State of the art commercial and research applications:

- Google Magenta (CocoNet and Music Transformer).
- MuseNet (OpenAI): A deep neural network that can generate 4-minute musical compositions with 10 different instruments, and can combine styles from country to Mozart to the Beatles.
- Spotify (Spotify Creator Technology Research Lab): The lab focuses on making tools to help artists in their creative process.
- Jukedeck: Generating Royalty Free Music for Youtube videos and other applications.
- AI.music: Several Products related to generating tailored versions of songs for each user.
- AIVA: AIVA, the Artificial Intelligence music composer that creates original & personalized music for your projects.
- AMPER: Amper is an AI music composition company that develops tools for content creators of all kinds.
- ALYSIA: ALYSIA allows everyone to create original songs in under 5 minutes. Including lyrics and melodies. Get ready to sing karaoke on your own original music.
- Mubert: Generative Channels. Each generative channel is based on a fixed number of tags which algorithm uses to create endless streams.
- Endel: Personalized audio environments that help you focus and relax.
- IBM Watson Beat: The Watson Beat code employs two methods of machine learning to assemble its compositions; reinforcement learning that using the tenets of modern western music theory to create reward functions, and a Deep Belief Network (DBN) trained on a simple input melody to create a rich complex melody layer.
- Microsoft Research: Music Generation with Azure Machine Learning.
- Sony Flow Machines: Flow Machines is a research and deployment project aimed at providing augmented creativity for music artists.

# Experiments

Here the various experiments that led us to the final model of music generation. At each step we identified the problems we encountered and tried to find a solution to move forward and get a step closer to the target solution we initially defined: generate a model capable of being trained with music and eventually generate one with similar style. 

## 1st Experiment

The way we approached the problem was trying to reproduce what has been done in classical NLP models, in which the text is segmented in small sequences forming a dictionary, used to predict the following sequence of characters. 

Thus we decided to convert MIDI files into text to then pass them through the first developed network, written in Keras.

We based our first experiment on an existing model, which tackled the problem in the same way (converting MIDI of into text and then using an LSTM to generate music), training it with classical music piano MIDI compositions.

Here the Github repository that inspired our first experiment:

[https://github.com/Skuldur/Classical-Piano-Composer/blob/master/predict.py](https://github.com/Skuldur/Classical-Piano-Composer/blob/master/predict.py)

As said the first step is to transform the MIDI files into text and we did that converting the input dataset into a dictionary, in which each note corresponded to a string of letter and number

e.g.

E3 
E-6
E6 
E-5 
A5 
E5 

Here the code to generate the vocabulary of notes

    def get_notes(n=-1):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for i,file in enumerate(glob.glob("/content/Classical-Piano-Composer/midi_songs/*.mid")):
        
        if n==i:
            break  
        
        midi = converter.parse(file)
    
        print("Parsing %s" % file)
    
        notes_to_parse = None
    
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
    
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    with open('/content/Classical-Piano-Composer/data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    return notes

### The Model

In this initial experiment the model presented two layers of LSTM with Dropout and one last layer of fully connected to generate the final vocabulary.

At the end an activation function (softmax) is applied and combined with a cross entropy loss. The model also optimises the results using rmsprop.

    def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
    512,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

### Training

Here the results of the trained model with the classical piano compositions

    0 0.007250308990478516 1.0
    1 0.026912927627563477 1.0
    2 0.00890620518475771 1.0
    3 0.06554309278726578 1.0
    4 0.3337826728820801 0.6666666865348816
    5 0.8159777522087097 0.6666666865348816

[ ... ]

    87 0.0015482902526855469 1.0
    88 0.0032432873267680407 1.0
    89 0.0057816109620034695 1.0
    90 0.0038841168861836195 1.0
    91 0.012463927268981934 1.0
    92 0.006290217395871878 1.0
    93 0.013199646957218647 1.0
    94 0.003600984811782837 1.0

### Example of output

Here below an example of the output we have got from the above model:

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#abef71644574492eb2133332cf75b343)

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#c9509969774e48b088c315052f398fa8](https://www.notion.so/014338557385464a9ede62b3dd04a29b#c9509969774e48b088c315052f398fa8)

### Conclusions of the first Experiment

Since this model is trained without note duration values, all the notes at the input and output have the same fixed duration. This limitation helps this small network to focus only on learning about harmony but it still does not perform amazingly well harmonically. The first part of this example sounds quite strange even if the ending part of this example sounds much better (closer to a human composition).

## 2nd Experiment

The biggest challenge with the second experiment has been transforming a first working model from Keras to Pytorch.

In order to achieve this first iteration we run a research to discover similar models within the music field that had been developed in Pytorch.

We particularly focused on a Network used for style transfer in which we specifically appreciated the way they generated the text-based original MIDI file.

Here is an extract from the paper:

*Firstly, we quantize each MIDI file to align to the particular time interval thereby eliminating imprecisions of the performer. Secondly, we encode the input MIDI file into a T ×P matrix where T is the number of time steps in the song and P is the number of pitches in the instrument (Example, for piano with 88 keys we have 88 pitches). Further, in each value of the matrix we encode the information regarding note using a 2-D vector i.e., [1 1] note articulated, [0 1] note sustained and [0 0] note off.*

(**Source** - ToneNet : A Musical Style Transfer [https://towardsdatascience.com/tonenet-a-musical-style-transfer-c0a18903c910](https://towardsdatascience.com/tonenet-a-musical-style-transfer-c0a18903c910))

Giving the similar dataset (piano composer) and similar problem to solve, we decided to adopt the 88 piano keys to define the length of our vector of notes.

Another concept that we introduced was the duration, for each of the time steps of the composition. Having a vector of only notes identified by the sole activation of a piano key would have not allowed us to create an homogeneous composition.

Thus we passed from a 0 / 1 activation for each note to one based on a couples of binary numbers where:

(1,0) = the note starts

(0,1) = the note continues

(0,0) = the note is not played

### The Model

The model chosen was an LSTM based on a sequence to one generator, in which each sequence of notes, only generated one single note as output.

    class NextNoteModel(nn.Module):
    def **init**(self, input_dim, rnn_dim=512, rnn_layers=2):
    super(NextNoteModel, self).**init**()
    self.rnn = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
    self.classifier = nn.Sequential(
    nn.Linear(rnn_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, input_dim)
    )

At this point in order to measure of error between computed outputs and the desired target outputs of the training data, we decided to introduce CrossEntropy Loss into our model.

    self.loss_function = nn.CrossEntropyLoss() 

    def forward(self, x):

    output, (hn, cn) = rnn(input, (h0, c0))

    output, (hn, cn) = self.rnn(x)
      return self.classifier(output[:,-1,:]) #no hace falta la softmax

    def loss(self, x,y):
    y_pred = y.argmax(dim=1)
    return self.loss_function(x,y_pred)

    def accuracy(self, x, y):
    x_pred = x.argmax(dim=1)
    y_pred = y.argmax(dim=1)
    return (x_pred == y_pred).float().mean()

### Training

Once we started training our network we could notice that while the training loss was going down, the validation loss initially decreased to eventually end up higher then it was at the beginning.

Accuracy increased during training but remained pretty low during validation, tending to be close to zero. 

    0 3.900423049926758 4.751437187194824 0.1875 0.0
    1 3.7298624515533447 4.751437187194824 0.125 0.0
    2 3.70011830329895 4.751437187194824 0.125 0.0
    3 3.7687315940856934 4.751437187194824 0.0625 0.0
    4 3.727893590927124 4.751437187194824 0.0 0.0
    5 3.28770112991333 4.751437187194824 0.0625 0.0

[ ... ]

    50 1.5549570322036743 5.31388521194458 0.625 0.2857142984867096
    51 1.3869068622589111 5.31388521194458 0.6875 0.2857142984867096
    52 1.51986825466156 5.31388521194458 0.625 0.2857142984867096
    53 1.9102293252944946 5.31388521194458 0.4375 0.2857142984867096
    54 1.8994156122207642 5.31388521194458 0.4375 0.2857142984867096
    55 1.4201409816741943 5.31388521194458 0.5625 0.2857142984867096

[ ... ]

    96 0.916029691696167 6.139454364776611 0.6875 0.0
    97 1.110654354095459 6.139454364776611 0.6875 0.0
    98 0.9400495886802673 6.139454364776611 0.6875 0.0
    99 1.3639230728149414 6.139454364776611 0.5625 0.0

**Learning Curves**

***Loss*** 

Despite the low numbers obtained in terms of both loss and accuracy, we observed that during the training and validation steps, our network overfitted.

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#ba03ced3160f4c0d94be56340b840fcf)

***Accuracy***

A similar behaviour can be observed for the learning curves of accuracy, where from an initiation convergence the two curves end up going far from each other.

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#3e76d822be7b497fb99210ac2f3b339c)

### Testing

[Results](https://www.notion.so/5289c8145ba54f46ba3a76dab90adde4)

### Example of output

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#be15dddf95c8466385221d61c0c0e36c](https://www.notion.so/014338557385464a9ede62b3dd04a29b#be15dddf95c8466385221d61c0c0e36c)

### Conclusions of the Experiment

Given the nature of the model, a Sequence to One, the network almost always predicted the same note. Thus instead of generating an harmonious composition the model created a very flat piece of music, far from being considered real.

Also the accuracy shown during both training and testing is very low probably due to the absence of any historical record kept by the model that is always predicting a new output based on a single note generated in previous time.

## 3rd Experiment

In this third experiment we started from where we finished with the earlier experiment and tried to improve it. Given the biggest problem, the monotony of the generated sequence, we decided to move towards a more standard Sequence to Sequence network.

Here the decoder was finally generating a sequence of the same length as the one sent to the encoder.

### The Model

**Encoder**

The encoder is made of 2 LSTM layers having 512 neurons each with a dropout layer in between them.

A linear transformation is then applied to reduce fdimensionality to 256

It follows an activation function used for transforming the summed weighted input from the node into the activation of the node or output for that input.

A dropout layer is then added before applying a Cross Entropy Loss function to the results. 

    class Seq2Seq(nn.Module):
    def **init**(self, input_dim, rnn_dim=512, rnn_layers=2):
    super(Seq2Seq, self).**init**()
    self.encoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
    self.decoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
    self.classifier = nn.Sequential(
    nn.Linear(rnn_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, input_dim)
    )
    self.loss_function = nn.CrossEntropyLoss() 

Results are then normalised using Softmax.

      self.norm = nn.Softmax()v

**Decoder**

The decoder used in here is providing what we missed in the previous experiment: an output sequence of the same length as the one injected in the encoder. In this way the network is able to generate a sequence made of logical sounds.

    def forward(self, x,y):          
    
          output, (hn, cn) = self.encoder(x)
    
          output, (hn, cn) = self.decoder(y, (hn,cn))
          
          shape = output.shape
          
          x=output.unsqueeze(2)
          
          x = self.classifier(x)
          
          x = x.view(shape[0],shape[1],-1)
          
          return x

### Training

Row results of Loss and Accuracy for both Training and Validation.

    0 1.0397608280181885 0.7631075978279114 0.8751183748245239 0.9065656661987305
    1 0.6295673847198486 0.7631075978279114 0.9144176244735718 0.9065656661987305
    2 0.53970867395401 0.7631075978279114 0.927438497543335 0.9065656661987305
    3 0.6158638000488281 0.7631075978279114 0.911695122718811 0.9065656661987305
    4 0.5996387600898743 0.7631075978279114 0.9122869372367859 0.9065656661987305
    5 0.5661443471908569 0.7631075978279114 0.917613685131073 0.9065656661987305

[ ... ]

    94 0.03222033753991127 0.34312954545021057 0.9917140603065491 0.9595959782600403
    95 0.03811696544289589 0.34312954545021057 0.9889914989471436 0.9595959782600403
    96 0.027607234194874763 0.34312954545021057 0.9906486868858337 0.9595959782600403
    97 0.029885871335864067 0.34312954545021057 0.990293562412262 0.9595959782600403
    98 0.030058154836297035 0.34312954545021057 0.9912405610084534 0.9595959782600403
    99 0.034393422305583954 0.34312954545021057 0.989938497543335 0.9595959782600403

**Learning Curves**

***Loss*** 

Both curves reduce the loss being close to 0 without overfitting. The validation curve in fact starts increasing at the end of the training. 

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#05499da4e4774d2694bb9d911caff89f)

***Accuracy***

The two curves are starting close to each other, to then separated and eventually end up both above the 95% of accuracy.

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#a6950f132de04bdc91cdc5c0031ff0f8)

After training the system with training and validation dataset and using the resulted weights, here below the mean and standard deviation of both Loss and Accuracy during the testing phase.

### Testing

[Results](https://www.notion.so/62b01d1b3ea2464e88d88853a0959274)

### Example of output

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#27d52bd5d00c46149deb15496f90b7d9](https://www.notion.so/014338557385464a9ede62b3dd04a29b#27d52bd5d00c46149deb15496f90b7d9)

### Conclusions of the Experiment

Contrary to the previous experiment here the loss function is able to detect sequences and therefore the output sound is no longer made of single monotonic notes.

Yet, although the network is now able to generate a sequence of (more or less) audible sounds, the biggest problem remains the inability of generating polyphonic compositions.

## 4th Experiment

In this forth experiment we tried to improve the previous model, starting by solving its main problem of generating only monophonic music.

### The Model

The first step to achieve our main goal was introducing a Binary Cross Entropy Loss. This loss function enabled us to parse more than a line of instrument at the time and so ending up generating polyphonic sounds.

Here we had two choices:

- BCELoss
- BCEWithLogitsLoss

The choice fell onto the second one as this loss combines a Sigmoid layer and the BCELoss in one single class. *This version*, according to pytorch main documentation, *is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.*

    self.loss_function = nn.BCEWithLogitsLoss()

On top of that we decided to introduce teacher forcing (initially set up with a threshold of 0.5) in order to improve model skill and stability at training time.

> Models that have recurrent connections from their outputs leading back into the model may be trained with teacher forcing.

— Page 372, [Deep Learning](http://amzn.to/2wHImcR), 2016.

In this way we were able to quickly and efficiently train our model that was using the ground truth from a prior time step as input.

    def forward(self, x,y,teacher_forcing_ratio = 0.5):

    output, (hn, cn) = self.encoder(x)
    
      seq_len = y.shape[1]
    
      outputs = torch.zeros(y.shape[0], seq_len, self.input_dim).to(device)
    
      input = y[:,0,:].view(y.shape[0],1,y.shape[2])
    
      for t in range(1, seq_len):
          output, (hn, cn) = self.decoder(input, (hn, cn))
    
          teacher_force = random.random() < teacher_forcing_ratio
          
          shape = output.shape
          
          x=output.unsqueeze(2)
    
          x = self.classifier(x) #no hace falta la softmax
    
          x = x.view(shape[0],shape[1],-1)
          
          output = (x > self.thr).float()
          
          input = (y[:,t,:].view(y.shape[0],1,y.shape[2]) if teacher_force else output.view(y.shape[0],1,y.shape[2]))
          
          outputs[:,t,:] = x.view(y.shape[0],-1)
    
      return outputs

At this point we faced a problem, in which our loss was rapidly going below 0, due to the nature of the text, in which most of the times, 86-87 notes of the 88 available were easily guessed as not being played and having a value of 0.

We learnt that in image classification, to solve tasks of object detection, some models (e.g. Yolo) had been using a type of loss that allowed them to weight each loss taking into account the several possible objects to be detected.

As our model presented a similar problem, we decided to introduce focal loss within our network.

    def focal_loss(self, x, y):
    '''Focal loss.
    Args:
    x: (tensor) sized [batch_size, n_forecast, n_classes(or n_levels)].
    y: (tensor) sized like x.
    Return:
    (tensor) focal loss.
    '''
    alpha = 0.5
    gamma = 2.0

    x = x.view(-1,x.shape[2])
        y = y.view(-1,y.shape[2])
    
        t = y.float()
    
        p = x.sigmoid().detach()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')

In this model we also introduced a new hyper parameter, brought by BCE Loss function that expects a threshold value to be checked against the output, when applying the Sigmoid function.

    def accuracy(self,x,y):
    x_pred = (x > self.thr).long() #if BCELoss expects sigmoid -> th 0.5, BCELossWithLogits expect real values -> th 0.0
    return (x_pred.float() == y).float().mean()

### Training

Row results of Loss and Accuracy for both Training and Validation.

    0 79114.9609375 23018.5625 0.7687227129936218 0.8313078284263611
    1 34343.3125 23018.5625 0.9112750291824341 0.8313078284263611
    2 22481.23828125 23018.5625 0.9555544853210449 0.8313078284263611
    3 18907.537109375 23018.5625 0.9658632874488831 0.8313078284263611
    4 17061.580078125 23018.5625 0.9707403779029846 0.8313078284263611

[ ... ]

    15 12815.693359375 3820.71337890625 0.9786549806594849 0.9762402176856995
    16 12821.419921875 3820.71337890625 0.9784191846847534 0.9762402176856995
    17 12690.97265625 3820.71337890625 0.9785776734352112 0.9762402176856995
    18 12475.1328125 3820.71337890625 0.9790005683898926 0.9762402176856995
    19 12495.986328125 3820.71337890625 0.9789242148399353 0.9762402176856995

**Learning Curves**

**Loss**

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#0493b14bf69041b599c72cceea18fa0e)

**Recall and Precision** 

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#d85672e5ac01459babeada02873fb282)

### Example of the output

**Test**

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#b5ccd567c4164d25af042518d13d5193](https://www.notion.so/014338557385464a9ede62b3dd04a29b#b5ccd567c4164d25af042518d13d5193)

**Train**

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#5bee853978ee49c0908523869e77e160](https://www.notion.so/014338557385464a9ede62b3dd04a29b#5bee853978ee49c0908523869e77e160)

**Predict #1 - Zeros**

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#289fbf166a484f12a0237453b09a8199](https://www.notion.so/014338557385464a9ede62b3dd04a29b#289fbf166a484f12a0237453b09a8199)

**Predict #2 - Rand**

[https://www.notion.so/014338557385464a9ede62b3dd04a29b#5f5c5d588c64425785f0ef7428f9953a](https://www.notion.so/014338557385464a9ede62b3dd04a29b#5f5c5d588c64425785f0ef7428f9953a)
### Conclusions of the Experiment
