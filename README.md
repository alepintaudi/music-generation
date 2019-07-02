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
### Motivation
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
### Motivation

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
![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9e45504e-488c-420a-8134-8cab0aadf572/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45NTMLRCPQ%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184525Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHoaCXVzLXdlc3QtMiJHMEUCIQC8M2yinxkvjwOPuHnGJNKzHLR4j4TmoeAzALxxaME8%2BAIgD%2BX2rGcJJ3PKGutnkj8i2ejsPY8j%2BqIhjNeKMRn0qQwq4wMIw%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwyNzQ1NjcxNDkzNzAiDBf70RRxMHrXxT9nIiq3A%2FvfxrAlDfnXqeFLXcCdw3jYrOoA7OorreuxhK0ZbjGKG2Xxcp5meeG8H8kNfucVOIUWb5VgXy4WEDqeLow8EPYCqqi74btB%2BQc6yl5byVro%2By17NNIuf1KvHxBLkKPx2xKDQYAmcaccx1hI461QQbqMp3zK29BEicrfyq4scZUJ3bbe92Mt5ZQ%2BnH2n1wbFtOicP%2BOZS4WoGi2osadgXEtCtSAGECBE4ebEUWNZwOa3QFVekTCLa8VYkvGtR%2F%2Ffq9KLfsC6KfYOuTmNExxl9sXsJX10YPA9Pt9QcrqPtAQT8odH9BTxEZu1AoKg2p5agYLX0D9lkzWVkyKVEXS1cYYidoQhGReSyaVrxda2ojzJhI8sIdOtClAZkF1uUMJ0hrTvVB%2Bdle1188sTLBkxVn0qYdh2BU8mRv8ZpXL7ZqK1ftVq4TIIGCsdkaYbT3gu5%2BmeYGksEleIJuyf4d4C0kEI2KlPcpnUp5t2Jt87RTCwhKUc%2FXaq125qkXAVBcsO0sMC%2BNyNkAbeGb2zsrmKNsy%2FYwbCG0FOrM5STi1issoaIcc7wnqUWLSDaRk73r3YNPsiexEyuoYwpr7u6AU6tAEftCq2K6YSwleG%2BC98nRmWHUDf%2BkkFabTIIGH99pgPIFPfWiY%2BbtDVYmuGcSWJlzMwVGq2EiMDAeduxHh44c%2FebPlvZfnB45LoubYnmqbY%2BgdVUIGuAmNO7W52t6aNyS150ozwz8klWOlgefwQXVObkXHRdtJggi6JOQVs6qEh2nRV5nGQP522JOpvfxF0VUgimxYv2XCq2pGxw9KItAfvVK5kY4GorY%2FFqHOPnMvRifOzQhE%3D&X-Amz-Signature=caeaaaeeda7d104841db291612f079f46acbf2fe41b19375be56b1b03952275b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#ba03ced3160f4c0d94be56340b840fcf)

***Accuracy***

A similar behaviour can be observed for the learning curves of accuracy, where from an initiation convergence the two curves end up going far from each other.
![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/43539d11-57f5-493d-a6ec-e8ae91802ebe/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45CA3SAACD%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184543Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHUaCXVzLXdlc3QtMiJHMEUCIFbBuYPvTtaXy%2BAQWCA47NJ86mOZjQX9weuS5qL92NcrAiEA8Sw8razTUBf4zmlQiFybkScsmxDjbStCJEyx0msuWvMq4wMIvv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwyNzQ1NjcxNDkzNzAiDBYOH59%2F2yDUlYm4gCq3A5WDDzXwPuqank8h5XMoWbPTZfgnkKe%2FkSneKlEZ0TKpzI%2F6f%2FJ7nYkFNqc4Racovec8DZ4726ObtSEPjhcs7sxT6K23ME1wKASkbBcc%2Fry3ALxeL9O1J8gKPrD2EJgDUhbUHWmc2hMFZUhXja6O%2Ft73e5LXeZH0bzz%2BtPdS%2FT%2Fi2Zy8vjjv2IqF5KQ7hT8iY4XANmSyPKGvqjsRRW62qXqvx5ZumFKRDzDt3jbHWCuxSMZzoagad%2F%2FF0L7qZAMNRUWopOGZ5TGxH57FlRp7kg%2BPbAOc2KynWv3od7M5HDnh9KK88ujzaITiClDeaSxzuAaFJ8YyCjEb7Tdfqko%2FMRLhfGGVVLV9GumBtc0oCvLo9pFAd4SWa2YP9Ob4vMic%2B%2FkWMYbDmu%2FBaGLUu4TBi7sTFNe%2BbU5mdx0Fm%2Fc%2BsxEgVGxW3ZqsWBxo1F4z9tD4DKxo6usks7YrFSbG8reSdThFCVdvQsw6wy2EGWBdbUyuksjocMmwNb5iJJcs3Mh%2FseDP8F2T0MyiRFopAQHxfkUCnwZVq5TY%2BaVI4TZbNof9UVnR%2B%2FCzqKpPHC7uHxEwipi6SuFaJy4wtKnt6AU6tAGF6CR86Fxj7SB71S8srTNFhnmD9o%2BNhs%2FrImq9DOCSFtV8Eu2YwLPrHJmx0No70cVlxpjTCQiUZ0hmEN3xIUrVAmYXGQ%2BfBWm0NcMZu1s028NgRADx2u7N7SS%2BF3nrjho%2BOaMtGzok%2FdoDGNEomCzYvEl8wLHPetrJFRmhvnarX4xd0nEI3tWU2Lai0BEwSGI4Wd2iI4ryjp%2BZa1SNRi8vy6910MY6G0SMhO5Lzq9WbkUkk1s%3D&X-Amz-Signature=aa17a6157e7b44f63757b4a66ce236aa33626ab8f6e6514aa11f90e6a6c13175&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

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

### Motivation

In this third experiment we tried to solve the issues found during the second experiment. The biggest problem found previously is the monotony of the generated output sequence (same note repeated over and over), we decided to try to solve it moving towards a more standard Sequence to Sequence network.

Here the decoder was finally generating a sequence of the same length as the one sent to the encoder.

### The Model

**Encoder**

The encoder is made of 2 LSTM layers having 512 neurons each with a dropout layer in between them. 

We then have a decoder with a first fully connected layer that reduces dimensionality to 256, and then an activation function and dropout before the final fully connected layer that reduces the dimensionality to the input size of the piano roll.

Cross-Entropy Loss function is finally applied to the results.

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
![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/43ce74a2-e17d-49ab-aa06-b0915bc2cdab/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45IXQGA3UZ%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184834Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHUaCXVzLXdlc3QtMiJHMEUCIH0Uc0LYTRFaAGpfO8fNBEd%2B92j1x6U%2BEBH3Vxmb8sEsAiEAs5zy13KLgNbfOFceR2o%2BCfLbaTsbzCtwBzg60VjYO1Mq4wMIvv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwyNzQ1NjcxNDkzNzAiDLaHNHfkb9nFt4MhLCq3AxN8es4Bx1L6UNW6rMJhHWZZXgpfFqDOTRbLq5Dv6FkQm5F30d3B98wyVnHgiOehA4KVjSGOtvQqa0Yi5VcFKLKugbtY2B0S%2FO8fvA9IYR2woqTVkir0UWbrnSUWRA%2BZSBtoXb8Ko2lNsHCGpCLazq1x6Vglc6xNbtzZ%2FiXMnZ7gYD4eBMioAD%2BpabECWpqidtPorxbg24aT1y%2BKgfqXmJg2M%2F99qrfW2tAWouY3cFVY0pARHgloQnKiedva1pX83ukzt7rzgqdJv6Cn72G9sFOPOvd2jZhEGOs5UrHkSQInNsjcPpaBrMk3x%2F8U9B0WswkXOh3HVT9D1%2F%2FZT3xF8MiN0YbQ4AHO%2B2%2Fy6NhEMAodWtXAUx55LKLQwCxSVKwMvWOv3%2BECFSPe0VSgH4J4W4W4UPuzAzpOFSMWC%2Br1MJBrNAMq0wYKJm7bZElNysmWWAfq9FMauIUyFvK8MmUPu8Y6AQvuR3%2Fr9KUaPwKtewgjxPUW3a6VgH%2BhrwU5EK8ab9yGnWlvmhlyyTo8YuINSzZSbddhit0oLHwQHkj0WceQFdEdR8nCAdwMDSRvUr2NhPNUz5B9QLcww6Tt6AU6tAE0PwxJ3t3sunzOFC5dP9nefaF%2FseTfOD4YeVmXFil1PwhG8NZhsWRAPn%2FJhpvcv0jjc%2FB6TErfg9roLfu6KwShiK22LR9WqltnI1IlOYn8blojG%2BIbtx7a8CyQxD4vLN4AcGOVP9le%2BcerW5aquIA%2BQq7onzaV3CBi%2B5XBOYmQoijtJwFfW4u4h%2BU1Qmb6G0Gmx7X39H8%2FEh1LMRjrP3fOAeIq%2F1%2BeuDr5mYLdhuckisR9%2F64%3D&X-Amz-Signature=44ec4a7034cab628a11790c27a3b151460acf794752ca0083a436db51f9b78d5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#05499da4e4774d2694bb9d911caff89f)

***Accuracy***

The two curves are starting close to each other, to then separated and eventually end up both above the 95% of accuracy.

![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5f77bbf0-f2bd-45ae-9c28-377248cb59b3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45N4LV7CPU%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184837Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHUaCXVzLXdlc3QtMiJIMEYCIQDAFAu1hSHYa%2F6t2vJt0aI%2BBPW%2FEKg4KcZGwTm66btVZAIhAMLMHtsDUGH%2BGVRa6AbPELOTPmWiptW7pg6oGc%2B8Ze8dKuMDCL7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMMjc0NTY3MTQ5MzcwIgwsS4%2Fc%2F%2FdXa4%2FEVBwqtwNQxC93cJhS3vJsKel2p79GMP9JDsra3Dfx4O30mhp6NtrbZZNmfdaEv2oGYztp6Bdptej7mpVDtAxehZlrNiawYCnrwtojVVcRqYSE4fJtiWIhLnyIpPgYdWcB%2FfIa6oXKHEIQLSv7zlgMWQzSyOoTvLEZjwOlnExPFP5uGmavuLStBYR5pTPEOMFbrlAszIajYy%2BnshM9YQ9T%2Bn0sxW3P2iKOEd8ajt55SsQAx67Dg%2FqvdKh%2B8o8dRYqFYS1gHf6wTb%2F7h3THShb2UeoaMYGgkVZXkeaVtzQuT4X8M54zfxEyVS%2FNDZOPvJ791jKl7%2FJq06SsgF5z6Mrn9QwlBue5bSJGcXmrOKRC5S3siGJghUVIec5%2Bw3XDcaZ59f1jX4R%2FTr8LDbMQrLkdDhCCCZC%2FYb1R%2FuobXJwNMrVuebsF2oXiZgTU7s3uEyGVm8WAJXBz09wqTerRcvg%2FAvSCxlBKkEidFLAhcoIBfm7PmWIZiqFp20ZXOJVbL4XbMUJE589UPsQ7Zu1OVGYwPvpg6W9q7D73PnCumardvuDFsdnu2b%2BSFpKTse3Bc367sa0JX%2F1cOcB8QFcHMLWc7egFOrMB96YGJvvuMjbNvH%2FoaUtaTMrHy2kOgotI8Km%2FhvVtQgPhvqu5t009EElK3pGwYlvQUCo8uqpcvMbHBWX0Ym%2BAHKCdjbX46%2FkqLKZW2%2B%2FiCWT%2Ff5r6uyxj0VioPoQlok9f4GMExg0f%2FNalR5nsd0NhH9YfbLfuqyjPp3ROIbV6WWt1KIc4IdDrfR%2BpVdJ4vUmHBqFZArJiXIgzmjj%2FLYChW2KzYNCyFZJkuqQIW5gSCQaI2B4%3D&X-Amz-Signature=4e876ea58b305780ba1e871b7600c1ec2e69486838c6a9e4f0be81eb9a2d9b0c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

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

### Motivation

In this fourth experiment we tried to improve the previous model, starting by solving its main problem of generating only monophonic music and providing the right architecture to generate polyphonic music at the output.

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

![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c410a8dc-ed31-44b5-879b-096f8044d105/Loss50.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45GFOPL7N3%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184937Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHUaCXVzLXdlc3QtMiJGMEQCIFPZvWZ3AwJkS0kDy2SXEJSPqRktCfzYAscyMwKGA%2B6GAiBtmlb32DhCSNF0ms%2FzQCNe6eTuUnx418iOGl0rIRsoqyrjAwi%2B%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDI3NDU2NzE0OTM3MCIMjbktq%2B4tvgcHGFyEKrcDzGR5Ii2lo1xeoFZO8vQeV4sahAGpKr5Oa1Ipdk%2F271lCm8%2BIDbmOtjx6ozzdofkzsVZ9JgxrsJFGLzFRAoOlFFl54wZ%2B3ISBMSst%2F%2Bws8KGgy0CXPG7k15zGKAOxp%2Fl7BQa3yzFXpC0eT%2B94gxvPNcUKy%2BoiREInNPhqb%2FNsIGTypwcBYHR6D%2BFKaXYHL9pQdFlugOnM%2FQtlbcA9o9qKfmftTEQgMKQHqvvxC%2FI4IQDeFt%2B8LZEmNt0eT1Lisdtju2P1bdqql4BrKF9p%2FF9F2f5MqMBCt1nfxaOU2cHLppsNBoKm01laIQay3Hgx7q0G6j9hfPURUpNpXI0neBVwlW%2BDgSm8lQ4ltXlLXgeCf5apFyp4s%2B%2FzGmJ7d9YJ13DPIUXuk9VxRZV4Ej40wlZgRMRuscaAFOqwhErjxtYVzu%2BPC19MKsPSoXXI1%2By3SxR4Y5M%2FVAD4W384tBsCMB94YSB8i2lo7BUfki9ERNPA19jPLhYr72Y05r4JGfHzO2v0nl6eDqOQNJxvV4iU%2B0H0eaJivwDKVdgPBKO39dRXkdq8sGRPfAs9ICXNJBWr7lZzAXDrGUiOIDCkru3oBTq1Abf6XeT55psPoz4sOG6YR2VPCqSdriJQ8Z7UuaN6Tp8rEioSF6aUjdOtrEiteZo3wE9OeUofEfoq%2FgEU%2Fk4d7CQ%2FBy1%2BAjnhySSHqsvpfhbukiCncIWbafwsKNn8aYlv8AVV%2FSOeSoTjrXVwYiCAilYNgMWOIkCvvxAmExg7OolZV7aCqS0IFaK2gHLzyXuT%2BHbBpbHPGWZHpyqIVVYnSBfy%2FTEWZ4S1uaKHcNb%2Fu4oz%2B0G2%2BCA%3D&X-Amz-Signature=cf47ac3354d9a93533dde19be756426dec384780e02d650f5b258f95a9b6c34b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Loss50.png%22)

[](https://www.notion.so/014338557385464a9ede62b3dd04a29b#0493b14bf69041b599c72cceea18fa0e)

**Recall and Precision** 

![title](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e761a4c6-f562-4726-ab69-c73097884cd9/recall_precision.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45A4M34CHX%2F20190702%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190702T184941Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHUaCXVzLXdlc3QtMiJHMEUCIGw709CVeLhDqRMcVat2YO6MLyk9Yx93Aruv0as0i5y2AiEAxCYTvd49mu4wHKzjKIlZlr8MTWBZwxQd6lZo%2BolTEpIq4wMIvv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwyNzQ1NjcxNDkzNzAiDOPdSG4XOkN7rsbVbiq3AzjTsPvwq0v%2FcJ2g3Fg2ISPlOxlOlUsmc%2FsVcCway6Y5nQHigUwEc3XmfFcv8ucSPebcM1ao3AOKvR%2Bib79h7yEwBWX6CuUwNOWruhMKrdfIq6YDS4gwPsXpDpzEtguP%2By9ckjZSBQScnPjB%2F4%2BHXmq1n7%2B0b5eE1xcQH2VQkem977nB2WX9aIYVNEIPKHzU4im5xv5SmWiGrC72daBWVp5bOxjxX3Zh4OyQsiXbs3ajihm%2BfxlNhe2IZCJXFSa3WuvaWC5u9pWc3HsrbwCOxhGzRwSSUFRpXPkgbD2el%2Bp9sRmA1Jivn%2FNryOpbLzCCkfO%2FYVBzsXiDQJzPsEBpdohoXCbGaGavkz73%2FOZATBLygAOgpoXnP%2BPk6cmvK4bHFdq%2FEI5ZFzys24X81hdpPWkUd%2FuhEBUJIqMcqiwjnzWfxRASsbzKNnT2mFKbANF8uyiXfzRx0lCjxtfS8I6jTFo2wU4vAYt86i5Rq2sWICR%2FpEVjQra3uHobumI04uOmfzV7w%2F9lXeiG2%2BV9o5B8R6zOOhsvUV6cBHY3gw2fdAM7pes1NyTEb906OUEofngFbCbFr8Ty0Kcw%2Fqnt6AU6tAHYVw6NKTxw3jy8syBuvlNRR%2Fy4XrM3rnVZ66Shee2GhIE2pshdGITEMn4fU03JuwrhilbRsN42iGDmIxMz7us43zktOrH6LkF3BazcRk9ez6L7vHj2TsF9MQPC6ysbJ4GlzWhMlZs2bdtEP0X0gX0TM9S4uTHLu%2BKpj1kblIG59i1KksSOWr54MQ%2Bp%2FYNovxO67%2BqZGrIAJ1868cpckR%2FCtY%2F%2FOnkI4lJB5Slz5fucFmfub7U%3D&X-Amz-Signature=a46c27eceb6bf755e9184c6e3fb4eae65e75240d0c596775a939e1d727b28ff7&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22recall_precision.png%22)

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
