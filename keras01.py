import numpy as np
from keras import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import copy
import sys


def print_steps_max(X, model, characters, length=32):
    X=copy.deepcopy(X[0:1])
    steps_num=X.shape[1]
    chars=list(map(lambda x: characters[np.argmax(x)], X[0]))
    chars.extend(["[", "-", "-", "]"])
    for i in range(length):
        pred=model.predict(X)[0][-1]
        new_char=characters[np.argmax(pred)]
        chars.append(new_char)
        X[0]=np.concatenate((X[0][1:], np.zeros((1, len(characters)), dtype=np.float32)), axis=0)
        X[0][-1][characters.index(new_char)]=1
    print ("".join(chars))

def pick_index_from_distribution(dist):
    r=np.random.random()
    s=0
    for i,p in enumerate(dist):
        s+=p
        if s>r:
            return i
    return len(dist)-1

def print_steps_probabilistic(X):
    X=copy.deepcopy(X[0:1])
    steps_num=X.shape[1]
    chars=list(map(lambda x: characters[np.argmax(x)], X[0]))
    chars.extend(["[", "-", "-", "]"])
    for i in range(length):
        pred=model.predict(X)[0][-1]
        new_char=characters[pick_index_from_distribution(pred)]
        chars.append(new_char)
        X[0]=np.concatenate((X[0][1:], np.zeros((1, len(characters)), dtype=np.float32)), axis=0)
        X[0][-1][characters.index(new_char)]=1
    print ("".join(chars))

def get_bookXYchars(filename="prince.txt", training_proportion=.9, offset=0, steps_num=100, characters=None):
    assert(training_proportion>0. and training_proportion<=1.0)
    with open(filename) as f:
        f.seek(offset)
        book=f.read()

    if characters is None:
        characters=sorted(list(set(book)))
        characters="".join(characters)

    def get_xy(book):
        len_book=len(book)
        book=book+" "
        x=np.zeros((len_book//steps_num, steps_num, len(characters)), dtype=np.float32)
        y=np.zeros((len_book//steps_num, steps_num, len(characters)), dtype=np.float32)
        for pos in range(0, len_book//steps_num):
            for step in range(steps_num):
                x[pos, step, characters.index(book[pos*(steps_num)+step])]=1
                y[pos, step, characters.index(book[pos*(steps_num)+step+1])]=1
        return (x,y)
    return get_xy(book[:int(len(book)*training_proportion)]), get_xy(book[int(len(book)*training_proportion):]), characters

filename="prince.txt"
steps_num=40
with open(filename) as f:
    book=f.read()
characters=sorted(list(set(book)))
characters="".join(characters)

get_XYc=lambda offset: get_bookXYchars(filename="prince.txt", training_proportion=.85, steps_num=steps_num, offset=offset, characters=characters)
# ((X,Y), (Xval, Yval), _)=get_XYc(0)
# print (X.shape, Y.shape, Xval.shape, Yval.shape)
# print (np.prod(X.shape)*(32/8)/1024/1024, "MB")
# print_steps_max(X[0], characters)


activation="tanh" if len(sys.argv)<2 else sys.argv[1]

model=Sequential()
model.add(LSTM(80, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=.2, activation=activation)) # maybe use elu? and dropout?
model.add(LSTM(80, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=.2, activation=activation)) # maybe use elu? and dropout?
model.add(Dense(len(characters), activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=.99, nesterov=True, decay=0.0001, lr=0.05))
model.summary()

losses=list()
validation_losses=list()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))
for iteration in range(40):
    ((X,Y), (Xval, Yval), _)=get_XYc(iteration)
    hist=model.fit(X, Y, validation_data=(Xval, Yval), batch_size=20, epochs=2, verbose=1) # more epochs?
    print (hist.history)
    losses.extend(hist.history["loss"])
    validation_losses.extend(hist.history["val_loss"])
    print_steps_max(X, model, characters)

plt.plot(np.arange(len(losses)), np.array(losses), "g", label="training")
plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), "b", label="validation")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3))
plt.legend()
plt.savefig("plot_%s.png"%activation)