import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Input,LSTM,Dense
# from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import copy
import argparse

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
    print ("ERROR: pick_index_from_distribution, s=%f", s)
    return len(dist)-1

def print_steps_probabilistic(X, model, characters, length=32):
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

parser = argparse.ArgumentParser()
parser.add_argument("--activation", type=str, default="linear")
parser.add_argument("--dropout", type=float, default=.0)
parser.add_argument("--iterations", type=int, default=40)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lstm_units", type=int, default=80)
parser.add_argument("--keep_state", type=int, default=0)
args=parser.parse_args()
print(args)

if args.keep_state:
    x=Input(shape=(steps_num, len(characters)))
    y, state_h, state_c=LSTM(args.lstm_units, input_shape=(steps_num, len(characters)), return_sequences=True, return_state=True, dropout=args.dropout, activation=args.activation)(x)
    y=LSTM(args.lstm_units, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=args.dropout, activation=args.activation)(y, initial_state=(state_h,state_c))
    y=Dense(len(characters), activation="softmax")(y)
    model=Model(inputs=x, outputs=y)
else:
    model=Sequential()
    model.add(LSTM(args.lstm_units, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=args.dropout, activation=args.activation))
    model.add(LSTM(args.lstm_units, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=args.dropout, activation=args.activation))
    model.add(Dense(len(characters), activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam") # optimizer=SGD(momentum=.99, nesterov=True, decay=0.0001, lr=0.05)
model.summary()

losses=list()
validation_losses=list()
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))
for iteration in range(args.iterations):
    ((X,Y), (Xval, Yval), _)=get_XYc(iteration)
    hist=model.fit(X, Y, validation_data=(Xval, Yval), batch_size=20, epochs=args.epochs, verbose=0) # more epochs?
    print (hist.history)
    losses.extend(hist.history["loss"])
    validation_losses.extend(hist.history["val_loss"])
    print_steps_probabilistic(X, model, characters, length=120)

plt.plot(np.arange(len(losses)), np.array(losses), "g", label="training")
plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), "b", label="validation")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3))
plt.legend()
plt.savefig("plot_%s.png"%str(args))


# ["beegfs_server"],["beegfs_storages"],["cpu"],["disk"],["diskio"],["infiniband"],["kernel"],["mem"],["nstat"],["nvidia_gpu"],["nvidia_proc"],["processes"],["swap"],["system"],["uprocstat"]
# use:
# cpu: usage_user from cpu-total and the mean, std, and var for all other cpu's (as a measurement for good parallelism)
# kernel: derivative of context_switches and interrupts
# mem: cached, used
# processes: running
