import numpy as np
from keras import Sequential
from keras.layers import LSTM,Dense
from keras.models import Model
from keras.layers import Input
# import matplotlib.pyplot as plt
from keras.models import load_model
import time


def get_book(filename):
    with open(filename) as f:
        book=f.read()

    return book, sorted(list(set(book)))

# def get_XY(book, characters, length=80, offset=0, step=1):
#     def char_to_onehot(c):
#         ret=np.zeros(len(characters), dtype=np.float32)
#         ret[characters.index(c)]=1.;
#         return ret

#     x=list()
#     y=list()
#     book=list(map(char_to_onehot, book))
#     for i in range(offset, len(book)-length-1, step):
#         x.append(book[i:i+length])
#         y.append(book[i+1:i+length+1])

#     return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def get_XY(book, characters, length=80, offset=0, step=1, max_elements=None):
    if max_elements is not None:
        X=np.zeros([min((len(book)-length-1-offset)//step+1, max_elements),length,len(characters)], dtype=np.float32)
        Y=np.zeros([min((len(book)-length-1-offset)//step+1, max_elements),length,len(characters)], dtype=np.float32)
    else:
        X=np.zeros([(len(book)-length-1-offset)//step+1,length,len(characters)], dtype=np.float32)
        Y=np.zeros([(len(book)-length-1-offset)//step+1,length,len(characters)], dtype=np.float32)

    last_i=0
    for i,ibook in enumerate(range(offset, len(book)-length-1, step)):
        if max_elements is not None and i>=max_elements:
            break
        for j in range(length):
            X[i,j,characters.index(book[ibook+j])]=1.
            Y[i,j,characters.index(book[ibook+j+1])]=1.
        last_i=i
    assert(last_i==X.shape[0]-1)

    return X,Y


def generate_text(length=1000):
    def pick_index_from_distribution(dist):
        r=np.random.random()
        s=0
        for i,p in enumerate(dist):
            s+=p
            if s>r:
                return i
        print("ERROR: pick_index_from_distribution, s=", s)
        return len(dist)-1

    xinput=Input(shape=(1,len(characters)))
    cinput1=Input(shape=(lstm_size,))
    hinput1=Input(shape=(lstm_size,))
    cinput2=Input(shape=(lstm_size,))
    hinput2=Input(shape=(lstm_size,))

    y,c1,h1=LSTM(lstm_size, return_state=True, return_sequences=True)(xinput, initial_state=[cinput1, hinput1])
    y,c2,h2=LSTM(lstm_size, return_state=True, return_sequences=True)(y, initial_state=[cinput2, hinput2])
    y=Dense(len(characters), activation="softmax")(y)
    model2=Model([xinput, cinput1, hinput1, cinput2, hinput2], [y,c1,h1,c2,h2])
    model2.compile(loss='categorical_crossentropy', optimizer="adam")
    model2.set_weights(model.get_weights())

    ret=list()
    x=np.zeros((1,1,len(characters)))
    x[0,0,np.random.randint(len(characters))]=1.
    c1=np.zeros((1,lstm_size))
    h1=np.zeros((1,lstm_size))
    c2=np.zeros((1,lstm_size))
    h2=np.zeros((1,lstm_size))

    for _ in range(length):
        [x,c1,h1,c2,h2]=model2.predict([x,c1,h1,c2,h2])
        #index=np.argmax(x[0,0,:])
        index=pick_index_from_distribution(x[0,0,:])
        ret.append(characters[index])
        x=np.zeros((1,1,len(characters)))
        x[0,0,index]=1.
    return "".join(ret)

model_filename="model.h5"
css_filename="prince.txt"
lstm_size=80
sequence_len=100
max_elements=200000
# Each element has [100 timesteps, 71 characters]*[4 bytes]*[2: once for X and once for Y] ~ 56kb
# To use 10gb there would have to be ~187k elements. Rounded up that gives 200k.
# Book variable should be about 5gb in size.

start=time.time()
book, characters=get_book(css_filename)
print("Book read in %.1f seconds"%(time.time()-start))
print("Book len:", len(book))
print("Book chars:", "".join(characters))

start=time.time()
X,Y=get_XY(book, characters, length=sequence_len, step=sequence_len, max_elements=max_elements) #, offset=np.random.randint(0,len(book)-max_elements*sequence_len)
print("X and Y calculated in in %.1f seconds"%(time.time()-start))
print("X shape:", X.shape)
print("Y shape:", Y.shape)
# del book

try:
    model=load_model(model_filename)
    print("Model loaded")
except:
    model=Sequential()
    model.add(LSTM(lstm_size, input_shape=(sequence_len, len(characters)), return_sequences=True))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(Dense(len(characters), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    print("Model created")

hist=model.fit(X, Y, batch_size=40, epochs=1, verbose=1)
model.save(model_filename)
print("[---]: ")
print(generate_text())
print("[---]: ")
print(generate_text())

