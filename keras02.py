import numpy as np
from keras import Sequential
from keras.layers import LSTM,Dense
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt

def get_book(filename):
    with open(filename) as f:
        book=f.read()

    return book, sorted(list(set(book)))

def get_XY(book, characters, length=20, offset=0, step=1):
    def char_to_onehot(c):
        ret=np.zeros(len(characters), dtype=np.float32)
        ret[characters.index(c)]=1.;
        return ret

    x=list()
    y=list()
    book=list(map(char_to_onehot, book))
    for i in range(offset, len(book)-length-1, step):
        x.append(book[i:i+length])
        y.append(book[i+1:i+length+1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


book, characters=get_book("prince.txt")
print(len(book))
(X,Y), (Xval, Yval)=get_XY(book[:int(len(book)*.92)], characters), get_XY(book[int(len(book)*.92):], characters)
print(X.shape, Y.shape, Xval.shape, Yval.shape)

model=Sequential()
model.add(LSTM(80, input_shape=X.shape[1:], return_sequences=True, dropout=.0, activation="linear"))
model.add(Dense(len(characters), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer="adam")

hist=model.fit(X, Y, validation_data=(Xval, Yval), batch_size=20, epochs=10, verbose=0)
losses=hist.history["loss"]
val_losses=hist.history["val_loss"]
plt.plot(np.arange(len(losses)), losses, "g", label="training")
plt.plot(np.arange(len(val_losses)), val_losses, "b", label="validation")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,2))
plt.legend()
plt.savefig("le_plot.png")
print(losses)
print(val_losses)
print(np.mean(np.array(val_losses[-3:])))



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
    cinput=Input(shape=(80,))
    hinput=Input(shape=(80,))
    y,c,h=LSTM(80, input_shape=X.shape[1:], return_state=True, return_sequences=True, dropout=.0, activation="linear")(xinput, initial_state=[cinput, hinput])
    y=Dense(len(characters), activation="softmax")(y)
    model2=Model([xinput, cinput, hinput], [y,c,h])
    model2.compile(loss='categorical_crossentropy', optimizer="adam")
    model2.set_weights(model.get_weights())
    
    ret=list()
    x=np.zeros((1,1,len(characters)))
    x[0,0,np.random.randint(len(characters))]=1.
    c=np.zeros((1,80))
    h=np.zeros((1,80))
    
    for _ in range(100):
        [x,c,h]=model2.predict([x,c,h])
        #index=np.argmax(x[0,0,:])
        index=pick_index_from_distribution(x[0,0,:])
        ret.append(characters[index])
        x=np.zeros((1,1,len(characters)))
        x[0,0,index]=1.
    return "".join(ret)

print(generate_text(), "\n\n")
print(generate_text(), "\n\n")
print(generate_text(), "\n\n")
print(generate_text(), "\n\n")
print(generate_text(), "\n\n")