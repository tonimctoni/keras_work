# import numpy as np
# from keras import Sequential
# from keras.models import Model
# from keras.layers import LSTM,Dense,TimeDistributed,Input,Dropout
# from keras.optimizers import SGD
# from keras.callbacks import TensorBoard
# from keras import backend as K
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt



# def main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, dropout1, dropout2, image_filename, text_filename):
#     characters=" !',-.:;?abcdefghijklmnopqrstuvwxyz\n"
#     character_num=len(characters)

#     def get_bookXY(): #279219 #9280812
#         train_size=9000000
#         eval_size=250000
#         with open("asoiaf_sentences.txt") as f:
#             # f.seek(np.random.randint(0,steps_num))
#             book=f.read(train_size)
#             book_val=f.read(eval_size)

#         X=np.zeros((train_size-steps_num, steps_num, character_num), dtype=np.float32)
#         for offset in xrange(0, steps_num):
#             for pos in xrange(0, (train_size-steps_num)/steps_num):
#                 for step in xrange(steps_num):
#                     X[offset*((train_size-steps_num)/steps_num)+pos,step,characters.index(book[(offset+pos*steps_num)+step])]=1

#         X=np.zeros((train_size, character_num), dtype=np.float32)
#         for i,c in enumerate(book):
#             X[i,characters.index(c)]=1.

#         Xval=np.zeros((eval_size, character_num), dtype=np.float32)
#         for i,c in enumerate(book_val):
#             Xval[i,characters.index(c)]=1.

#         Y=np.concatenate((X[1:], X[0:1]))
#         Yval=np.concatenate((Xval[1:], Xval[0:1]))

#         data=(X.reshape(-1,steps_num, character_num), Y.reshape(-1,steps_num, character_num))
#         data_val=(Xval.reshape(-1,steps_num, character_num), Yval.reshape(-1,steps_num, character_num))

#         return data, data_val

#     def pick_index_from_distribution(dist):
#         r=np.random.random()
#         s=0
#         for i,p in enumerate(dist):
#             s+=p
#             if s>r:
#                 return i
#         return len(dist)-1

#     def generate_text(model, text_length=steps_num*2):
#         X=np.zeros((1, steps_num, character_num), np.float32)
#         X[0,0,characters.index("\n")]=1.
#         s=list()
#         for i in xrange(steps_num):
#             Y=model.predict(X)[0,i,:]
#             index=np.argmax(Y)
#             # index=pick_index_from_distribution(Y)
#             if i<steps_num-1: X[0,i+1,index]=1.
#             s.append(characters[index])
#         s1="".join(s)

#         X=np.zeros((1, steps_num, character_num), np.float32)
#         X[0,0,characters.index("\n")]=1.
#         s=list()
#         for i in xrange(text_length):
#             Y=model.predict(X)[0,i if i<steps_num else -1,:]
#             index=pick_index_from_distribution(Y)
#             if i<steps_num-1:
#                 X[0,i+1,index]=1.
#             else:
#                 X=np.concatenate((X[0:1,1:,:], np.zeros((1,1,character_num))), axis=1)
#                 X[0,-1,index]=1.
#             s.append(characters[index])
#         s2="".join(s)

#         if text_filename:
#             with open(text_filename, "a") as f:
#                 f.write("-"*80+"\n")
#                 f.write("[start]"+s1+"[end]\n")
#                 f.write(s2+"\n\n")


#     model=Sequential()
#     model.add(LSTM(hidden_size1, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout1, activation="elu"))
#     model.add(LSTM(hidden_size2, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout2, activation="elu"))
#     model.add(Dense(hidden_size2, activation="elu"))
#     model.add(Dropout(dropout2))

#     model.add(Dense(character_num, activation="softmax"))
#     # def make_model():
#     #     X=Input(shape=(steps_num, character_num))
#     #     Y1=LSTM(hidden_size1, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout1)(X)
#     #     Y1=Dropout(dropout2)(Y1)
#     #     Y2=LSTM(hidden_size1, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout1)(X)
#     #     Y2=Dropout(dropout2)(Y2)
#     #     Y3=LSTM(hidden_size1, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout1)(X)
#     #     Y3=Dropout(dropout2)(Y3)
#     #     Y4=LSTM(hidden_size1, input_shape=(steps_num, character_num), return_sequences=True, dropout=dropout1)(X)
#     #     Y4=Dropout(dropout2)(Y4)
#     #     # Y=TimeDistributed(Concatenate(input_shape=(steps_num, character_num)))([Y1,Y2])
#     #     # Y = TimeDistributed(Merge([Y1, Y2], mode='concat'))
#     #     # Y = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:])([Y1,Y2])
#     #     Y=Add()([Y1,Y2,Y3,Y4])
#     #     Y=Dense(character_num, activation="softmax")(Y)
#     #     return Model(X,Y)
#     # model=make_model()

#     optimizer=SGD(momentum=.99, nesterov=True, decay=decay, lr=lr)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#     model.summary()
#     print optimizer.get_config()

#     losses=list()
#     validation_losses=list()
#     for it in xrange(iterations_num):
#         print "Iteration:", it
#         ((X,Y), validation_data)=get_bookXY()
#         hist=model.fit(X, Y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose)
#         losses.extend(hist.history["loss"])
#         validation_losses.extend(hist.history["val_loss"])
#         generate_text(model)

#     generate_text(model, 2000)
#     if text_filename:
#         with open(text_filename, "a") as f:
#             f.write("losses: "+str(losses)+"\n")
#             f.write("validation_losses: "+str(validation_losses)+"\n\n")

#     if image_filename:
#         plt.plot(np.arange(len(losses)), np.array(losses), "g")
#         plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), "b")
#         x1,x2,y1,y2 = plt.axis()
#         plt.axis((x1,x2,0,3))
#         plt.savefig(image_filename)

# steps_num=100
# iterations_num=10
# epochs=1
# verbose=1

# batch_size=20
# hidden_size1=1024
# hidden_size2=1024
# decay=0.0001
# lr=0.05
# dropout1=0.2
# dropout2=0.2


# from sys import argv
# try:
#     # dropout1=float(argv[1])
#     # dropout2=float(argv[2])
#     steps_num=int(argv[1])
# except:
#     pass
# batch_size, hidden_size1, hidden_size2, decay, lr, 
# v=(dropout1, dropout2)
# v=steps_num

# image_filename="1024"+".png"
# text_filename="1024"+".txt"

# print (steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, dropout1, dropout2, image_filename, text_filename)
# main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, dropout1, dropout2, image_filename, text_filename)
# from threading import Thread

# todo_list=[
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0., 0., image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0.1, 0., image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0., 0.1, image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0.1, 0.1, image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0.2, 0., image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0., 0.2, image_filename, text_filename),
# lambda : main(steps_num, hidden_size1, hidden_size2, iterations_num, batch_size, epochs, verbose, decay, lr, 0.2, 0.2, image_filename, text_filename),
# ]


# threads=list()
# for todo in todo_list:
#     threads.append(Thread(target=todo))

# for thread in threads:
#     thread.join




import numpy as np
from keras import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import copy


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

def print_steps_probabilistic(X):
    pass

def get_bookXYchars(filename="prince.txt", training_proportion=.9, offset=0, steps_num=100, characters=None):
    assert(training_proportion>0. and training_proportion<=1.0)
    with open(filename) as f:
        f.seek(offset)
        book=f.read()

    if characters is None:
        characters=sorted(list(set(book)))
        characters="".join(characters)

    def get_xy(book):
        book=book+" "
        x=np.zeros((len(book)//steps_num, steps_num, len(characters)), dtype=np.float32)
        y=np.zeros((len(book)//steps_num, steps_num, len(characters)), dtype=np.float32)
        for pos in range(0, len(book)//steps_num):
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


model=Sequential()
model.add(LSTM(80, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=.0, activation="tanh")) # maybe use elu? and dropout?
model.add(LSTM(80, input_shape=(steps_num, len(characters)), return_sequences=True, dropout=.0, activation="tanh")) # maybe use elu? and dropout?
model.add(Dense(len(characters), activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=.99, nesterov=True, decay=0.0001, lr=0.05))
model.summary()

losses=list()
validation_losses=list()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))
for iteration in range(80):
    ((X,Y), (Xval, Yval), _)=get_XYc(iteration)
    hist=model.fit(X, Y, validation_data=(Xval, Yval), batch_size=20, epochs=2, verbose=1) # more epochs?
    losses.extend(hist.history["loss"])
    validation_losses.extend(hist.history["val_loss"])
    print_steps_max(X, model, characters)

plt.plot(np.arange(len(losses)), np.array(losses), "g", label="training")
plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), "b", label="validation")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,4))
plt.legend()
plt.savefig("plot.png")