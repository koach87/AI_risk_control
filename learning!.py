import numpy as np
import tensorflow as tf
import io_module
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def show_loss_acc(history):
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper left') 
    plt.show()

def output_test_y(model):
    xtest = io_module.get_test_data()
    # print(xtest)
    y = model.predict(xtest)
    ans = []
    for i in range(len(xtest)):
        # if(i==3):
            # break
        ans.append(['Y','N'][np.bool(y[i][0]>y[i][1])])
        
    with open('ans.csv','w',encoding="utf-8") as csvfile:
        for i in ans:
            csvfile.write(i+'\n')
    
def current_rate(model):
    reg = model.predict(X_test)
    ans_YN = []
    ans_10 = []
    
    for i in range(len(reg)):
        ans_YN.append(['Y','N'][np.bool(reg[i][0]>reg[i][1])])
        ans_10.append([1,0][np.bool(reg[i][0]>reg[i][1])])
        
    cnt = 0
    for i in range(len(ans_10)) :
        if(ans_10[i]!=y_test[i]):
            cnt += 1
    return 1 - cnt/len(ans_10)

    
    
xtrain,ytrain = io_module.get_train_data()
xtrain, X_test, ytrain, y_test = train_test_split( xtrain, ytrain, test_size=0.33, random_state=42)
# print(xtrain,ytrain)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[len(xtrain[0])]),
    tf.keras.layers.Dense(64, activation='linear'),
    tf.keras.layers.Dense(64, activation='linear'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

#optimizer
# https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
opt = [
    tf.keras.optimizers.RMSprop(learning_rate=0.00001),
    tf.keras.optimizers.Adam(learning_rate=0.001),
    ]

los = [
    'sparse_categorical_crossentropy'
    ]


model.compile(optimizer = opt[1], loss=los[0], metrics=['accuracy'])
model.summary()

history = model.fit(
    xtrain,
    ytrain,
    epochs = 25,
    # batch_size = 512, 
    # steps_per_epoch = 30
    )

print("Correct rate : " + str(current_rate(model)))
show_loss_acc(history)

output_test_y(model)








