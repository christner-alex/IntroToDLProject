import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading Data")
data_file_1 = open('youtube_action_train_data_part1.pkl','rb')
train_data_1, train_labels_1 = pickle.load(data_file_1)
data_file_1.close()

data_file_2 = open('youtube_action_train_data_part2.pkl','rb')
train_data_2, train_labels_2 = pickle.load(data_file_2)
data_file_2.close()

print("Normalizing")
train_data = np.concatenate([train_data_1, train_data_2], axis=0).astype(np.float32) / 255.0
del train_data_1, train_data_2
train_labels = np.concatenate([train_labels_1, train_labels_2], axis=0)
del train_labels_1, train_labels_2

#%%

print("Constructing model")

seq_length = 30
num_classes = 11
num_units_cnn_fc_layer = 20
num_units_lstm = 30
num_units_out_fc = 40

initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(0.0001)

x = tf.placeholder(shape=[None, seq_length, 64, 64, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)

x_concat = tf.reshape(x,[-1, 64, 64, 3]) #-1 = batch_size*seq_length(30)

conv1 = tf.layers.conv2d(inputs=x_concat,filters=32,kernel_size=[5,5],padding='valid',activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=regularizer,name="conv1")
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=(2,2),padding='valid',name="pool1")
conv2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[5,5],padding='valid',activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=regularizer,name="conv2")
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=(2,2),padding='valid',name="pool2")
conv3 = tf.layers.conv2d(inputs=pool2,filters=32,kernel_size=[5,5],padding='valid',activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=regularizer,name="conv3")
pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=(2,2),padding='valid',name="pool3")
pool3_flat = tf.reshape(pool3, [-1, 4*4*32]) #-1 = batch_size*seq_length(30)
cnn_fc = tf.layers.dense(inputs=pool3_flat,units=num_units_cnn_fc_layer,activation=None,kernel_initializer=initializer,kernel_regularizer=regularizer,name="cnn_dense")
cnn_out = tf.reshape(cnn_fc,[-1, seq_length, num_units_cnn_fc_layer]) #-1 = batch_size

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units_lstm, name="lstm_cell")
h_val, _ = tf.nn.dynamic_rnn(lstm_cell, cnn_out, dtype=tf.float32)

h_val_flat = tf.reshape(h_val,[-1, seq_length*num_units_lstm]) #-1 = batch_size
output_fc_1 = tf.layers.dense(inputs=h_val_flat,units=num_units_out_fc,activation=None,kernel_initializer=initializer,kernel_regularizer=regularizer,name="output_dense_1")
output_fc_2 = tf.layers.dense(inputs=output_fc_1,units=num_classes,activation=None,kernel_initializer=initializer,kernel_regularizer=regularizer,name="output_dense_2")

sm = tf.nn.softmax(logits=output_fc_2)
predict_op = tf.argmax(sm,axis=1)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output_fc_2) + tf.losses.get_regularization_loss()

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss=loss)

# Add opts to the collection
tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', predict_op)

#%%

np.set_printoptions(precision=3)

#split the data into a train set and a validation set
def split_data(x_data, y_data, train_size):
    indexes = np.arange(len(x_data))
    np.random.shuffle(indexes)
    batch_x = np.array([x_data[i] for i in indexes])
    batch_y = np.array([y_data[i] for i in indexes])
    return batch_x[:train_size], batch_y[:train_size], batch_x[train_size:], batch_y[train_size:]

#get "num" random batches of size "size" from the data
def get_batches(x_data, y_data, size, num):
    inds = np.random.randint(0,len(x_data)-size,num)
    xout = np.array([x_data[i:i + size] for i in inds])
    yout = np.array([y_data[i:i + size] for i in inds])
    return xout, yout

#return the total accuracy, class accuracy, and the confusion matrix given the truth and prediction
def calc_accuracy(truth,predictions,num_classes):
    
    confusion_matrix = np.zeros([num_classes,num_classes])
    
    for t,p in zip(truth,predictions):
        confusion_matrix[t,p] += 1
        
    total_acc = 0 if np.sum(confusion_matrix)==0 else np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        
    row_sums = confusion_matrix.sum(axis=1)
    for c in range(num_classes):
        if row_sums[c] > 0:
            confusion_matrix[c] = confusion_matrix[c] / row_sums[c]
    
    class_acc = np.diag(confusion_matrix)
    
    return confusion_matrix, class_acc, total_acc

#make predictions from the data and current model, save them in the record and return the confusion matrix
def test_accuracy(sess,x_data,y_data,batch_size,batches_per_epoch,class_acc_record,total_acc_record,str1,str2):
    predictions = []
    x_batches, y_batches = get_batches(x_data,y_data,batch_size,batches_per_epoch)
    for xb,yb in zip(x_batches,y_batches):
        predictions.append(sess.run(predict_op, feed_dict={x:xb,y:yb}))
    confusion_matrix, class_acc, total_acc = calc_accuracy(np.concatenate(y_batches),np.concatenate(predictions),num_classes)
    class_acc_record.append(class_acc)
    total_acc_record.append(total_acc)
    print(str1 + str(total_acc))
    print(str2)
    print(confusion_matrix)
    return confusion_matrix
    
#calculate from the data and current model, save them in the record and return the confusion matrix
def test_loss(sess,x_data,y_data,batch_size,batches_per_epoch,loss_record,str1):
    losses = []
    x_batches, y_batches = get_batches(x_data,y_data,batch_size,batches_per_epoch)
    for xb,yb in zip(x_batches,y_batches):
        losses.append(sess.run(loss, feed_dict={x:xb,y:yb}))
    avg_loss = np.average(np.array(losses))
    loss_record.append(avg_loss)
    print(str1 + str(avg_loss))
    
#%%

print("Splitting data")
tx, ty, vx, vy = split_data(train_data,train_labels,5000)
del train_data, train_labels

#%%

train_class_acc_record = []
train_total_acc_record = []
valid_class_acc_record = []
valid_total_acc_record = []

train_loss_record = []
valid_loss_record = []

train_acc_change_record = []
valid_acc_change_record = []
train_loss_change_record = []
valid_loss_change_record = []

final_train_confusion_matrix = 0
final_valid_confusion_matrix = 0

#%%

max_epoch = 30
batch_size = 10
batches_per_epoch_train = 400
batches_per_epoch_test = 100

counter = 0
counter_target = 25
target_acc_change = 0
target_loss_change = 0
target_accuracy = 0.9

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    epoch = 0
    while True:
        epoch += 1
        
        print("------------------------------")
        print("Epoch " + str(epoch))
        
        final_train_confusion_matrix = test_accuracy(sess,tx,ty,batch_size,batches_per_epoch_test,train_class_acc_record,train_total_acc_record,"Total Train Accuracy: ","Train Confusion Matrix:")
        
        final_valid_confusion_matrix = test_accuracy(sess,vx,vy,batch_size,batches_per_epoch_test,valid_class_acc_record,valid_total_acc_record,"Total Validation Accuracy: ","Validation Confusion Matrix:")
        
        test_loss(sess,tx,ty,batch_size,batches_per_epoch_test,train_loss_record,"Train Loss: ")
        
        test_loss(sess,vx,vy,batch_size,batches_per_epoch_test,valid_loss_record,"Validation Loss: ")
        
        if epoch > 1:
            train_acc_change = train_total_acc_record[-1] - train_total_acc_record[-2]
            valid_acc_change = valid_total_acc_record[-1] - valid_total_acc_record[-2]
            train_loss_change = train_loss_record[-1] - train_loss_record[-2]
            valid_loss_change = valid_loss_record[-1] - valid_loss_record[-2]
            print("Train Accuracy Change: " + str(train_acc_change))
            print("Validation Accuracy Change: " + str(valid_acc_change))
            print("Train Loss Change: " + str(train_loss_change))
            print("Validation Loss Change: " + str(valid_loss_change))
            train_acc_change_record.append(train_acc_change)
            valid_acc_change_record.append(valid_acc_change)
            train_loss_change_record.append(train_loss_change)
            valid_loss_change_record.append(valid_loss_change)
            counter += train_acc_change <= target_acc_change
            counter += valid_acc_change <= target_acc_change
            counter += train_loss_change >= target_loss_change
            counter += valid_loss_change >= target_loss_change
            
        print("Counter: " + str(counter))    
        if counter >= counter_target or epoch >= max_epoch or valid_total_acc_record[-1] >= target_accuracy:
            break
        
        x_batches, y_batches = get_batches(tx,ty,batch_size,batches_per_epoch_train)
        for xb,yb in zip(x_batches,y_batches):
            sess.run(train_step, feed_dict={x:xb,y:yb})
            
    save_path = saver.save(sess, "./my_model")

#%%
    
train_loss = np.array(train_loss_record)
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Total Training Loss')
plt.show()

valid_loss = np.array(valid_loss_record)
plt.plot(valid_loss)
plt.xlabel('Epoch')
plt.ylabel('Total Validation Loss')
plt.show()
    
train_acc = np.array(train_total_acc_record)
plt.plot(train_acc)
plt.xlabel('Epoch')
plt.ylabel('Total Training Accuracy')
plt.show()

train_acc = np.array(train_class_acc_record)
for j in range(num_classes):
    plt.plot(train_acc[:,j])
plt.xlabel('Epoch')
plt.ylabel('Class Training Accuracy')
plt.legend(['b_shooting', 'cycling', 'diving', 'g_swinging', 'hb_riding', 's_juggling', 'swinging,', 't_swinging', 't_jumping', 'v_spiking', 'd_walking'], loc='upper left')
plt.show()

validation_loss = np.array(valid_total_acc_record)
plt.plot(validation_loss)
plt.xlabel('Epoch')
plt.ylabel('Total Validation Accuracy')
plt.show()

valid_acc = np.array(train_class_acc_record)
for j in range(num_classes):
    plt.plot(valid_acc[:,j])
plt.xlabel('Epoch')
plt.ylabel('Class Validation Accuracy')
plt.legend(['b_shooting', 'cycling', 'diving', 'g_swinging', 'hb_riding', 's_juggling', 'swinging,', 't_swinging', 't_jumping', 'v_spiking', 'd_walking'], loc='upper left')
plt.show()

#%%

f = open("results.txt","w+")
f.write("Final Total Training Accuracy: "+str(train_total_acc_record[-1])+"\n")
f.write("Final Validation Training Accuracy: "+str(valid_total_acc_record[-1])+"\n")
f.write("Final Training Loss: "+str(train_loss_record[-1])+"\n")
f.write("Final Validation Loss: "+str(valid_loss_record[-1])+"\n\n")
f.write("Final Training Confusion Matrix:\n")
f.write(str(final_train_confusion_matrix))
f.write("\n\n")
f.write("Final Validation Confusion Matrix:\n")
f.write(str(final_valid_confusion_matrix))
f.close()