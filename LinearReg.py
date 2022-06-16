import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

data = pd.read_csv("Product_Sales.csv")

train_imp = data.pop('PickedUp')

train_imp.head()

categorial_column = ['Agent_Name']
numerical_column = ['AgentID','CallID','CustomerID','Duration','ProductSold']

feature_columns = []
for feature_name in categorial_column:
  Numeric_list = data[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,Numeric_list))
for feature_name in numerical_column:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype = tf.float64))

print(feature_columns)

def input_fun(data_df,label_df,epochs=10,shuffle=True,batch_size=32):
  def inp_fun():
    data_set = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
      data_set = data_set.shuffle(1000)
      data_set = data_set.batch(batch_size).repeat(epochs)
    return data_set
  return inp_fun

training = input_fun(data,train_imp)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(training)

result = linear_est.evaluate(training)

print(result['accuracy'])

predictions = list(linear_est.predict(training))
print(data.loc[10])
print(train_imp.loc[10])
print(predictions[10]['probabilities'][1])

