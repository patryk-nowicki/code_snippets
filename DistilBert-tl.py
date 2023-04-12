import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load data
df = pd.read_csv('data.csv')

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer and encode text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

# Convert labels to integers
label2id = {'negative': 0, 'positive': 1}
train_labels = [label2id[label] for label in train_df['label'].tolist()]
test_labels = [label2id[label] for label in test_df['label'].tolist()]

# Convert data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(len(train_df)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)

# Load pre-trained DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Freeze DistilBERT layers
for layer in model.layers[:-1]:
    layer.trainable = False

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Train model
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# Evaluate model on test set
loss, accuracy = model.evaluate(test_dataset)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
