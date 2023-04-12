import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# Define the input data
texts = ['This is a positive sentence.', 'This is a negative sentence.']
labels = [1, 0]

# Tokenize the texts and prepare them for input to the model
inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Use the pre-trained BERT model to generate embeddings for the input texts
outputs = bert_model(input_ids, attention_mask=attention_mask)
embeddings = outputs[1]

# Define a simple classification model on top of the BERT embeddings
input_layer = tf.keras.Input(shape=(768,))
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model and train it on the input data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(embeddings, labels, epochs=5, batch_size=2)

# Use the trained model to make predictions on new texts
new_texts = ['This is another positive sentence.', 'This is another negative sentence.']
new_inputs = tokenizer(new_texts, return_tensors='tf', padding=True, truncation=True)
new_input_ids = new_inputs['input_ids']
new_attention_mask = new_inputs['attention_mask']

new_outputs = bert_model(new_input_ids, attention_mask=new_attention_mask)
new_embeddings = new_outputs[1]

predicted_labels = model.predict(new_embeddings)
predicted_classes = [1 if p >= 0.5 else 0 for p in predicted_labels]

print(predicted_classes)
