import json

import nltk
from nltk.corpus import wordnet

import re
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, create_optimizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random


# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def paraphrase(sentence, num_return_sequences=5, num_beams=5):
    # Preprocess the input sentence
    text = "paraphrase: " + sentence + " </s>"
    encoding = tokenizer.encode_plus(text, max_length=512, padding='max_length', return_tensors="pt")
    
    # Generate paraphrases
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_length=512,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    
    # Decode and return the paraphrased sentences
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
    return paraphrases

# Example usage
original_sentence = "The restaurant's aesthetic, too, has grown with time, now an Instagram-worthy ambiance from every corner."
paraphrased_sentences = paraphrase(original_sentence, num_return_sequences=5)

for i, sentence in enumerate(paraphrased_sentences):
    print(f"Paraphrase {i+1}: {sentence}")

# Step 1: Load the JSON file
with open('/Users/pablonieuwenhuys/EatzAI/latest.json', 'r') as f:
    data = json.load(f)


augmented_sentences = []
augmented_labels = []

for item in data:
    text = item.get('data', {}).get('text', "")
    annotations = item.get('annotations', [])
    if annotations and 'result' in annotations[0]:
        results = annotations[0]['result']
        if results and 'value' in results[0] and 'choices' in results[0]['value']:
            sentence_labels = results[0]['value']['choices']
        else:
            sentence_labels = []
    else:
        sentence_labels = []

    # Assuming 'Ambiance' and 'Service' are underrepresented
    if 'ambiance' in sentence_labels or 'service' in sentence_labels:
        # Generate paraphrases
        paraphrased_sentences = paraphrase(text, num_return_sequences=5)
        for paraphrased_sentence in paraphrased_sentences:
            augmented_sentences.append(paraphrased_sentence)
            augmented_labels.append(sentence_labels)

# Combine original and augmented data
for item in data:
    text = item.get('data', {}).get('text', "")
    annotations = item.get('annotations', [])
    if annotations and 'result' in annotations[0]:
        results = annotations[0]['result']
        if results and 'value' in results[0] and 'choices' in results[0]['value']:
            sentence_labels = results[0]['value']['choices']
        else:
            sentence_labels = []
    else:
        sentence_labels = []

    augmented_sentences.append(text)
    augmented_labels.append(sentence_labels)

# Check the number of sentences generated
num_sentences = len(augmented_sentences)
print(f"Total number of sentences generated (including original and augmented): {num_sentences}")

import json

# Combine sentences and labels into a list of dictionaries
augmented_data = [{'sentence': sentence, 'labels': label} for sentence, label in zip(augmented_sentences, augmented_labels)]

# Define the output JSON file path
output_json_file = 'augmented_sentences.json'

# Write to the JSON file
with open(output_json_file, 'w') as file:
    json.dump(augmented_data, file, indent=4)

print(f"Augmented sentences and labels saved to {output_json_file}")

# Step 3: Preprocess Text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

nltk.download('punkt')
sentences = [preprocess_text(sentence) for sentence in augmented_sentences]
labels = augmented_labels

# Step 4: Encode Labels
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)

for row in labels_encoded:
    print(row)

    # Step 5: Tokenize Sentences using DistilBERT Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(sentences, truncation=True, padding=True)

input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']

# Step 6: Split the Data into Training and Validation Sets
train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, labels_encoded, test_size=0.2, random_state=0
)

# Convert input_ids and attention_masks to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_input_ids,
        'attention_mask': train_attention_masks
    },
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': val_input_ids,
        'attention_mask': val_attention_masks
    },
    val_labels
))

# Convert the list of lists to a tuple of tuples
lst_of_tuples = [tuple(sublist) for sublist in labels_encoded]

# Use a dictionary to count the occurrences of each tuple
counter = {}
for sublist in lst_of_tuples:
    if sublist in counter:
        counter[sublist] += 1
    else:
        counter[sublist] = 1

# Print the results
for sublist, count in counter.items():
    print(f'The sublist {list(sublist)} appears {count} times.')

train_dataset = train_dataset.shuffle(buffer_size=1000)

# Step 7: Define and Compile the Model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=labels_encoded.shape[1])

num_train_steps = len(train_dataset) * 5  # Assuming 5 epochs
# Define the learning rate schedule with warmup steps
lr_schedule = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=3e-5,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=1.0,
    alpha=0.0,
    name=None
)

# Define the optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

tf.config.run_functions_eagerly(True)

# Step 8: Train the Model
history = model.fit(
    train_dataset.batch(32),
    epochs=10,
    validation_data=val_dataset.batch(16),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Step 9: Evaluate the Model
val_predictions = model.predict(val_dataset.batch(16))
val_probs = tf.nn.sigmoid(val_predictions.logits)
val_preds = (val_probs > 0.5).numpy()  # Adjust threshold as needed

# Print evaluation metrics (e.g., precision, recall, F1-score)
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=mlb.classes_, zero_division=0))

precision = precision_score(val_labels, val_preds, average='micro', zero_division=0)
recall = recall_score(val_labels, val_preds, average='micro', zero_division=0)
f1 = f1_score(val_labels, val_preds, average='micro', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Sum over the samples (axis=0) to count the number of occurrences for each label
label_counts = np.sum(labels_encoded, axis=0)

# Convert to a pandas DataFrame for easier plotting
label_counts_df = pd.DataFrame({
    'Label': mlb.classes_,
    'Count': label_counts
})

# Sort the DataFrame by count for better visualization
label_counts_df = label_counts_df.sort_values('Count', ascending=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(label_counts_df['Label'], label_counts_df['Count'], color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Number of Samples')
plt.title('Label Distribution in the Dataset')
plt.xticks(rotation=45)
plt.show()
print(label_counts_df['Count'], label_counts_df['Label'])

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(np.argmax(val_labels, axis=1), np.argmax(val_preds, axis=1))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()