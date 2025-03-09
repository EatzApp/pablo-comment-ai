import json
import nltk
from nltk.corpus import wordnet
import random
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, create_optimizer

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to replace words with synonyms
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if len(synonyms) > 0:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

# Step 1: Load the JSON file
with open('/Users/pablonieuwenhuys/EatzAI/latest.json', 'r') as f:
    data = json.load(f)

# Step 2: Augment the Data for Underrepresented Classes
augmented_sentences = []
augmented_labels = []

for item in data:
    text = item['data']['text']
    sentence_labels = item['annotations'][0]['result'][0]['value']['choices']
    
    # Assuming 'Ambiance' and 'Service' are underrepresented
    if 'Ambiance' in sentence_labels or 'Service' in sentence_labels:
        for _ in range(5):  # Generate 5 augmented samples per sentence
            new_sentence = synonym_replacement(text, n=2)
            augmented_sentences.append(new_sentence)
            augmented_labels.append(sentence_labels)

# Combine original and augmented data
for item in data:
    text = item['data']['text']
    sentence_labels = item['annotations'][0]['result'][0]['value']['choices']
    augmented_sentences.append(text)
    augmented_labels.append(sentence_labels)

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

# Step 7: Define and Compile the Model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=labels_encoded.shape[1])

for layer in model.layers[:-2]:
    layer.trainable = False

num_train_steps = len(train_dataset) * 7  # Assuming 7 epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=3e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=100
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Step 8: Train the Model
history = model.fit(
    train_dataset.batch(8),
    epochs=5,
    validation_data=val_dataset.batch(16),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
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

# Alternatively, you can print the detailed classification report again if needed
# It's not necessary to compute the classification report twice with the same data.


# Assuming labels_encoded is a numpy array with shape (num_samples, num_labels)
# and mlb is your MultiLabelBinarizer instance with the class names

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
