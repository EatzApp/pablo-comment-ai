import nltk

# Ensure that you have the punkt tokenizer
nltk.download('punkt')

def process_reviews(input_file, output_file):
    # Open the input file and read its content
    with open(input_file, 'r', encoding='utf-8') as file:
        reviews = file.read().strip().split('\n')
    
    # Initialize a list to store sentences
    sentences = []
    
    # Process each review
    for review in reviews:
        # Split the review into sentences
        review_sentences = nltk.sent_tokenize(review)
        
        # Add each sentence to the sentences list
        sentences.extend(review_sentences)
    
    # Write the sentences to the output file, each on a new line
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

    print(f"Processed {len(reviews)} reviews and wrote {len(sentences)} sentences to {output_file}")

# Example usage:
input_file = '/Users/pablonieuwenhuys/EatzAIORIGINAL/eatzCommentAi/training_data_generator/reviews.txt'
output_file = '/Users/pablonieuwenhuys/EatzAI/SentencesList.txt'

process_reviews(input_file, output_file)
