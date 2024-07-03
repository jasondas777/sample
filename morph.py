import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

words = ['running', 'ran', 'jumps', 'jumping', 'eaten', 'ate']
# Porter Stemming
porter = PorterStemmer()
print("Porter Stemmer\n")
for word in words:
   stemmed_word = porter.stem(word)
   print(f"{word} -> {stemmed_word}")

# Lancaster Stemming
lancaster = LancasterStemmer()
print("\n")
print("Lancaster Stemmer\n")
for word in words:
   stemmed_word = lancaster.stem(word)
   print(f"{word} -> {stemmed_word}")
# Snowball Stemming
snowball = SnowballStemmer('english')
print("\n")
print("Snowball Stemmer\n")
for word in words:
   stemmed_word = snowball.stem(word)
   print(f"{word} -> {stemmed_word}")
# Lemmatization
lemmatizer = WordNetLemmatizer()
print("\n")
print("Lemmatization\n")
for word in words:
   pos_tag = nltk.pos_tag([word])[0][1][0].lower() # Get the part of speech tag
   lemmatized_word = lemmatizer.lemmatize(word, pos=pos_tag)
   print(f"{word} -> {lemmatized_word}")
