# Credits: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
# importing libs
from nltk.tokenize import sent_tokenize, word_tokenize
import tensorflow as tf
import tensorflow as hub
import numpy as np
import networkx as nx
import nltk
import warnings
import tensorflow_hub as hub
import os
warnings.filterwarnings('ignore')

# downloading nltk english vocab
nltk.download('punkt')
# downloading USE 512 dim word emebedding
if os.path.exists(r"D:\B2B_Git_Instance\sentence_summarizer/use/512/") != True:
    print("downloading USE(512 dim) model")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    useLarge = os.path.join(r"D:\B2B_Git_Instance\sentence_summarizer", "use/512/")
    tf.saved_model.save(embed, useLarge)
else:
    print("Already exists")
    
class SentenceSummarizer:
    
    ''' The instance of the class will accept a sentence / text  and will provide a 5 point summary of the same.
        This is an extractive summarization. '''
    
    def __init__(self, sentence):
        
        ''' setence: paragraph '''
        
        self.sentence = sentence
        self.embed = tf.saved_model.load(export_dir=r"D:\B2B_Git_Instance\sentence_summarizer\use\512")

    def embedder(self):
        
        ''' this will perform the embedding using USE 512 dim vector '''
        
        sent = sent_tokenize(self.sentence)
        word_embed = []
        for i in sent:
            word_embed.append(self.embed([i]))
        return word_embed, sent

    def similarityMatrix(self,word_embed,sentence):
        
        ''' Calcuation of similarity matrix among sentences.
            word_embed: embedded vector from embedder()
            sentence: original sentence from embedder() '''
        
        self.word_embed = word_embed
        self.sentence = sentence
        similarity_matrix = []
        for i in range(0, len(self.word_embed)):
            temp = []
            for j in range(0, len(self.word_embed)):
                if self.sentence[i] != self.sentence[j]:
                    temp.append(
                        np.absolute(tf.keras.losses.cosine_similarity(self.word_embed[i], self.word_embed[j]).numpy())[
                            0])
                else:
                    temp.append(0)
            similarity_matrix.append(temp)
        similarity_matrix = np.matrix(similarity_matrix)
        return similarity_matrix

    def sentRank(self,similarity_matrix,sentence):
        
        ''' Rank the most similar sentence first using page rank algorithm.
            similarity_matrix: outpur from similarityMatrix() '''
        
        self.similarity_matrix = similarity_matrix
        self.sentence = sentence
        sentence_similarity_graph = nx.from_numpy_array(self.similarity_matrix)
        scores = nx.pagerank(sentence_similarity_graph)
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(self.sentence)), reverse=True)
        return ranked_sentence

    def sentSummarize(self,ranked_sentence):
        
        ''' ranked_sentence: output from sentRank() '''
        
        self.ranked_sentence = ranked_sentence
        lines = 5
#         lines = int(
#             input("How many line summary you want? Has to be less than the number of sentence in the original para"))
        summarize_text = []
        for i in range(lines):
            summarize_text.append(self.ranked_sentence[i][1])
        return ''.join(summarize_text)
    
a = SentenceSummarizer(para)
vec, sent = a.embedder()
sim_mat = a.similarityMatrix(word_embed=vec, sentence=sent)
rank = a.sentRank(similarity_matrix=sim_mat, sentence=sent)
a.sentSummarize(rank)

# para = ['''Gradient Descent is an optimisation algorithm that can be used in a wide variety of problems. The general idea of this method is to iteratively tweak the parameters of a model in order to reach the set of parameter values that minimises the error that such model makes in its predictions.
# After the parameters of the model have been initialised randomly, each iteration of gradient descent goes as follows: with the given values of such parameters, we use the model to make a prediction for every instance of the training data, and compare that prediction to the actual target value.
# Once we have computed this aggregated error (known as cost function), we measure the local gradient of this error with respect to the model parameters, and update these parameters by pushing them in the direction of descending gradient, thus making the cost function decrease.
# The following figure shows graphically how this is done: we start at the orange point, which is the initial random value of the model parameters. After one iteration of gradient descent, we move to the blue point which is directly right and down from the initial orange point: we have gone in the direction of descending gradient.''']
# a = SentenceSummarizer(para[0])
# vec, sent = a.embedder()
# sim_mat = a.similarityMatrix(word_embed=vec, sentence=sent)
# rank = a.sentRank(similarity_matrix=sim_mat, sentence=sent)
# print(a.sentSummarize(rank))