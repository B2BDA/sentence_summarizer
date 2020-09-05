from flask import Flask
import json
from flask import request
from  paragraph_summarization import SentenceSummarizer

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to  Data Science Project'

@app.route('/SentenceSummary/sentence/', methods=['POST'])
def inputSentence():
    sentence = request.json
    sentence = sentence['sentence']
    ss = SentenceSummarizer(sentence[0])
    vec, sent = ss.embedder()
    sim_mat = ss.similarityMatrix(word_embed=vec, sentence=sent)
    rank = ss.sentRank(similarity_matrix=sim_mat, sentence=sent)
    summary = ss.sentSummarize(rank)
    return json.dumps(summary)

if __name__ == '__main__':
    app.run(debug=True)


# curl -i -H "Content-Type: application/json" -X POST -d '{"sentence": ["Gradient Descent is an optimisation algorithm that can be used in a wide variety of problems. The general idea of this method is to iteratively tweak the parameters of a model in order to reach the set of parameter values that minimises the error that such model makes in its predictions.After the parameters of the model have been initialised randomly, each iteration of gradient descent goes as follows: with the given values of such parameters, we use the model to make a prediction for every instance of the training data, and compare that prediction to the actual target value. Once we have computed this aggregated error (known as cost function), we measure the local gradient of this error with respect to the model parameters, and update these parameters by pushing them in the direction of descending gradient, thus making the cost function decrease. The following figure shows graphically how this is done: we start at the orange point, which is the initial random value of the model parameters. After one iteration of gradient descent, we move to the blue point which is directly right and down from the initial orange point: we have gone in the direction of descending gradient"]}' http://127.0.0.1:5000//SentenceSummary/sentence/

# curl -i -H "Content-Type: application/json" -X POST -d '{"sentence": ["Once the poet was walking down a road and then there was a diversion, there were two different paths and he had to choose one out them. The poet says that as he was one person, he could travel on one road only. He had to choose one out of these two roads Yellow wood means a forest with leaves which are wearing out and they have turned yellow in colour – the season of autumn. It represents a world which is full of people, where people have been living for many years. They represent people who are older than the poet. The poet kept standing there and looked at the path very carefully as far as he could see it. Before taking the path, he wanted to know how it was. Was it suitable for him or no. He was able to see the path till from where it curved after which it was covered with trees and was hidden. It happens in our life also when we have choices, we have alternatives, but we have to choose only one out of them, we take time to think about the pros and cons, whether it is suitable for us or not and only then, we take a decision on what path we should choose."]}' http://127.0.0.1:5000//SentenceSummary/sentence/