# CS50’s Introduction to AI with Python 🚀

Welcome to my journey through CS50’s Introduction to AI with Python! This course takes a deep dive into the concepts and algorithms that form the foundation of modern artificial intelligence.

Caution: Prior to accessing the files in this repository, make sure you've checked out the [academic honesty guidelines](https://cs50.harvard.edu/college/2021/fall/syllabus/#academic-honesty) of CS50.

Certificate of Completion: (Under Progress)

## Notes

Throughout the lectures, I've taken notes on key concepts and algorithms that was covered. Here's a glimpse of what I've learned:

## Week 0: Search

### Concepts
- **Agent**: An entity that perceives its environment and takes action based on that perception.
- **State**: A configuration of the agent and its environment.
- **Actions**: Choices that can be made in a given state.
- **Transition model**: A description of the state resulting from performing any applicable action.
- **Path cost**: A numerical cost associated with a specific path.
- **Evaluation function**: A function that estimates the expected utility of a game from a given state.

### Algorithms
- **DFS** (depth-first search): Explores the deepest node in the frontier first.
- **BFS** (breadth-first search): Explores the shallowest node in the frontier first.
- **Greedy best-first search**: Expands the node that is closest to the goal, based on an estimated heuristic function.
- **A\* search**: Expands the node with the lowest value of "cost to reach node" plus "estimated goal cost."
- **Minimax**: An adversarial search algorithm.

### Projects
- [Degrees](https://github.com/Shaileshsaravanan/CS50AI/tree/main/Week%200%20-%20Search/Degrees)
- [Tic-Tac-Toe](https://github.com/Shaileshsaravanan/CS50AI/tree/main/Week%200%20-%20Search/Tic-Tac-Toe)
- [Quiz 0](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%200%20-%20Search/Quiz%200.png)

## Lecture 1: Knowledge

### Concepts
- **Sentence**: An assertion about the world in a knowledge representation language.
- **Knowledge base**: A set of sentences known by a knowledge-based agent.
- **Entailment**: Sentence A entails sentence B if, in every model where sentence A is true, sentence B is also true.
- **Inference**: The process of deriving new sentences from existing ones.
- **Conjunctive normal form**: A logical sentence that is a conjunction of clauses.
- **First-order logic**: Propositional logic.
- **Second-order logic**: Proposition logic with universal and existential quantification.

### Algorithms
- **Model checking**: Enumerating all possible models to check the truth of a proposition.
- **Conversion to CNF** and **Inference by resolution**

### Projects
- [Knights](https://github.com/Shaileshsaravanan/CS50AI/tree/main/Week%201%20-%20Knowledge/Knights)
- Minesweeper: Under Completion
- [Quiz 1](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%201%20-%20Knowledge/Quiz%201.png)

## Lecture 2: Uncertainty

### Concepts
- **Unconditional probability**: Degree of belief in a proposition in the absence of any other evidence.
- **Conditional probability**: Degree of belief in a proposition given some evidence that has already been revealed.
- **Random variable**: A variable in probability theory with a domain of possible values.
- **Independence**: The knowledge that one event occurs does not affect the probability of the other event.
- **Bayes' Rule**: P(a) P(b| a) = P(b) P(a|b) |
- **Bayesian network**: A data structure that represents the dependencies among random variables.
- **Markov assumption**: The assumption that the current state depends on only a finite fixed number of previous states.
- **Markov chain**: A sequence of random variables where the distribution of each variable follows the Markov assumption.
- **Hidden Markov Model**: A Markov model for a system with hidden states that generate some observed events.

### Algorithms
- **Inference by enumeration**
- **Sampling**
- **Likelihood weighting**

### Projects
- Heredity: Under Completion
- PageRank: Under Completion
- [Quiz 2](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%202%20-%20Uncertainty/Quiz%202.png)

## Lecture 3: Optimization

### Concepts
- **Optimization**: The process of choosing the best option from a set of options.

### Algorithms
- **Local Search Hill Climbing**
    - **Steepest-ascent**: Choose the highest-valued neighbor.
    - **Stochastic**: Choose randomly from higher-valued neighbors.
    - **First-choice**: Choose the first higher-valued neighbor.
    - **Random-restart**: Conduct hill climbing multiple times.
    - **Local beam search**: Choose the k highest-valued neighbors.
- **Simulated Annealing**: Accept worse-valued neighbors early on to explore different solutions.
- **Linear Programming**
    - **Simplex**
    - **Interior-Point**
- **Constraint Satisfaction Problems**
    - **Arc consistency**: Make X arc-consistent with respect to Y by removing elements from X's domain until every choice for X has a possible choice for Y.
    - **Backtracking search**

### Projects
- Crossword: Under Completion
- [Quiz 3](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%203%20-%20Optimization/Quiz%203.png)

## Lecture 4: Learning

### Concepts
- **Supervised Learning**: Learning a function to map inputs to outputs using a data set of input-output pairs.
    - **Classification**: Learning a function that maps an input point to a discrete category.
    - **Regression**: Learning a function that maps an input point to a continuous value.
    - **Loss function**: Measures how poorly a hypothesis performs.
    - **Overfitting**: When a model fits too closely to a particular data set and fails to generalize.
    - **Regularization**: Penalizing complex hypotheses in favor of simpler ones.
    - **Holdout Cross-Validation**: Splitting data into training and test sets for evaluation.
    - **k-fold Cross-Validation**: Splitting data into k sets for evaluation, using each set as a test set once.
- **Reinforcement Learning**: Learning what actions to take in the future based on rewards or punishments.
- **Unsupervised Learning**: Learning patterns from input data without additional feedback.
- **Clustering**: Organizing objects into groups based on their similarities.

### Algorithms
- **k-Nearest Neighbor Classification**: Choosing the most common class among the k nearest data points to an input.
- **Support Vector Machines (SVM)**
- **Markov Decision Process**: A model for decision-making with states, actions, and rewards.
- **Q-Learning**: Learning a function Q(s, a) that estimates the value of performing action a in state s.
- **Greedy Decision-Making**
- **Epsilon-Greedy**
- **k-Means Clustering**: Clustering data by assigning points to clusters and updating cluster centers.

### Projects
- Shopping: Under Completion
- Nim: Under Completion
- [Quiz 4](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%204%20-%20Learning/Quiz%204.png)

## Lecture 5: Neural Networks

### Concepts
- **Neural Network**: A network of interconnected artificial neurons that can learn from data.
- **Artificial Neuron/Perceptron**: A mathematical function that takes inputs, applies weights and biases, and produces an output.
- **Activation Function**: A function that determines the output of an artificial neuron.
- **Feedforward Neural Network**: A neural network where information flows in one direction, from the input layer to the output layer.
- **Backpropagation**: A learning algorithm for adjusting the weights and biases of a neural network.
- **Convolutional Neural Network (CNN)**: A type of neural network commonly used for image recognition and processing.
- **Recurrent Neural Network (RNN)**: A type of neural network that can process sequential data by using feedback connections.
- **Long Short-Term Memory (LSTM)**: A type of RNN that can learn long-term dependencies and is effective in handling sequential data.

### Algorithms
- **Perceptron Learning Algorithm**: An algorithm for training a single artificial neuron to classify linearly separable data.
- **Gradient Descent**: An optimization algorithm that adjusts the weights and biases of a neural network based on the error gradient.
- **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that randomly selects a subset of training examples (mini-batch) for each iteration.
- **Convolution**: An operation that applies a filter/kernel to an input image to extract features.
- **Pooling**: An operation that reduces the spatial size of the input by selecting the maximum or average value within a region.
- **Recurrent Neural Network (RNN)**: A type of neural network that uses recurrent connections to process sequential data.
- **Long Short-Term Memory (LSTM)**: A type of RNN that can learn long-term dependencies by using a gating mechanism to control information flow.

### Projects
- Digits: Under Completion
- Traffic: Under Completion
- [Quiz 5](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%205%20-%20Neural%20Networks/Quiz%205.png)

## Lecture 6: Language

### Concepts
- **Language Modeling**: Building statistical models of language to predict the probability of a sequence of words.
- **n-gram**: A sequence of n words or characters used in language modeling.
- **Part-of-Speech Tagging**: Assigning a grammatical category (noun, verb, adjective, etc.) to each word in a sentence.
- **Hidden Markov Models (HMM)**: Probabilistic models used to model sequential data with hidden states.
- **Information Retrieval**: Finding relevant information from a collection of documents based on a user's query.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: A numerical statistic used to reflect the importance of a term in a document within a collection.
- **Word Embeddings**: Dense vector representations of words that capture semantic relationships.
- **Word2Vec**: A popular word embedding model that learns word representations based on the context in which they appear.

### Algorithms
- **n-gram Language Modeling**: Estimating the probability of a word given its previous n-1 words.
- **Hidden Markov Models (HMM)**: Modeling sequential data with hidden states and observed emissions.
- **Viterbi Algorithm**: Finding the most likely sequence of hidden states in an HMM given an observed sequence.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: Calculating the importance of a term in a document relative to a collection.
- **Vector Space Model**: Representing documents and queries as vectors in a high-dimensional space for information retrieval.
- **Cosine Similarity**

### Projects
- Parser: Under Completion
- Questions: Under Completion
- [Quiz 6](https://github.com/Shaileshsaravanan/CS50AI/blob/main/Week%206%20-%20Language/Quiz%206.png)