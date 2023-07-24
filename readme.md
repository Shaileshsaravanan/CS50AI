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
