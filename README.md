# Monte Carlo Tree Search (MCTS) Uno Agent (Java)

An artificial intelligence agent built in Java designed to play fully observable and partially observable variants of Uno using Monte Carlo Tree Search (MCTS) algorithms.

## Features & Decision Algorithms

* **Expected-Outcome MCTS (`ExpectedOutcomeAgent`):** Evaluates full game trees in fully observable settings by combining minimax-style search with rollout distributions at leaf nodes to estimate $Q(s,a)$ values.
* **Upper Confidence Tree (`UCTAgent`):** Implements the UCT/UCB1 selection rule to balance exploration and exploitation during tree expansion:
  $$UCB1 = \bar{Q}(s,a) + \sqrt{\frac{2 \ln N(s)}{N(s,a)}}$$
* **Partially Observable MCTS (`UnoMCTSAgent`):** Extends search decision trees to handle hidden information by stochastically inferring opponent hands through card counting and probabilistic sampling.

## Setup & Execution

### Compilation
```bash
# Mac / Linux
javac -cp "./lib/*:." @uno.srcs

# Windows
javac -cp "./lib/*;." @uno.srcs
```

### Running Games
Play an automated game using custom MCTS agents:
```bash
# Example running two MCTS agents
java -cp "./lib/*:." edu.bu.pas.uno.SingleGameMain src.pas.uno.agents.UCTAgent src.pas.uno.agents.ExpectedOutcomeAgent
```

---

## File Structure
* `src/pas/uno/agents/ExpectedOutcomeAgent.java`: Expected-Outcome MCTS implementation.
* `src/pas/uno/agents/UCTAgent.java`: Upper Confidence Bound Tree MCTS implementation.
* `src/pas/uno/agents/UnoMCTSAgent.java`: Partially Observable MCTS implementation for imperfect-information play.
