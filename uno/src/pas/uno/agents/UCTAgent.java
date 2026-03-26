package src.pas.uno.agents;


// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Hand;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.agents.RandomAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;
import edu.bu.pas.uno.tree.Node.NodeState;
import edu.bu.pas.uno.tree.Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs;
import edu.bu.pas.uno.tree.Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;


// JAVA PROJECT IMPORTS


public class UCTAgent
    extends MCTSAgent
{

    /**
     * minimal agent stub — holds the correct playerIdx for Move.createMove.
     */
    private static class DummyAgent extends Agent
    {
        public DummyAgent(int playerIdx)
        {
            super(playerIdx, 0L);
        }

        @Override
        public Move chooseCardToPlay(GameView game) { return null; }

        @Override
        public Move maybePlayDrawnCard(GameView game, int drawnCardIdx) { return null; }
    }

    private DummyAgent[] buildDummyAgents(final GameView game)
    {
        int n = game.getNumPlayers();
        DummyAgent[] agents = new DummyAgent[n];
        for (int i = 0; i < n; i++)
        {
            int playerIdx = game.getPlayerOrder().getAgentIdx(i);
            agents[i] = new DummyAgent(playerIdx);
            agents[i].setLogicalPlayerIdx(i);
        }
        return agents;
    }

    private RandomAgent[] buildRandomAgents(final GameView game)
    {
        int n = game.getNumPlayers();
        RandomAgent[] agents = new RandomAgent[n];
        for (int i = 0; i < n; i++)
        {
            int playerIdx = game.getPlayerOrder().getAgentIdx(i);
            agents[i] = new RandomAgent(playerIdx, 0L);
            agents[i].setLogicalPlayerIdx(i);
        }
        return agents;
    }

    public static class MCTSNode
        extends Node
    {
        private final DummyAgent[] agents;

        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent,
                        final DummyAgent[] agents)
        {
            super(game, logicalPlayerIdx, parent);
            this.agents = agents;
        }

        @Override
        public Node getChild(final Move move)
        {
            Game childGame = new Game(this.getGameView(), this.agents);

            if (this.getNodeState() == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
            {
                int currentLogical = this.getGameView().getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                int total          = childGame.getUnresolvedCards().total();
                childGame.drawTotal(currentHand, total);
                childGame.getUnresolvedCards().clear();
            }
            else if (this.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD)
            {
                int currentLogical = this.getGameView().getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                childGame.drawCard(currentHand);
            }

            childGame.resolveMove(move);

            GameView childView        = childGame.getOmniscientView();
            int childLogicalPlayerIdx = childView.getPlayerOrder().getCurrentLogicalPlayerIdx();

            return new MCTSNode(childView, childLogicalPlayerIdx, this, this.agents);
        }
    }

    public UCTAgent(final int playerIdx,
                    final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    // random rollout using RandomAgent proxies via the game loop
    private float rollout(final GameView startView,
                          final int myLogicalPlayerIdx,
                          final RandomAgent[] randoms)
    {
        Game sim = new Game(startView, randoms);

        while (!sim.isOver())
        {
            Move move = sim.getMove();
            sim.resolveMove(move);
        }

        int winner = findWinner(sim);
        return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
    }

    // winner = player with fewest cards, ties go to lowest logical idx
    private int findWinner(final Game sim)
    {
        int numPlayers = sim.getNumPlayers();
        int bestIdx    = 0;
        int bestCount  = Integer.MAX_VALUE;

        for (int i = 0; i < numPlayers; i++)
        {
            int count = sim.getHand(i).size();
            if (count < bestCount)
            {
                bestCount = count;
                bestIdx   = i;
            }
        }
        return bestIdx;
    }

    // picks the color we have the most of — best choice when playing a wild
    private Color bestColor(final HandView hv)
    {
        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
        int[] counts   = new int[4];
        for (int i = 0; i < hv.size(); i++)
        {
            Card c = hv.getCard(i);
            for (int j = 0; j < 4; j++)
                if (c.color() == colors[j]) counts[j]++;
        }
        int best = 0;
        for (int j = 1; j < 4; j++)
            if (counts[j] > counts[best]) best = j;
        return colors[best];
    }

    /**
     * selects action index using UCB rule.
     * Q̄(s,a) + sqrt(2 * log(N(s)) / N(s,a))
     * unvisited actions get infinite priority.
     */
    private int ucbSelect(final MCTSNode node, final int numActions)
    {
        // total visits = sum of all q counts
        long stateCount = 0;
        for (int i = 0; i < numActions; i++)
        {
            stateCount += node.getQCount(i);
        }

        if (stateCount == 0)
        {
            return this.getRandom().nextInt(numActions);
        }

        int bestIdx    = -1;
        double bestVal = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numActions; i++)
        {
            long actionCount = node.getQCount(i);
            double ucb;

            if (actionCount == 0)
            {
                // never tried — explore first
                ucb = Double.POSITIVE_INFINITY;
            }
            else
            {
                double avgQ    = node.getQValue(i);
                double explore = Math.sqrt(2.0 * Math.log(stateCount) / actionCount);
                ucb            = avgQ + explore;
            }

            if (ucb > bestVal)
            {
                bestVal = ucb;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * one iteration of uct:
     * walk tree with ucb until unvisited node, rollout, backprop.
     */
    private float uct(final MCTSNode node,
                      final int myLogicalPlayerIdx,
                      final DummyAgent[] dummies,
                      final RandomAgent[] randoms)
    {
        GameView view     = node.getGameView();
        NodeState state   = node.getNodeState();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Random rng        = this.getRandom();
        DummyAgent agent  = dummies[currentPlayer];

        // terminal — real result
        if (node.isTerminal())
        {
            Game termGame = new Game(view, dummies);
            int winner    = findWinner(termGame);
            return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
        }

        float sample;
        int moveIdx;

        if (state == NodeState.HAS_LEGAL_MOVES)
        {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            int numMoves = legalMoves.size();

            moveIdx     = ucbSelect(node, numMoves);
            int cardIdx = legalMoves.get(moveIdx);
            HandView hv = view.getHandView(currentPlayer);
            Card card   = hv.getCard(cardIdx);

            Move move;
            if (card.isWild())
            {
                move = Move.createMove(agent, cardIdx, Color.getRandomColor(rng));
            }
            else
            {
                move = Move.createMove(agent, cardIdx);
            }

            MCTSNode child = (MCTSNode) node.getChild(move);

            if (node.getQCount(moveIdx) == 0)
            {
                // first visit — rollout from child
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, randoms);
            }
            else
            {
                // already visited — recurse deeper
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms);
            }
        }
        else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
        {
            moveIdx        = DrawUnresolvedCardsIdxs.MOVE_IDX;
            MCTSNode child = (MCTSNode) node.getChild(null);

            if (node.getQCount(moveIdx) == 0)
            {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, randoms);
            }
            else
            {
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms);
            }
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            // two options: play or keep — use ucb to pick
            moveIdx = ucbSelect(node, 2);

            HandView hv  = view.getHandView(currentPlayer);
            int drawnIdx = hv.size() - 1;
            Card drawn   = hv.getCard(drawnIdx);

            MCTSNode child;
            if (moveIdx == DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX)
            {
                Move playMove;
                if (drawn.isWild())
                {
                    playMove = Move.createMove(agent, drawnIdx, Color.getRandomColor(rng));
                }
                else
                {
                    playMove = Move.createMove(agent, drawnIdx);
                }
                child = (MCTSNode) node.getChild(playMove);
            }
            else
            {
                child = (MCTSNode) node.getChild(null);
            }

            if (node.getQCount(moveIdx) == 0)
            {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, randoms);
            }
            else
            {
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms);
            }
        }

        // backprop
        node.setQValueTotal(moveIdx, node.getQValueTotal(moveIdx) + sample);
        node.setQCount(moveIdx, node.getQCount(moveIdx) + 1);

        return sample;
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int myLogicalIdx      = this.getLogicalPlayerIdx();
        int currentLogicalIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        DummyAgent[]  dummies = buildDummyAgents(game);
        RandomAgent[] randoms = buildRandomAgents(game);
        MCTSNode root         = new MCTSNode(game, currentLogicalIdx, null, dummies);
        

        // run for a fixed number of iterations since we call this synchronously
        int iters = 0;
        while (!Thread.currentThread().isInterrupted() && iters < 5000)
        {
            uct(root, myLogicalIdx, dummies, randoms);
            iters++;
        }

        return root;
    }

    @Override
    public Move argmaxQValues(final Node node)
    {
        if (node == null) return null;

        NodeState state   = node.getNodeState();
        GameView view     = node.getGameView();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        DummyAgent agent  = buildDummyAgents(view)[currentPlayer];

        if (state == NodeState.HAS_LEGAL_MOVES)
        {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            HandView hv  = view.getHandView(currentPlayer);
            int bestIdx  = 0;
            float bestQV = Float.NEGATIVE_INFINITY;

            for (int i = 0; i < legalMoves.size(); i++)
            {
                float qv = node.getQValue(i);
                if (qv > bestQV)
                {
                    bestQV  = qv;
                    bestIdx = i;
                }
            }

            int cardIdx = legalMoves.get(bestIdx);
            Card card   = hv.getCard(cardIdx);

            if (card.isWild())
            {
                return Move.createMove(agent, cardIdx, bestColor(hv));
            }
            else
            {
                return Move.createMove(agent, cardIdx);
            }
        }
        else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
        {
            return null;
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            float playQV = node.getQValue(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX);
            float keepQV = node.getQValue(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX);

            if (playQV >= keepQV)
            {
                HandView hv  = view.getHandView(currentPlayer);
                int drawnIdx = hv.size() - 1;
                Card card    = hv.getCard(drawnIdx);

                if (card.isWild())
                {
                    return Move.createMove(agent, drawnIdx, bestColor(hv));
                }
                else
                {
                    return Move.createMove(agent, drawnIdx);
                }
            }
            else
            {
                return null;
            }
        }
    }

    @Override
    public Move chooseCardToPlay(final GameView game)
    {
        Node root = search(game, null);
        return argmaxQValues(root);
    }

    @Override
    public Move maybePlayDrawnCard(final GameView game, final int drawnCardIdx)
    {
        Node root = search(game, drawnCardIdx);
        return argmaxQValues(root);
    }
}