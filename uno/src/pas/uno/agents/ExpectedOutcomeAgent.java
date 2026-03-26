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


public class ExpectedOutcomeAgent
    extends MCTSAgent
{

    // how deep before we stop expanding and just rollout
    private static final int MAX_DEPTH = 3;

    // time limit per search call in ms — leaves buffer for the game engine timeout
    private static final long TIME_LIMIT_MS = 500;

    private long searchStartTime;

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

    /**
     * builds a dummy agent array indexed by logical player.
     * dummy agents have the correct playerIdx for creating moves.
     */
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

    // random agents used as rollout proxies — they pick legal moves via game.getMove()
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
                // draw the full unresolved stack before advancing the turn
                int currentLogical = this.getGameView().getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                int total          = childGame.getUnresolvedCards().total();
                childGame.drawTotal(currentHand, total);
                childGame.getUnresolvedCards().clear();
            }
            else if (this.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD)
            {
                // draw one card — it lands at the end of the hand
                int currentLogical = this.getGameView().getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                childGame.drawCard(currentHand);
                // move is either null (keep) or a Move to play the drawn card
            }

            childGame.resolveMove(move);

            GameView childView        = childGame.getOmniscientView();
            int childLogicalPlayerIdx = childView.getPlayerOrder().getCurrentLogicalPlayerIdx();

            return new MCTSNode(childView, childLogicalPlayerIdx, this, this.agents);
        }
    }

    public ExpectedOutcomeAgent(final int playerIdx,
                                final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    /**
     * runs a random game from startView using RandomAgent proxies via the game loop.
     * returns 1.0 if myLogicalPlayerIdx wins, 0.0 otherwise.
     */
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
     * expected-outcome mcts — expands ALL children at each node like minimax,
     * does NUM_ROLLOUTS rollouts at each artificial leaf to estimate value.
     * returns the expected utility from myLogicalPlayerIdx's perspective.
     */
    private static final int NUM_ROLLOUTS = 5;

    private float expectedOutcome(final MCTSNode node,
                                  final int myLogicalPlayerIdx,
                                  final int depth,
                                  final DummyAgent[] dummies,
                                  final RandomAgent[] randoms)
    {
        GameView view     = node.getGameView();
        NodeState state   = node.getNodeState();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Random rng        = this.getRandom();
        DummyAgent agent  = dummies[currentPlayer];

        // terminal — real utility
        if (node.isTerminal())
        {
            Game termGame = new Game(view, dummies);
            int winner    = findWinner(termGame);
            return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
        }

        // treat as leaf if we're running out of time
        if (depth == 0 || (System.currentTimeMillis() - searchStartTime) > TIME_LIMIT_MS)
        {
            float total = 0.0f;
            for (int r = 0; r < NUM_ROLLOUTS; r++)
            {
                total += rollout(view, myLogicalPlayerIdx, randoms);
            }
            return total / NUM_ROLLOUTS;
        }

        if (state == NodeState.HAS_LEGAL_MOVES)
        {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            int numMoves = legalMoves.size();
            float[] childVals = new float[numMoves];
            HandView hv       = view.getHandView(currentPlayer);
            Set<Integer> actualLegal = hv.getLegalMoves(view);

            // expand every child
            for (int i = 0; i < numMoves; i++)
            {
                int cardIdx = legalMoves.get(i);

                // skip if not actually legal per the game view
                if (!actualLegal.contains(cardIdx)) continue;

                Card card = hv.getCard(cardIdx);

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
                float val      = expectedOutcome(child, myLogicalPlayerIdx, depth - 1, dummies, randoms);
                childVals[i]   = val;

                node.setQValueTotal(i, node.getQValueTotal(i) + val);
                node.setQCount(i, node.getQCount(i) + 1);
            }

            // minimax: max on our turn, min on opponent's
            if (currentPlayer == myLogicalPlayerIdx)
            {
                float best = Float.NEGATIVE_INFINITY;
                for (float v : childVals) best = Math.max(best, v);
                return best;
            }
            else
            {
                float worst = Float.POSITIVE_INFINITY;
                for (float v : childVals) worst = Math.min(worst, v);
                return worst;
            }
        }
        else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
        {
            // only one child
            MCTSNode child = (MCTSNode) node.getChild(null);
            float val      = expectedOutcome(child, myLogicalPlayerIdx, depth - 1, dummies, randoms);

            int idx = DrawUnresolvedCardsIdxs.MOVE_IDX;
            node.setQValueTotal(idx, node.getQValueTotal(idx) + val);
            node.setQCount(idx, node.getQCount(idx) + 1);

            return val;
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            HandView hv        = view.getHandView(currentPlayer);
            // the drawn card will be at index hv.size() after getChild draws it
            int drawnIdx       = hv.size();
            Set<Integer> legal = hv.getLegalMoves(view);

            // always explore keep child (null move = keep the drawn card)
            MCTSNode keepChild = (MCTSNode) node.getChild(null);
            float keepVal      = expectedOutcome(keepChild, myLogicalPlayerIdx, depth - 1, dummies, randoms);

            node.setQValueTotal(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX,
                node.getQValueTotal(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX) + keepVal);
            node.setQCount(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX,
                node.getQCount(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX) + 1);

            float playVal = keepVal; // default if not playable

            // check the child's hand for the drawn card legality
            // since drawing happens inside getChild, check using the keep child's view
            HandView keepHv     = keepChild.getGameView().getHandView(currentPlayer);
            int actualDrawnIdx  = keepHv.size() - 1; // last card = drawn card
            Card drawn          = keepHv.getCard(actualDrawnIdx);
            Set<Integer> keepLegal = keepHv.getLegalMoves(keepChild.getGameView());

            if (keepLegal.contains(actualDrawnIdx))
            {
                Move playMove;
                if (drawn.isWild())
                {
                    playMove = Move.createMove(agent, actualDrawnIdx, Color.getRandomColor(rng));
                }
                else
                {
                    playMove = Move.createMove(agent, actualDrawnIdx);
                }
                MCTSNode playChild = (MCTSNode) node.getChild(playMove);
                playVal            = expectedOutcome(playChild, myLogicalPlayerIdx, depth - 1, dummies, randoms);

                node.setQValueTotal(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX,
                    node.getQValueTotal(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX) + playVal);
                node.setQCount(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX,
                    node.getQCount(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX) + 1);
            }

            if (currentPlayer == myLogicalPlayerIdx)
            {
                return Math.max(playVal, keepVal);
            }
            else
            {
                return Math.min(playVal, keepVal);
            }
        }
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

        this.searchStartTime = System.currentTimeMillis();
        expectedOutcome(root, myLogicalIdx, MAX_DEPTH, dummies, randoms);

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
            // no choice — game engine handles drawing
            return null;
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            // pick whichever of play/keep has the better q-value
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
                // keep the drawn card — return null per api contract
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