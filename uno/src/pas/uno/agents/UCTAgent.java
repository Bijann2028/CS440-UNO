package src.pas.uno.agents;


// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Hand;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
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

    // max steps per rollout to avoid infinite games
    private static final int MAX_ROLLOUT_STEPS = 500;

    // number of uct iterations per turn
    private static final int NUM_ITERATIONS = 5000;

    /**
     * minimal agent stub — just holds a playerIdx so Move.createMove gets the right id.
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
     * builds a dummy agent array with the correct playerIdx for each logical slot.
     */
    private DummyAgent[] buildDummyAgents(final GameView game)
    {
        int n = game.getNumPlayers();
        DummyAgent[] agents = new DummyAgent[n];
        for (int logicalIdx = 0; logicalIdx < n; logicalIdx++)
        {
            int playerIdx = game.getPlayerOrder().getAgentIdx(logicalIdx);
            agents[logicalIdx] = new DummyAgent(playerIdx);
            agents[logicalIdx].setLogicalPlayerIdx(logicalIdx);
        }
        return agents;
    }

    public static class MCTSNode
        extends Node
    {
        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent)
        {
            super(game, logicalPlayerIdx, parent);
        }

        /**
         * applies the move to a copy of the current state and returns the resulting child node.
         */
        @Override
        public Node getChild(final Move move)
        {
            Game childGame = new Game(this.getGameView());

            try
            {
                childGame.resolveMove(move);
            }
            catch (NullPointerException e)
            {
                // null rng in copied game — return node with current state
            }

            GameView childView        = childGame.getOmniscientView();
            int childLogicalPlayerIdx = childView.getPlayerOrder().getCurrentLogicalPlayerIdx();

            return new MCTSNode(childView, childLogicalPlayerIdx, this);
        }
    }

    public UCTAgent(final int playerIdx,
                    final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    // random rollout from startView, returns 1.0 if we win else 0.0
    private float rollout(final GameView startView,
                          final int myLogicalPlayerIdx,
                          final DummyAgent[] agents)
    {
        Game stateCopy = new Game(startView);
        int steps      = 0;

        try
        {
            while (!stateCopy.isOver() && steps < MAX_ROLLOUT_STEPS)
            {
                int currentLogical = stateCopy.getPlayerOrder().getCurrentLogicalPlayerIdx();
                DummyAgent agent   = agents[currentLogical];
                GameView simView   = stateCopy.getOmniscientView();
                Hand currentHand   = stateCopy.getHand(currentLogical);

                List<Card> cardList = new ArrayList<>();
                for (int i = 0; i < currentHand.size(); i++)
                {
                    cardList.add(currentHand.getCard(i));
                }
                HandView hv        = new HandView(cardList);
                Set<Integer> legal = hv.getLegalMoves(simView);

                Move move;

                if (!stateCopy.getUnresolvedCards().isEmpty() && legal.isEmpty())
                {
                    move = null;
                }
                else if (legal.isEmpty())
                {
                    int drawnIdx = stateCopy.drawCard(currentHand);
                    Card drawn   = currentHand.getCard(drawnIdx);

                    List<Card> newList = new ArrayList<>();
                    for (int i = 0; i < currentHand.size(); i++)
                    {
                        newList.add(currentHand.getCard(i));
                    }
                    HandView hv2           = new HandView(newList);
                    Set<Integer> afterDraw = hv2.getLegalMoves(stateCopy.getOmniscientView());

                    if (afterDraw.contains(drawnIdx))
                    {
                        if (drawn.isWild())
                        {
                            move = Move.createMove(agent, drawnIdx,
                                                   Color.getRandomColor(this.getRandom()));
                        }
                        else
                        {
                            move = Move.createMove(agent, drawnIdx);
                        }
                    }
                    else
                    {
                        move = null;
                    }
                }
                else
                {
                    List<Integer> legalList = new ArrayList<>(legal);
                    int cardIdx = legalList.get(this.getRandom().nextInt(legalList.size()));
                    Card card   = currentHand.getCard(cardIdx);

                    if (card.isWild())
                    {
                        move = Move.createMove(agent, cardIdx,
                                               Color.getRandomColor(this.getRandom()));
                    }
                    else
                    {
                        move = Move.createMove(agent, cardIdx);
                    }
                }

                stateCopy.resolveMove(move);
                steps++;
            }
        }
        catch (NullPointerException e)
        {
            // null rng when reshuffling — treat as inconclusive
            return 0.5f;
        }

        int winner = findWinner(stateCopy);
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

    /**
     * selects the action index to follow using the UCB rule.
     * Q̄(s,a) + sqrt(2 * log(N(s)) / N(s,a))
     * unvisited actions get infinite priority.
     */
    private int ucbSelect(final MCTSNode node, final int numActions)
    {
        long stateCount = node.getStateCount();

        // if node has never been visited, pick randomly
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
                // unvisited action — always explore first
                ucb = Double.POSITIVE_INFINITY;
            }
            else
            {
                double avgQ      = node.getQValue(i);
                double explore   = Math.sqrt(2.0 * Math.log(stateCount) / actionCount);
                ucb              = avgQ + explore;
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
     * 1. walk tree using ucb until we reach an unvisited or terminal node
     * 2. rollout from that node
     * 3. backprop the result up through all ancestors
     */
    private float uct(final MCTSNode node,
                      final int myLogicalPlayerIdx,
                      final DummyAgent[] agents)
    {
        GameView view     = node.getGameView();
        NodeState state   = node.getNodeState();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Random rng        = this.getRandom();
        DummyAgent agent  = agents[currentPlayer];

        // terminal — return actual result
        if (node.isTerminal())
        {
            Game termGame = new Game(view);
            int winner    = findWinner(termGame);
            return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
        }

        float sample;

        if (state == NodeState.HAS_LEGAL_MOVES)
        {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            int numMoves = legalMoves.size();

            // use ucb to pick which action to follow
            int i       = ucbSelect(node, numMoves);
            int cardIdx = legalMoves.get(i);
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

            // if this action hasn't been visited yet, rollout from the child
            // otherwise recurse deeper into the tree
            MCTSNode child = (MCTSNode) node.getChild(move);

            if (node.getQCount(i) == 0)
            {
                // new node — rollout from here
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, agents);
            }
            else
            {
                // already in tree — keep walking
                sample = uct(child, myLogicalPlayerIdx, agents);
            }

            // backprop
            node.setQValueTotal(i, node.getQValueTotal(i) + sample);
            node.setQCount(i, node.getQCount(i) + 1);
        }
        else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
        {
            // only one action — forced draw
            MCTSNode child = (MCTSNode) node.getChild(null);
            int idx        = DrawUnresolvedCardsIdxs.MOVE_IDX;

            if (node.getQCount(idx) == 0)
            {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, agents);
            }
            else
            {
                sample = uct(child, myLogicalPlayerIdx, agents);
            }

            node.setQValueTotal(idx, node.getQValueTotal(idx) + sample);
            node.setQCount(idx, node.getQCount(idx) + 1);
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            // must play the drawn card — only one action
            HandView hv  = view.getHandView(currentPlayer);
            int drawnIdx = hv.size() - 1;
            Card drawn   = hv.getCard(drawnIdx);

            Move playMove;
            if (drawn.isWild())
            {
                playMove = Move.createMove(agent, drawnIdx, Color.getRandomColor(rng));
            }
            else
            {
                playMove = Move.createMove(agent, drawnIdx);
            }

            MCTSNode playChild = (MCTSNode) node.getChild(playMove);
            int idx            = DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX;

            if (node.getQCount(idx) == 0)
            {
                sample = rollout(playChild.getGameView(), myLogicalPlayerIdx, agents);
            }
            else
            {
                sample = uct(playChild, myLogicalPlayerIdx, agents);
            }

            node.setQValueTotal(idx, node.getQValueTotal(idx) + sample);
            node.setQCount(idx, node.getQCount(idx) + 1);
        }

        return sample;
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int myLogicalIdx      = this.getLogicalPlayerIdx();
        int currentLogicalIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        MCTSNode root         = new MCTSNode(game, currentLogicalIdx, null);
        DummyAgent[] agents   = buildDummyAgents(game);

        int iters = 0;
        while (!Thread.currentThread().isInterrupted() && iters < NUM_ITERATIONS)
        {
            uct(root, myLogicalIdx, agents);
            iters++;
        }

        return root;
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
            int bestMoveIdx  = -1;
            float bestQValue = Float.NEGATIVE_INFINITY;

            for (int i = 0; i < legalMoves.size(); i++)
            {
                float qv = node.getQValue(i);
                if (qv > bestQValue)
                {
                    bestQValue  = qv;
                    bestMoveIdx = i;
                }
            }

            if (bestMoveIdx < 0)
            {
                bestMoveIdx = 0;
            }

            int cardIdx = legalMoves.get(bestMoveIdx);
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
            // must play the drawn card
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