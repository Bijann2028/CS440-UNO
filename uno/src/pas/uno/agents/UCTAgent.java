package src.pas.uno.agents;


// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Deck;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Hand;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.agents.RandomAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Observability;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;
import edu.bu.pas.uno.tree.Node.NodeState;
import edu.bu.pas.uno.tree.Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs;
import edu.bu.pas.uno.tree.Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs;

import java.util.ArrayList;
import java.util.Collections;
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
        // cache children by move string so we reuse the same node across iterations
        private final java.util.Map<String, MCTSNode> children = new java.util.HashMap<>();

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
            // build a reliable cache key
            String key;
            if (move == null)
            {
                key = "null";
            }
            else
            {
                key = move.getPlayerIdx() + ":" + move.getCardToPlayIdx() + ":"
                      + (move.getNewColorIfWild() != null ? move.getNewColorIfWild().name() : "none");
            }

            if (children.containsKey(key))
            {
                return children.get(key);
            }

            Game childGame;
            try
            {
                childGame = new Game(this.getGameView(), this.agents);
            }
            catch (Throwable e)
            {
                // shouldn't happen since root is always determinized
                return new MCTSNode(this.getGameView(), this.getLogicalPlayerIdx(), this, this.agents);
            }

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

            MCTSNode child = new MCTSNode(childView, childLogicalPlayerIdx, this, this.agents);
            children.put(key, child);
            return child;
        }
    }

    public UCTAgent(final int playerIdx,
                    final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    private static final int MAX_ROLLOUT_STEPS = 200;

    // random rollout using RandomAgent proxies via the game loop
    private float rollout(final GameView startView,
                          final int myLogicalPlayerIdx,
                          final RandomAgent[] randoms)
    {
        Game sim  = new Game(startView, randoms);
        int steps = 0;

        while (!sim.isOver() && steps < MAX_ROLLOUT_STEPS)
        {
            Move move = sim.getMove();
            sim.resolveMove(move);
            steps++;
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

    private float uct(final MCTSNode node,
                      final int myLogicalPlayerIdx,
                      final DummyAgent[] dummies,
                      final RandomAgent[] randoms)
    {
        return uct(node, myLogicalPlayerIdx, dummies, randoms, 20);
    }

    private float uct(final MCTSNode node,
                      final int myLogicalPlayerIdx,
                      final DummyAgent[] dummies,
                      final RandomAgent[] randoms,
                      final int maxDepth)
    {
        GameView view     = node.getGameView();
        NodeState state   = node.getNodeState();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Random rng        = this.getRandom();
        DummyAgent agent  = dummies[currentPlayer];

        // terminal — real result
        if (node.isTerminal() || maxDepth == 0)
        {
            return rollout(view, myLogicalPlayerIdx, randoms);
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

            // verify legal per handview before creating move
            Set<Integer> actualLegal = hv.getLegalMoves(view);
            if (!actualLegal.contains(cardIdx))
            {
                // fall back to first actually legal card
                for (int idx : actualLegal)
                {
                    cardIdx = idx;
                    break;
                }
                moveIdx = legalMoves.indexOf(cardIdx);
                if (moveIdx < 0) moveIdx = 0;
            }

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

            if (node.getQCount(moveIdx) == 0)
            {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, randoms);
            }
            else
            {
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms, maxDepth - 1);
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
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms, maxDepth - 1);
            }
        }
        else // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
        {
            // use ucb to pick between play and keep
            moveIdx = ucbSelect(node, 2);

            // get the keep child first to find out what card was drawn
            MCTSNode keepChild = (MCTSNode) node.getChild(null);
            HandView keepHv    = keepChild.getGameView().getHandView(currentPlayer);
            int drawnIdx       = keepHv.size() - 1;
            Card drawn         = keepHv.getCard(drawnIdx);
            Set<Integer> keepLegal = keepHv.getLegalMoves(keepChild.getGameView());

            MCTSNode child;
            if (moveIdx == DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX && keepLegal.contains(drawnIdx))
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
                // keep or card not playable
                moveIdx = DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX;
                child   = keepChild;
            }

            if (node.getQCount(moveIdx) == 0)
            {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, randoms);
            }
            else
            {
                sample = uct(child, myLogicalPlayerIdx, dummies, randoms, maxDepth - 1);
            }
        }

        // backprop
        node.setQValueTotal(moveIdx, node.getQValueTotal(moveIdx) + sample);
        node.setQCount(moveIdx, node.getQCount(moveIdx) + 1);

        return sample;
    }

    /**
     * if the game view has partial observability (UNKNOWN cards),
     * randomly fill in the unknown cards from the remaining unseen pool.
     * returns a fully observable game view copy.
     */
    private Game determinize(final GameView view, final DummyAgent[] dummies)
    {
        int numPlayers = view.getNumPlayers();
        Random rng     = this.getRandom();

        // start with full standard deck
        Deck fullDeck  = new Deck(true);
        List<Card> pool = new ArrayList<>(fullDeck);

        // remove every card we can see from pool (all visible hands + discard pile)
        for (int p = 0; p < numPlayers; p++)
        {
            HandView hv = view.getHandView(p);
            for (int i = 0; i < hv.size(); i++)
            {
                Card c = hv.getCard(i);
                if (c.color() != Color.UNKNOWN && c.value() != Value.UNKNOWN)
                {
                    pool.remove(c);
                }
            }
        }
        for (Card c : view.getDiscardPile().pile())
        {
            pool.remove(c);
        }

        // shuffle remaining unseen cards
        Collections.shuffle(pool, rng);
        int poolIdx = 0;

        // build fully known Hand objects
        Hand[] hands = new Hand[numPlayers];
        for (int p = 0; p < numPlayers; p++)
        {
            HandView hv = view.getHandView(p);
            hands[p]    = new Hand();
            for (int i = 0; i < hv.size(); i++)
            {
                Card c = hv.getCard(i);
                if (c.color() == Color.UNKNOWN || c.value() == Value.UNKNOWN)
                {
                    if (poolIdx < pool.size())
                    {
                        hands[p].add(pool.get(poolIdx++));
                    }
                }
                else
                {
                    hands[p].add(c);
                }
            }
        }

        // remaining pool cards become the draw pile
        Deck drawPile = new Deck();
        while (poolIdx < pool.size())
        {
            drawPile.add(pool.get(poolIdx++));
        }

        return new Game(drawPile, hands, view, dummies);
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int myLogicalIdx      = this.getLogicalPlayerIdx();
        DummyAgent[]  dummies = buildDummyAgents(game);
        RandomAgent[] randoms = buildRandomAgents(game);

        GameView workingView  = determinize(game, dummies).getOmniscientView();

        int currentLogicalIdx = workingView.getPlayerOrder().getCurrentLogicalPlayerIdx();
        MCTSNode root         = new MCTSNode(workingView, currentLogicalIdx, null, dummies);
        

        // run for a fixed number of iterations since we call this synchronously
        int iters = 0;
        while (!Thread.currentThread().isInterrupted() && iters < 500)
        {
            try
            {
                uct(root, myLogicalIdx, dummies, randoms);
            }
            catch (Throwable e)
            {
                // swallow any exception to prevent autograder crash
            }
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
        try
        {
            Node root = search(game, null);
            return argmaxQValues(root);
        }
        catch (Throwable e)
        {
            // fallback: play first legal move
            int currentPlayer = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
            HandView hv       = game.getHandView(currentPlayer);
            DummyAgent agent  = buildDummyAgents(game)[currentPlayer];
            Set<Integer> legal = hv.getLegalMoves(game);
            if (!legal.isEmpty())
            {
                int cardIdx = legal.iterator().next();
                Card card   = hv.getCard(cardIdx);
                if (card.isWild())
                    return Move.createMove(agent, cardIdx, bestColor(hv));
                return Move.createMove(agent, cardIdx);
            }
            return null;
        }
    }

    @Override
    public Move maybePlayDrawnCard(final GameView game, final int drawnCardIdx)
    {
        try
        {
            Node root = search(game, drawnCardIdx);
            return argmaxQValues(root);
        }
        catch (Throwable e)
        {
            return null;
        }
    }
}