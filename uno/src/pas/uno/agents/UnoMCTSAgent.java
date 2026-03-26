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


public class UnoMCTSAgent
    extends MCTSAgent
{

    private static final int MAX_ROLLOUT_STEPS = 200;

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
        private final UnoMCTSAgent outer;
        private final java.util.Map<String, MCTSNode> children = new java.util.HashMap<>();

        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent,
                        final DummyAgent[] agents,
                        final UnoMCTSAgent outer)
        {
            super(game, logicalPlayerIdx, parent);
            this.agents = agents;
            this.outer  = outer;
        }

        @Override
        public Node getChild(final Move move)
        {
            String key = (move == null) ? "null"
                : move.getPlayerIdx() + ":" + move.getCardToPlayIdx() + ":"
                  + (move.getNewColorIfWild() != null ? move.getNewColorIfWild().name() : "none");

            if (children.containsKey(key))
            {
                return children.get(key);
            }

            // ensure no UNKNOWN cards before constructing mutable Game
            GameView safeView;
            try
            {
                new Game(this.getGameView(), this.agents); // test if view is clean
                safeView = this.getGameView();
            }
            catch (Throwable e)
            {
                // view has UNKNOWN cards — determinize it
                safeView = outer.determinize(this.getGameView(), this.agents).getOmniscientView();
            }

            Game childGame = new Game(safeView, this.agents);

            if (this.getNodeState() == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
            {
                int currentLogical = safeView.getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                int total          = childGame.getUnresolvedCards().total();
                childGame.drawTotal(currentHand, total);
                childGame.getUnresolvedCards().clear();
            }
            else if (this.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD)
            {
                int currentLogical = safeView.getPlayerOrder().getCurrentLogicalPlayerIdx();
                Hand currentHand   = childGame.getHand(currentLogical);
                childGame.drawCard(currentHand);
            }

            childGame.resolveMove(move);

            GameView childView        = childGame.getOmniscientView();
            int childLogicalPlayerIdx = childView.getPlayerOrder().getCurrentLogicalPlayerIdx();

            MCTSNode child = new MCTSNode(childView, childLogicalPlayerIdx, this, this.agents, this.outer);
            children.put(key, child);
            return child;
        }
    }

    public UnoMCTSAgent(final int playerIdx,
                        final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    // random rollout using RandomAgent proxies via game loop
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

    // picks the color we have the most of in our hand
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

    // ucb selection: Q̄(s,a) + sqrt(2 * log(N(s)) / N(s,a))
    private int ucbSelect(final MCTSNode node, final int numActions)
    {
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
     * randomly fill UNKNOWN cards from the unseen card pool.
     * re-determinized on every iteration for better partial observability coverage.
     */
    private Game determinize(final GameView view, final DummyAgent[] dummies)
    {
        int numPlayers = view.getNumPlayers();
        Random rng     = this.getRandom();

        Deck fullDeck   = new Deck(true);
        List<Card> pool = new ArrayList<>(fullDeck);

        // remove all visible cards from pool
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

        Collections.shuffle(pool, rng);
        int poolIdx = 0;

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

        Deck drawPile = new Deck();
        while (poolIdx < pool.size())
        {
            drawPile.add(pool.get(poolIdx++));
        }

        return new Game(drawPile, hands, Observability.FULL, view, dummies);
    }

    // one uct iteration on a determinized root
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

            // verify legal before creating move
            Set<Integer> actualLegal = hv.getLegalMoves(view);
            if (!actualLegal.contains(cardIdx))
            {
                for (int idx : actualLegal) { cardIdx = idx; break; }
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
            moveIdx = ucbSelect(node, 2);

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

        node.setQValueTotal(moveIdx, node.getQValueTotal(moveIdx) + sample);
        node.setQCount(moveIdx, node.getQCount(moveIdx) + 1);

        return sample;
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int myLogicalIdx      = this.getLogicalPlayerIdx();
        DummyAgent[]  dummies = buildDummyAgents(game);
        RandomAgent[] randoms = buildRandomAgents(game);

        // determinize once for the root
        GameView workingView  = determinize(game, dummies).getOmniscientView();
        int currentLogicalIdx = workingView.getPlayerOrder().getCurrentLogicalPlayerIdx();
        MCTSNode root         = new MCTSNode(workingView, currentLogicalIdx, null, dummies, this);

        long deadline = System.currentTimeMillis() + 4000;

        while (!Thread.currentThread().isInterrupted()
               && System.currentTimeMillis() < deadline)
        {
            try
            {
                // re-determinize each iteration for partial observability
                GameView iterView = determinize(game, dummies).getOmniscientView();
                MCTSNode iterRoot = new MCTSNode(iterView, currentLogicalIdx, null, dummies, this);
                float sample      = uct(iterRoot, myLogicalIdx, dummies, randoms);
                updateRootFromIter(root, iterRoot, sample);
            }
            catch (Throwable e)
            {
                // swallow to prevent crash
            }
        }

        return root;
    }

    /**
     * copies q-value updates from an iteration root back into the shared root.
     * we find which action was sampled by checking which q-count changed.
     */
    private void updateRootFromIter(final MCTSNode root,
                                    final MCTSNode iterRoot,
                                    final float sample)
    {
        NodeState state = root.getNodeState();

        if (state == NodeState.HAS_LEGAL_MOVES)
        {
            int n = root.getOrderedLegalMoves().size();
            for (int i = 0; i < n; i++)
            {
                if (iterRoot.getQCount(i) > 0)
                {
                    root.setQValueTotal(i, root.getQValueTotal(i) + iterRoot.getQValue(i) * iterRoot.getQCount(i));
                    root.setQCount(i, root.getQCount(i) + iterRoot.getQCount(i));
                    break;
                }
            }
        }
        else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
        {
            int idx = DrawUnresolvedCardsIdxs.MOVE_IDX;
            root.setQValueTotal(idx, root.getQValueTotal(idx) + sample);
            root.setQCount(idx, root.getQCount(idx) + 1);
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                if (iterRoot.getQCount(i) > 0)
                {
                    root.setQValueTotal(i, root.getQValueTotal(i) + iterRoot.getQValue(i) * iterRoot.getQCount(i));
                    root.setQCount(i, root.getQCount(i) + iterRoot.getQCount(i));
                    break;
                }
            }
        }
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

            // use the original game view's hand for the final move
            HandView realHv = view.getHandView(currentPlayer);
            // but we need to use the actual game passed to chooseCardToPlay
            // the root's gameview IS the determinized view, so indices match
            Card card = realHv.getCard(cardIdx);

            if (card.isWild())
            {
                // use our real hand for best color — root view IS our omniscient view
                return Move.createMove(agent, cardIdx, bestColor(realHv));
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