// package src.pas.uno.agents;


// // SYSTEM IMPORTS
// import edu.bu.pas.uno.Card;
// import edu.bu.pas.uno.Game.GameView;
// import edu.bu.pas.uno.Hand.HandView;
// import edu.bu.pas.uno.agents.MCTSAgent;
// import edu.bu.pas.uno.enums.Color;
// import edu.bu.pas.uno.enums.Value;
// import edu.bu.pas.uno.moves.Move;
// import edu.bu.pas.uno.tree.Node;

// import java.util.Random;
// import java.util.Set;

// import edu.bu.pas.uno.Deck;
// import edu.bu.pas.uno.DiscardPile;
// import edu.bu.pas.uno.DiscardPile.DiscardPileView;



// // JAVA PROJECT IMPORTS


// public class UnoMCTSAgent
//     extends MCTSAgent
// {

//     public static class MCTSNode
//         extends Node
//     {
//         public MCTSNode(final GameView game,
//                         final int logicalPlayerIdx,
//                         final Node parent)
//         {
//             super(game, logicalPlayerIdx, parent);
//         }

//         @Override
//         public Node getChild(final Move move)
//         {
//             return null;
//         }
//     }

//     public UnoMCTSAgent(final int playerIdx,
//                         final long maxThinkingTimeInMS)
//     {
//         super(playerIdx, maxThinkingTimeInMS);
//     }

//     /**
//      * A method to perform the MCTS search on the game tree
//      *
//      * @param   game            The {@link GameView} that should be the root of the game tree
//      * @param   drawnCardIdx    This will be non-null when this method is being called by the 
//      *                          <code>maybePlayDrawnCard</code> method of {@link Agent} and will
//      *                          be <code>null</code> when being called by <code>chooseCardToPlay</code>
//      *                          method of {@link Agent}
//      * @return  The {@link Node} of the root who'se q-values should now be populated and ready to argmax
//      */
//     @Override
//     public Node search(final GameView game,
//                        final Integer drawnCardIdx)
//     {
//         // TODO: implement me!
//         return null;
//     }

//     /**
//      * A method to argmax the Q values inside a {@link Node}
//      *
//      * @param   node            The {@link Node} who has populated q-values
//      * @return  The {@link Move} corresponding to whichever {@link Move} has the largest q-value. Note
//      *          that this can be <code>null</code> if you choose to not play the drawn card (you will
//      *          have to detect whether or not you are in that scenario by examining the @{link Node}'s state).
//      */
//     @Override
//     public Move argmaxQValues(final Node node)
//     {
//         // TODO: implement me!
//         return null;
//     }
// }

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

import java.util.Random;
import java.util.Set;

import edu.bu.pas.uno.Deck;
import edu.bu.pas.uno.DiscardPile;
import edu.bu.pas.uno.DiscardPile.DiscardPileView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// JAVA PROJECT IMPORTS

public class UnoMCTSAgent
        extends MCTSAgent {
    private static final int MAX_ROLLOUT_STEPS = 500;
    private static final int NUM_ITERATIONS = 500;

    private boolean rootWasDrawDecision = false;
    private boolean keepDrawnCardAtRoot = false;
    private int rootDrawnCardIdx = -1;

    private static class DummyAgent
            extends Agent {
        public DummyAgent(final int playerIdx) {
            super(playerIdx, 0L);
        }

        @Override
        public Move chooseCardToPlay(final GameView game) {
            return null;
        }

        @Override
        public Move maybePlayDrawnCard(final GameView game, final int drawnCardIdx) {
            return null;
        }
    }

    public static class MCTSNode
            extends Node {
        public MCTSNode(final GameView game,
                final int logicalPlayerIdx,
                final Node parent) {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move) {
            DummyAgent[] agents = buildDummyAgents(this.getGameView());

            // debugUnknowns("getChild-before-newGame", this.getGameView());
            Game childGame = new Game(this.getGameView(), agents);

            int currentLogical = childGame.getPlayerOrder().getCurrentLogicalPlayerIdx();
            Hand currentHand = childGame.getHand(currentLogical);
            NodeState state = this.getNodeState();

            try {
                if (state == NodeState.HAS_LEGAL_MOVES) {
                    childGame.resolveMove(move);
                } else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                    int total = childGame.getUnresolvedCards().total();
                    childGame.drawTotal(currentHand, total);
                    childGame.resolveMove(null);
                } else {
                    
                    int drawnIdx = childGame.drawCard(currentHand);
                    Card drawnCard = currentHand.getCard(drawnIdx);

                    if (move == null) {
                        childGame.resolveMove(null);
                    } else {
                        if (drawnCard.canBePlayedAsDrawCard(childGame)) {
                            childGame.resolveMove(move);
                        } else {
                            childGame.resolveMove(null);
                        }
                    }
                }
            } catch (NullPointerException e) {
               
            }

            GameView childView = childGame.getOmniscientView();
            
            
            Deck actualDrawPile = childGame.getDrawPile();
            Deck childDrawPile = childView.getDrawPile();

            for (int i = 0; i < childDrawPile.size(); i++) {
                childDrawPile.set(i, actualDrawPile.get(i));
            }

            int childLogicalPlayerIdx = childView.getPlayerOrder().getCurrentLogicalPlayerIdx();
            return new MCTSNode(childView, childLogicalPlayerIdx, this);
        }
    }

    public UnoMCTSAgent(final int playerIdx,
            final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    private static DummyAgent[] buildDummyAgents(final GameView game) {
        int n = game.getNumPlayers();
        DummyAgent[] agents = new DummyAgent[n];

        for (int logicalIdx = 0; logicalIdx < n; logicalIdx++) {
            int playerIdx = game.getPlayerOrder().getAgentIdx(logicalIdx);
            agents[logicalIdx] = new DummyAgent(playerIdx);
            agents[logicalIdx].setLogicalPlayerIdx(logicalIdx);
        }

        return agents;
    }

    private int findWinner(final Game sim) {
        int numPlayers = sim.getNumPlayers();
        int bestIdx = 0;
        int bestCount = Integer.MAX_VALUE;

        for (int i = 0; i < numPlayers; i++) {
            int count = sim.getHand(i).size();
            if (count < bestCount) {
                bestCount = count;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    private Color bestColor(final HandView hv) {
        Color[] colors = { Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW };
        int[] counts = new int[4];

        for (int i = 0; i < hv.size(); i++) {
            Card c = hv.getCard(i);
            for (int j = 0; j < 4; j++) {
                if (c.color() == colors[j]) {
                    counts[j]++;
                }
            }
        }

        int best = 0;
        for (int j = 1; j < 4; j++) {
            if (counts[j] > counts[best]) {
                best = j;
            }
        }

        return colors[best];
    }

    private float rollout(final GameView startView,
            final int myLogicalPlayerIdx,
            final DummyAgent[] agents) {

                // debugUnknowns("rollout-before-newGame", startView);
        Game stateCopy = new Game(startView, agents);
        int steps = 0;

        try {
            while (!stateCopy.isOver() && steps < MAX_ROLLOUT_STEPS) {
                int currentLogical = stateCopy.getPlayerOrder().getCurrentLogicalPlayerIdx();
                DummyAgent agent = agents[currentLogical];
                GameView simView = stateCopy.getOmniscientView();
                Hand currentHand = stateCopy.getHand(currentLogical);

                List<Card> cardList = new ArrayList<>();
                for (int i = 0; i < currentHand.size(); i++) {
                    cardList.add(currentHand.getCard(i));
                }

                HandView hv = new HandView(cardList);
                Set<Integer> legal = hv.getLegalMoves(simView);

                Move move;

                if (!stateCopy.getUnresolvedCards().isEmpty() && legal.isEmpty()) {
                    stateCopy.drawTotal(currentHand, stateCopy.getUnresolvedCards().total());
                    move = null;
                } else if (legal.isEmpty()) {
                    int drawnIdx = stateCopy.drawCard(currentHand);
                    Card drawn = currentHand.getCard(drawnIdx);

                    List<Card> newList = new ArrayList<>();
                    for (int i = 0; i < currentHand.size(); i++) {
                        newList.add(currentHand.getCard(i));
                    }

                    HandView hv2 = new HandView(newList);
                    Set<Integer> afterDraw = hv2.getLegalMoves(stateCopy.getOmniscientView());

                    if (afterDraw.contains(drawnIdx)) {
                        if (drawn.isWild()) {
                            move = Move.createMove(
                                    agent,
                                    drawnIdx,
                                    Color.getRandomColor(this.getRandom()));
                        } else {
                            move = Move.createMove(agent, drawnIdx);
                        }
                    } else {
                        move = null;
                    }
                } else {
                    List<Integer> legalList = new ArrayList<>(legal);
                    int cardIdx = legalList.get(this.getRandom().nextInt(legalList.size()));
                    Card card = currentHand.getCard(cardIdx);

                    if (card.isWild()) {
                        move = Move.createMove(
                                agent,
                                cardIdx,
                                Color.getRandomColor(this.getRandom()));
                    } else {
                        move = Move.createMove(agent, cardIdx);
                    }
                }

                stateCopy.resolveMove(move);
                steps++;
            }
        } catch (NullPointerException e) {
            return 0.5f;
        }

        int winner = findWinner(stateCopy);
        return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
    }

    private int ucbSelect(final MCTSNode node, final int numActions) {
        long stateCount = node.getStateCount();

        if (stateCount == 0) {
            return this.getRandom().nextInt(numActions);
        }

        int bestIdx = -1;
        double bestVal = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numActions; i++) {
            long actionCount = node.getQCount(i);
            double ucb;

            if (actionCount == 0) {
                ucb = Double.POSITIVE_INFINITY;
            } else {
                double avgQ = node.getQValue(i);
                double explore = Math.sqrt(2.0 * Math.log(stateCount) / actionCount);
                ucb = avgQ + explore;
            }

            if (ucb > bestVal) {
                bestVal = ucb;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    private float uct(final MCTSNode node,
            final int myLogicalPlayerIdx,
            final DummyAgent[] agents) {
        GameView view = node.getGameView();
        NodeState state = node.getNodeState();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Random rng = this.getRandom();
        DummyAgent agent = agents[currentPlayer];

        if (node.isTerminal()) {
            // debugUnknowns("uct-terminal-before-newGame", view);
            Game termGame = new Game(view, agents);
        
            int winner = findWinner(termGame);
            return (winner == myLogicalPlayerIdx) ? 1.0f : 0.0f;
        }

        float sample;

        if (state == NodeState.HAS_LEGAL_MOVES) {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            int numMoves = legalMoves.size();

            int i = ucbSelect(node, numMoves);
            int cardIdx = legalMoves.get(i);
            HandView hv = view.getHandView(currentPlayer);
            Card card = hv.getCard(cardIdx);

            Move move;
            if (card.isWild()) {
                move = Move.createMove(agent, cardIdx, Color.getRandomColor(rng));
            } else {
                move = Move.createMove(agent, cardIdx);
            }

            MCTSNode child = (MCTSNode) node.getChild(move);

            if (node.getQCount(i) == 0) {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, agents);
            } else {
                sample = uct(child, myLogicalPlayerIdx, agents);
            }

            node.setQValueTotal(i, node.getQValueTotal(i) + sample);
            node.setQCount(i, node.getQCount(i) + 1);
        } else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            int idx = DrawUnresolvedCardsIdxs.MOVE_IDX;
            MCTSNode child = (MCTSNode) node.getChild(null);

            if (node.getQCount(idx) == 0) {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, agents);
            } else {
                sample = uct(child, myLogicalPlayerIdx, agents);
            }

            node.setQValueTotal(idx, node.getQValueTotal(idx) + sample);
            node.setQCount(idx, node.getQCount(idx) + 1);
        } else {
            int actionIdx = ucbSelect(node, 2);
            Move move;

            if (actionIdx == DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX) {
                move = null;
            } else {
                int drawnIdx = view.getHandView(currentPlayer).size();
                move = Move.createMove(agent, drawnIdx, Color.getRandomColor(rng));
            }

            MCTSNode child = (MCTSNode) node.getChild(move);

            if (node.getQCount(actionIdx) == 0) {
                sample = rollout(child.getGameView(), myLogicalPlayerIdx, agents);
            } else {
                sample = uct(child, myLogicalPlayerIdx, agents);
            }

            node.setQValueTotal(actionIdx, node.getQValueTotal(actionIdx) + sample);
            node.setQCount(actionIdx, node.getQCount(actionIdx) + 1);
        }

        return sample;
    }

    private boolean[][] buildUnknownHandMask(final GameView game) {
        boolean[][] mask = new boolean[game.getNumPlayers()][];

        for (int p = 0; p < game.getNumPlayers(); p++) {
            HandView hv = game.getHandView(p);
            mask[p] = new boolean[hv.size()];

            for (int i = 0; i < hv.size(); i++) {
                Card c = hv.getCard(i);
                mask[p][i] = (c.color() == Color.UNKNOWN || c.value() == Value.UNKNOWN);
            }
        }

        return mask;
    }

    private boolean[] buildUnknownDrawPileMask(final GameView game) {
        Deck drawPile = game.getDrawPile();
        boolean[] mask = new boolean[drawPile.size()];

        for (int i = 0; i < drawPile.size(); i++) {
            Card c = drawPile.get(i);
            mask[i] = (c.color() == Color.UNKNOWN || c.value() == Value.UNKNOWN);
        }

        return mask;
    }

    private void determinizeHiddenInfoInPlace(final GameView game,
            final boolean[][] unknownHandMask,
            final boolean[] unknownDrawPileMask) {
        Deck remaining = new Deck();

        for (int p = 0; p < game.getNumPlayers(); p++) {
            HandView hv = game.getHandView(p);
            for (int i = 0; i < hv.size(); i++) {
                if (!unknownHandMask[p][i]) {
                    remaining.remove(hv.getCard(i));
                }
            }
        }

        Deck drawPile = game.getDrawPile();
        for (int i = 0; i < drawPile.size(); i++) {
            if (!unknownDrawPileMask[i]) {
                remaining.remove(drawPile.get(i));
            }
        }

        DiscardPile discard = new DiscardPile(game.getDiscardPile());
        for (Card c : discard.getPile()) {
            remaining.remove(c);
        }

        List<Card> sampled = new ArrayList<>(remaining);
        Collections.shuffle(sampled, this.getRandom());

        int next = 0;

        for (int p = 0; p < game.getNumPlayers(); p++) {
            HandView hv = game.getHandView(p);
            for (int i = 0; i < hv.size(); i++) {
                if (unknownHandMask[p][i]) {
                    hv.setCard(i, sampled.get(next));
                    next++;
                }
            }
        }

        for (int i = 0; i < drawPile.size(); i++) {
            if (unknownDrawPileMask[i]) {
                drawPile.set(i, sampled.get(next));
                next++;
            }
        }
    }

    private float evaluateKeepAtRoot(final GameView game,
            final int myLogicalIdx,
            final DummyAgent[] agents) {

        // debugUnknowns("keepRoot-before-newGame", game);


        Game keepGame = new Game(game, agents);
        keepGame.resolveMove(null);

        GameView keepView = keepGame.getOmniscientView();
        Deck actualDrawPile = keepGame.getDrawPile();
        Deck keepDrawPile = keepView.getDrawPile();

        for (int i = 0; i < keepDrawPile.size(); i++) {
            keepDrawPile.set(i, actualDrawPile.get(i));
        }

        return rollout(keepView, myLogicalIdx, agents);
    }

    private float evaluatePlayDrawnCardAtRoot(final GameView game,
            final int myLogicalIdx,
            final int drawnCardIdx,
            final DummyAgent[] agents) {
        
                Game playGame = new Game(game, agents);
        int currentLogical = playGame.getPlayerOrder().getCurrentLogicalPlayerIdx();
        DummyAgent agent = agents[currentLogical];
        Card drawn = playGame.getHand(currentLogical).getCard(drawnCardIdx);

        Move move;
        if (drawn.isWild()) {
            move = Move.createMove(agent, drawnCardIdx, Color.getRandomColor(this.getRandom()));
        } else {
            move = Move.createMove(agent, drawnCardIdx);
        }

        playGame.resolveMove(move);

        GameView playView = playGame.getOmniscientView();
        Deck actualDrawPile = playGame.getDrawPile();
        Deck playDrawPile = playView.getDrawPile();

        for (int i = 0; i < playDrawPile.size(); i++) {
            playDrawPile.set(i, actualDrawPile.get(i));
        }

        return rollout(playView, myLogicalIdx, agents);
    }

    /**
     * A method to perform the MCTS search on the game tree
     *
     * @param game         The {@link GameView} that should be the root of the game
     *                     tree
     * @param drawnCardIdx This will be non-null when this method is being called by
     *                     the
     *                     <code>maybePlayDrawnCard</code> method of {@link Agent}
     *                     and will
     *                     be <code>null</code> when being called by
     *                     <code>chooseCardToPlay</code>
     *                     method of {@link Agent}
     * @return The {@link Node} of the root who'se q-values should now be populated
     *         and ready to argmax
     */
    @Override
    public Node search(final GameView game,
            final Integer drawnCardIdx) {
        int myLogicalIdx = this.getLogicalPlayerIdx();
        int currentLogicalIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        DummyAgent[] agents = buildDummyAgents(game);

        boolean[][] unknownHandMask = buildUnknownHandMask(game);
        boolean[] unknownDrawPileMask = buildUnknownDrawPileMask(game);

        this.rootWasDrawDecision = (drawnCardIdx != null);
        this.keepDrawnCardAtRoot = false;
        this.rootDrawnCardIdx = (drawnCardIdx == null) ? -1 : drawnCardIdx;

        if (drawnCardIdx != null) {
            float keepTotal = 0.0f;
            float playTotal = 0.0f;

            for (int i = 0; !Thread.currentThread().isInterrupted() && i < NUM_ITERATIONS; i++) {
                determinizeHiddenInfoInPlace(game, unknownHandMask, unknownDrawPileMask);
            
                keepTotal += evaluateKeepAtRoot(game, myLogicalIdx, agents);
                playTotal += evaluatePlayDrawnCardAtRoot(game, myLogicalIdx, drawnCardIdx, agents);
            }

            this.keepDrawnCardAtRoot = keepTotal > playTotal;
            return new MCTSNode(game, currentLogicalIdx, null);
        }

        MCTSNode root = new MCTSNode(game, currentLogicalIdx, null);

        int iters = 0;
        while (!Thread.currentThread().isInterrupted() && iters < NUM_ITERATIONS) {
            determinizeHiddenInfoInPlace(game, unknownHandMask, unknownDrawPileMask);
            uct(root, myLogicalIdx, agents);
            iters++;
        }

        return root;
    }

    /**
     * A method to argmax the Q values inside a {@link Node}
     *
     * @param node The {@link Node} who has populated q-values
     * @return The {@link Move} corresponding to whichever {@link Move} has the
     *         largest q-value. Note
     *         that this can be <code>null</code> if you choose to not play the
     *         drawn card (you will
     *         have to detect whether or not you are in that scenario by examining
     *         the @{link Node}'s state).
     */
    @Override
    public Move argmaxQValues(final Node node) {
        if (node == null) {
            return null;
        }

        GameView view = node.getGameView();
        int currentPlayer = view.getPlayerOrder().getCurrentLogicalPlayerIdx();
        DummyAgent agent = buildDummyAgents(view)[currentPlayer];

        if (this.rootWasDrawDecision) {
            if (this.keepDrawnCardAtRoot) {
                return null;
            }

            HandView hv = view.getHandView(currentPlayer);
            Card card = hv.getCard(this.rootDrawnCardIdx);

            if (card.isWild()) {
                return Move.createMove(agent, this.rootDrawnCardIdx, bestColor(hv));
            } else {
                return Move.createMove(agent, this.rootDrawnCardIdx);
            }
        }

        NodeState state = node.getNodeState();

        if (state == NodeState.HAS_LEGAL_MOVES) {
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            HandView hv = view.getHandView(currentPlayer);

            int bestMoveIdx = -1;
            float bestQValue = Float.NEGATIVE_INFINITY;

            for (int i = 0; i < legalMoves.size(); i++) {
                if (node.getQCount(i) == 0) {
                    continue;
                }

                float qv = node.getQValue(i);
                if (qv > bestQValue) {
                    bestQValue = qv;
                    bestMoveIdx = i;
                }
            }

            if (bestMoveIdx < 0) {
                bestMoveIdx = 0;
            }

            int cardIdx = legalMoves.get(bestMoveIdx);
            Card card = hv.getCard(cardIdx);

            if (card.isWild()) {
                return Move.createMove(agent, cardIdx, bestColor(hv));
            } else {
                return Move.createMove(agent, cardIdx);
            }
        } else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            return null;
        } else {
            long playCount = node.getQCount(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX);
            long keepCount = node.getQCount(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX);

            float playQ = (playCount == 0)
                    ? Float.NEGATIVE_INFINITY
                    : node.getQValue(DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX);

            float keepQ = (keepCount == 0)
                    ? Float.NEGATIVE_INFINITY
                    : node.getQValue(DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX);

            if (keepQ > playQ) {
                return null;
            }

            HandView hv = view.getHandView(currentPlayer);
            int drawnIdx = hv.size() - 1;
            Card card = hv.getCard(drawnIdx);

            if (card.isWild()) {
                return Move.createMove(agent, drawnIdx, bestColor(hv));
            } else {
                return Move.createMove(agent, drawnIdx);
            }
        }
    }

    // private static int countUnknownHandCards(final GameView view) {
    //     int count = 0;
    //     for (int p = 0; p < view.getNumPlayers(); p++) {
    //         HandView hv = view.getHandView(p);
    //         for (int i = 0; i < hv.size(); i++) {
    //             Card c = hv.getCard(i);
    //             if (c.color() == Color.UNKNOWN || c.value() == Value.UNKNOWN) {
    //                 count++;
    //             }
    //         }
    //     }
    //     return count;
    // }

    // private static int countUnknownDrawPileCards(final GameView view) {
    //     int count = 0;
    //     Deck drawPile = view.getDrawPile();
    //     for (int i = 0; i < drawPile.size(); i++) {
    //         Card c = drawPile.get(i);
    //         if (c.color() == Color.UNKNOWN || c.value() == Value.UNKNOWN) {
    //             count++;
    //         }
    //     }
    //     return count;
    // }

    // private static void debugUnknowns(final String tag, final GameView view) {
    //     System.err.println(
    //             tag
    //                     + " unknownHand=" + countUnknownHandCards(view)
    //                     + " unknownDrawPile=" + countUnknownDrawPileCards(view)
    //                     + " currentPlayer=" + view.getPlayerOrder().getCurrentLogicalPlayerIdx());
    // }
}