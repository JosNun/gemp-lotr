package com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Play/use/pass for any card-action decision the phase-specific answerers do not take: the Archery and Assignment
 * phases, Regroup actions that are "Use" rather than "Play", and any prompt worded differently from the FotR
 * starters'. Without it those were answered at random -- about 4,000 times over 1,030 starter games.
 * <p>
 * One model across phases, so the decision's phase and the action's kind are inputs: {@link #CONTEXTS} and
 * {@link #KINDS} one-hot, read from the prompt and the action text since a stored step keeps no phase. The state
 * vector is padded to the skirmish length, because the game-state features hand skirmish and response decisions
 * four more numbers than the rest.
 */
public class AnyPhaseCardActionTrainer extends AbstractTrainer {
    private static final int STATE_LENGTH = 30;
    private static final List<String> CONTEXTS = List.of("fellowship", "shadow", "maneuver", "archery", "assignment",
            "skirmish", "regroup", "response");
    private static final List<String> KINDS = List.of("play", "use", "heal", "transfer");

    /**
     * -DanyPhase.rich (training) / an "any-phase-rich" file in the model directory (play): append what the per-phase
     * models see and this one did not -- an item's class and its holder, and whether a card can touch anyone in the
     * current skirmish (EXTRA numbers). The per-phase models beat this one however it was fit (BOT-NOTES.md).
     */
    public static final boolean RICH_TRAINING = Boolean.getBoolean("anyPhase.rich");
    /** -DanyPhase.perContext (training): one listwise model per {@link #CONTEXTS} entry, chosen by the prompt. */
    public static final boolean PER_CONTEXT_TRAINING = Boolean.getBoolean("anyPhase.perContext");
    /**
     * -DanyPhase.byPhase (with perContext): pick the model by the game's phase, read from the state's phase one-hot,
     * not by the prompt. "Choose action to play or Pass" -- skirmishes, mostly -- names no phase, so by the prompt it
     * went to the catch-all model: in October 2022's games the bot lost 2.7 points to people there.
     */
    public static final boolean BY_PHASE_TRAINING = Boolean.getBoolean("anyPhase.byPhase");
    static final int EXTRA = 10;
    /**
     * -DanyPhase.probe (training) / a "skirmish-probe" file in the model directory (play): skirmish options end with
     * what they did on a copy of the game, {@link EffectProbe}'s numbers. Other prompts' vectors are unchanged.
     */
    public static final boolean PROBE_TRAINING = Boolean.getBoolean("anyPhase.probe");
    /**
     * -DanyPhase.probeContexts=fellowship,maneuver,none,... (training) / a "probe-contexts" file listing them (play):
     * the options of these contexts' prompts end with all of {@link EffectProbe}'s numbers. "none" is the prompt naming no
     * phase, the skirmish's. Supersedes "skirmish-probe" for the contexts it names.
     */
    public static final java.util.Set<String> PROBE_CONTEXTS = names(System.getProperty("anyPhase.probeContexts"));
    /** -DanyPhase.probeLength: how many of the probe's numbers to fit on (22 leaves out the hand). */
    public static final int PROBE_LENGTH = Integer.getInteger("anyPhase.probeLength", EffectProbe.LENGTH);
    /**
     * -DanyPhase.refitFrom=&lt;a PerContextModel file&gt; -DanyPhase.refitContexts=a,b: refit only those contexts'
     * models from these steps, keeping every other context's as it was. -DanyPhase.refitSkirmishFrom=&lt;file&gt; is
     * the same with "none" alone.
     */
    private static final String REFIT_FROM = System.getProperty("anyPhase.refitFrom",
            System.getProperty("anyPhase.refitSkirmishFrom"));
    private static final java.util.Set<String> REFIT_CONTEXTS = System.getProperty("anyPhase.refitContexts") != null
            ? names(System.getProperty("anyPhase.refitContexts")) : java.util.Set.of("none");

    /**
     * -DanyPhase.cardText=&lt;vocabulary file&gt; (training) / a "card-text" file in the model directory (play): every
     * option ends with what its card's definition says it does ({@link CardTextFeatures}), after the probe's numbers so
     * a context fit without them reads the same vector as before. The file is built from the definitions if missing.
     */
    private static final String CARD_TEXT = System.getProperty("anyPhase.cardText");
    private static volatile List<String> cardTextVocab;

    static List<String> trainingCardText() {
        if (CARD_TEXT == null)
            return null;
        if (cardTextVocab == null)
            synchronized (AnyPhaseCardActionTrainer.class) {
                if (cardTextVocab == null)
                    try {
                        cardTextVocab = CardTextFeatures.vocabulary(java.nio.file.Path.of(CARD_TEXT));
                    } catch (java.io.IOException e) {
                        throw new IllegalStateException("anyPhase.cardText " + CARD_TEXT, e);
                    }
            }
        return cardTextVocab;
    }

    /**
     * -DanyPhase.response (training) / a "response-forecast" file in the model directory (play): each Fellowship-phase
     * option ends with what the Shadow player is likely to field after it -- the twilight it leaves them once the
     * fellowship moves, the minion strength a model of people's Shadow phases forecasts for that twilight at the next
     * site (scripts/shadow_response_fit.py: held-out R2 0.37; the Shadow player's hidden hand decides the rest), and
     * that strength less the fellowship's. Every card played and every companion (a twilight a move) feeds the
     * opponent's next Shadow phase; the options' numbers say so directly. Zeros in other contexts.
     */
    public static final boolean RESPONSE_TRAINING = Boolean.getBoolean("anyPhase.response");
    /**
     * -DanyPhase.opp (training, from steps recorded with HumanReplay --opp-block) / an "opp-features" file (play): every
     * option ends with what the deciding player has seen of its opponent this game (KnowledgeFeatures.opponent: its
     * public cards by type and culture), last of all -- for a model that answers a racer and a swarm differently.
     */
    public static final boolean OPP_TRAINING = Boolean.getBoolean("anyPhase.opp");
    static final int RESPONSE = 3;

    static double[] response(double[] state, String blueprintId, String actionText, String context) {
        double[] r = new double[RESPONSE];
        if (!"fellowship".equals(context) || CardFeatures.version() < 2)
            return r;
        java.util.function.ToDoubleFunction<String> f =
                n -> state[com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.index(n)];
        double cost = 0, companions = f.applyAsDouble("fpCompanions"), strength = f.applyAsDouble("fpCompanionStrength");
        if (blueprintId != null && !blueprintId.equals(CardFeatures.PASS) && actionText != null
                && actionText.toLowerCase().startsWith("play"))
            try {
                var bp = CardFeatures.blueprint(blueprintId);
                cost = Math.max(0, bp.getTwilightCost());
                if (bp.getCardType() == com.gempukku.lotro.common.CardType.COMPANION) {
                    companions++;
                    strength += bp.getStrength();
                }
            } catch (CardNotFoundException ignored) {
            }
        double site = f.applyAsDouble("site") + 1;
        // the move adds the companions and the next site's Shadow number, taken here as about its site number
        double twilight = f.applyAsDouble("twilight") + cost + companions + site;
        double forecast = -11.357797 + 1.121309 * twilight - 0.906065 * Math.max(0, twilight - 16) + 0.557772 * site
                - 5.185563 - 0.205025 * companions + 1.748264 * f.applyAsDouble("oppHand");
        r[0] = twilight;
        r[1] = forecast;
        r[2] = forecast - strength;
        return r;
    }

    /** Context names from a comma or line separated list; empty for null. */
    public static java.util.Set<String> names(String list) {
        java.util.Set<String> out = new java.util.LinkedHashSet<>();
        if (list != null)
            for (String n : list.split("[,\\s]+"))
                if (!n.isBlank()) {
                    String name = n.trim().toLowerCase();
                    if (!name.equals("none") && !CONTEXTS.contains(name))
                        throw new IllegalArgumentException("unknown context " + name + "; one of " + CONTEXTS + " or none");
                    out.add(name);
                }
        return out;
    }

    /** The context a prompt is scored in: the first of {@link #CONTEXTS} it names, or "none". */
    public static String contextName(String text) {
        String lower = text == null ? "" : text.toLowerCase();
        for (String c : CONTEXTS)
            if (lower.contains(c))
                return c;
        return "none";
    }

    private static int contextIndex(String name) {
        return name.equals("none") ? CONTEXTS.size() : CONTEXTS.indexOf(name);
    }

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        String[] actionIds = decision.getDecisionParameters().get("actionId");
        return decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE
                && actionIds != null && actionIds.length > 0;
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        if (step.decision.getDecisionType() != AwaitingDecisionType.CARD_ACTION_CHOICE)
            return false;
        String[] actionIds = step.decision.getDecisionParameters().get("actionId");
        return step.action instanceof CardActionChoiceAction && actionIds != null && actionIds.length > 0;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        if (model == null)
            return null;
        boolean rich = modelRegistry.isAnyPhaseRich();
        // all the probe's numbers where the model directory names the context; the first 20 for a skirmish-only model
        int probeLength = modelRegistry.getProbeContexts().contains(contextName(decision.getText()))
                ? modelRegistry.getProbeLength()
                : modelRegistry.isSkirmishProbe() && EffectProbe.applies(decision) ? EffectProbe.SKIRMISH_LENGTH : 0;
        boolean probe = probeLength > 0;
        List<String> text = modelRegistry.getCardText();
        String contextName = contextName(decision.getText());
        boolean forecast = modelRegistry.isResponseForecast();
        double[] opp = modelRegistry.isOppFeatures()
                ? com.gempukku.lotro.bots.rl.fotrstarters.KnowledgeFeatures.opponent(gameState, playerName) : null;
        double[] state = features.extractFeatures(gameState, decision, playerName);
        double[] context = context(decision.getText());
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] actionIds = decision.getDecisionParameters().get("actionId");
        String[] actionTexts = decision.getDecisionParameters().get("actionText");
        java.util.function.Function<String, com.gempukku.lotro.bots.BotPlayer> bots = p -> new FotrStarterBot(features, p,
                modelRegistry, null, 0.0, true);

        int best = -1; // -1 is pass
        double bestScore, passScore;
        double[] scores = new double[cardIds.length];
        Arrays.fill(scores, Double.NEGATIVE_INFINITY);
        try {
            passScore = bestScore = score(model, state, context, null, CardFeatures.PASS, 0, rich ? new double[EXTRA] : null,
                    probe ? EffectProbe.fit(EffectProbe.pass(gameState, playerName), probeLength) : null,
                    text == null ? null : new double[text.size()],
                    forecast ? response(state, null, null, contextName) : null, opp);
        } catch (CardNotFoundException e) {
            return null; // cannot happen: PASS is all zeros
        }
        for (int i = 0; i < cardIds.length; i++) {
            try {
                int cardId = Integer.parseInt(cardIds[i]);
                int wounds = 0;
                for (PhysicalCard card : gameState.getInPlay())
                    if (card.getCardId() == cardId)
                        wounds = gameState.getWounds(card);
                String bp = gameState.getBlueprintId(cardId);
                double[] did = null;
                if (probe) {
                    did = EffectProbe.fit(EffectProbe.probe(gameState, playerName, actionIds[i], bots), probeLength);
                }
                double s = score(model, state, context, actionTexts[i], bp, wounds,
                        rich ? liveExtra(gameState, cardId, bp) : null, did,
                        text == null ? null : CardTextFeatures.of(text, bp, actionTexts[i], contextName),
                        forecast ? response(state, bp, actionTexts[i], contextName) : null, opp);
                scores[i] = s;
                if (s > bestScore) {
                    bestScore = s;
                    best = i;
                }
            } catch (CardNotFoundException | RuntimeException ignored) {
                // a card the features cannot describe is not offered
            }
        }
        LAST_SCORES.set(append(scores, new double[]{passScore}));
        var net = modelRegistry.getBoardNet();
        if (net != null && com.gempukku.lotro.bots.rl.fotrstarters.BoardNet.applies(decision)) {
            // the board network mixed with these trees: (1 - w) * its log-probabilities + w * the trees' (a softmax over
            // the logits of their listwise probabilities); held-out July 66.5% against the trees' 64.5% and its 66.0%
            double[] lp = net.logProbs(gameState, decision, playerName, state, text);
            if (lp != null && lp.length == scores.length + 1) {
                double w = modelRegistry.getBoardNetTreeWeight();
                double[] t = new double[lp.length];
                for (int i = 0; i < scores.length; i++)
                    t[i] = scores[i] == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY : logit(scores[i]);
                t[scores.length] = logit(passScore);
                double max = Double.NEGATIVE_INFINITY, sum = 0;
                for (double v : t)
                    max = Math.max(max, v);
                for (double v : t)
                    sum += v == Double.NEGATIVE_INFINITY ? 0 : Math.exp(v - max);
                double norm = max + Math.log(sum);
                int pick = -1;
                double bestMix = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < lp.length; i++) {
                    double m = (1 - w) * lp[i] + w * (t[i] - norm);
                    if (m > bestMix) {
                        bestMix = m;
                        pick = i;
                    }
                }
                if (pick >= 0)
                    return pick == scores.length ? "" : String.valueOf(pick);
            }
        }
        if (modelRegistry.isPassByMass())
            return passByMass(passScore, scores);
        return best < 0 ? "" : String.valueOf(best);
    }

    private static double logit(double p) {
        double q = Math.min(1 - 1e-9, Math.max(1e-9, p));
        return Math.log(q / (1 - q));
    }

    /** The scores of the last decision this thread answered: each option's, then pass's (for BoardDump). */
    public static final ThreadLocal<double[]> LAST_SCORES = new ThreadLocal<>();

    /**
     * Pass only if passing is likelier than playing anything at all, else the likeliest play. The likeliest single
     * option passes whenever pass beats each play on its own though playing something is likelier -- pass is one
     * option against several -- and the bot's loss to people was passing where they played (skirmishes: Shadow -3.7 a
     * time, Free Peoples -1.7). Scores are the listwise model's probabilities: the logistic of each option's score.
     */
    static String passByMass(double pass, double[] plays) {
        double[] score = new double[plays.length + 1];
        score[0] = com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ReconcileChoiceTrainer.logit(pass);
        for (int i = 0; i < plays.length; i++)
            score[i + 1] = plays[i] == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY
                    : com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ReconcileChoiceTrainer.logit(plays[i]);
        double[] share = com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ReconcileChoiceTrainer.softmax(score);
        if (share[0] >= 0.5 || plays.length == 0)
            return "";
        int best = 1;
        for (int i = 2; i < share.length; i++)
            if (share[i] > share[best])
                best = i;
        return share[best] > 0 ? String.valueOf(best - 1) : "";
    }

    private double score(SoftClassifier<double[]> model, double[] state, double[] context, String actionText,
                         String blueprintId, int wounds, double[] extra, double[] probe, double[] text,
                         double[] response, double[] opp) throws CardNotFoundException {
        double[] probs = new double[2];
        model.predict(append(append(append(append(vector(state, context, actionText, blueprintId, wounds, extra), probe),
                text), response), opp), probs);
        return probs[1];
    }

    private static double[] append(double[] x, double[] more) {
        if (more == null)
            return x;
        double[] out = Arrays.copyOf(x, x.length + more.length);
        System.arraycopy(more, 0, out, x.length, more.length);
        return out;
    }

    /** The EXTRA numbers from the game, as the per-phase answerers compute theirs. */
    private static double[] liveExtra(GameState gs, int cardId, String blueprintId) {
        String holder = null;
        for (PhysicalCard c : gs.getInPlay())
            if (c.getCardId() == cardId && c.getAttachedTo() != null)
                holder = c.getAttachedTo().getBlueprintId();
        var skirmish = gs.getSkirmish();
        String fp = null;
        boolean fpCanExert = false, sourceIn = false;
        List<String> minions = new ArrayList<>();
        if (skirmish != null && skirmish.getFellowshipCharacter() != null) {
            var fc = skirmish.getFellowshipCharacter();
            fp = fc.getBlueprintId();
            fpCanExert = gs.getWounds(fc) + 1 < fc.getBlueprint().getVitality();
            sourceIn = fc.getCardId() == cardId;
            for (PhysicalCard m : skirmish.getShadowCharacters())
                minions.add(m.getBlueprintId());
        }
        return extra(blueprintId, holder, fp, fpCanExert, minions, sourceIn);
    }

    /** An item's hand weapon / armor / shield class and its holder (Ring-bearer, cost, strength, vitality); in a skirmish,
     * whether the card can affect someone in it, whether it is the companion skirmishing, and whether that one can exert. */
    static double[] extra(String blueprintId, String holder, String fp, boolean fpCanExert, List<String> minions,
                          boolean sourceIn) {
        double[] e = new double[EXTRA];
        try {
            if (holder != null) {
                double[] item = CardFeatures.getItemCardFeatures(blueprintId, holder, 0);
                System.arraycopy(item, 3, e, 0, 7);
            }
            if (fp != null) {
                List<Boolean> no = new ArrayList<>();
                for (String ignored : minions)
                    no.add(false);
                e[7] = CardFeatures.getSkirmishPlayCardFeatures(blueprintId, fp, fpCanExert, minions, no, sourceIn)[3];
                e[8] = sourceIn ? 1 : 0;
                e[9] = fpCanExert ? 1 : 0;
            }
        } catch (CardNotFoundException | RuntimeException ignored) {
            // a card the item or skirmish features cannot describe: zeros, as for pass
        }
        return e;
    }

    /** The training vector of a stored step, as {@link #getAnswer} builds it in a game. */
    public static double[] stepVector(LearningStep step, boolean rich) throws CardNotFoundException {
        CardActionChoiceAction action = (CardActionChoiceAction) step.action;
        String blueprint = action.getSourceBlueprint() != null ? action.getSourceBlueprint() : CardFeatures.PASS;
        double[] extra = null;
        if (rich)
            extra = action.getSourceBlueprint() == null ? new double[EXTRA] : extra(blueprint, action.getHolderBlueprint(),
                    action.getSkirmishingFpBlueprintId(), action.getRemainingVitalityOnSkirmishFp() > 1,
                    action.getSkirmishingMinionBlueprintIds() != null ? action.getSkirmishingMinionBlueprintIds() : List.of(),
                    action.isSourceInSkirmish());
        double[] x = vector(step.state, context(step.decision.getText()),
                action.getSourceBlueprint() != null ? action.getActionText() : null, blueprint, action.getWoundsOnSource(),
                extra);
        if (PROBE_CONTEXTS.contains(contextName(step.decision.getText())))
            x = append(x, EffectProbe.fit(action.getProbe(), PROBE_LENGTH));
        else if (PROBE_TRAINING && EffectProbe.applies(step.decision))
            x = append(x, EffectProbe.fit(action.getProbe(), EffectProbe.SKIRMISH_LENGTH));
        List<String> text = trainingCardText();
        if (text != null)
            x = append(x, CardTextFeatures.of(text, action.getSourceBlueprint(),
                    action.getSourceBlueprint() != null ? action.getActionText() : null, contextName(step.decision.getText())));
        if (RESPONSE_TRAINING)
            x = append(x, response(step.state, action.getSourceBlueprint() != null ? action.getSourceBlueprint() : null,
                    action.getSourceBlueprint() != null ? action.getActionText() : null, contextName(step.decision.getText())));
        if (OPP_TRAINING)
            x = append(x, step.opp != null ? step.opp
                    : new double[com.gempukku.lotro.bots.rl.fotrstarters.KnowledgeFeatures.opponentLength()]);
        return x;
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            try {
                data.add(new LabeledPoint(step.reward > 0 ? 1 : 0, stepVector(step, RICH_TRAINING), step.group));
            } catch (CardNotFoundException ignored) {
            }
        }
        return data;
    }

    private static double[] vector(double[] state, double[] context, String actionText, String blueprintId, int wounds)
            throws CardNotFoundException {
        return vector(state, context, actionText, blueprintId, wounds, null);
    }

    private static double[] vector(double[] state, double[] context, String actionText, String blueprintId, int wounds,
                                   double[] extra) throws CardNotFoundException {
        double[] card = CardFeatures.getCardFeatures(blueprintId, wounds);
        if (extra != null) {
            card = Arrays.copyOf(card, card.length + extra.length);
            System.arraycopy(extra, 0, card, card.length - extra.length, extra.length);
        }
        double[] kind = new double[KINDS.size()];
        if (actionText != null)
            for (int k = 0; k < KINDS.size(); k++)
                if (actionText.toLowerCase().startsWith(KINDS.get(k)))
                    kind[k] = 1;
        double[] option = new double[context.length + kind.length + card.length];
        System.arraycopy(context, 0, option, 0, context.length);
        System.arraycopy(kind, 0, option, context.length, kind.length);
        System.arraycopy(card, 0, option, context.length + kind.length, card.length);
        // set 1 states are 26 or 30 long by decision type; set 2's are one length already
        double[] fixedState = CardFeatures.version() >= 2 ? state : Arrays.copyOf(state, STATE_LENGTH);
        return CardFeatures.join(fixedState, option, blueprintId);
    }

    @Override
    public SoftClassifier<double[]> train(List<LearningStep> steps) {
        List<LabeledPoint> points = extractTrainingData(steps);
        if (REFIT_FROM != null)
            return refitContexts(points);
        if (!PER_CONTEXT_TRAINING || points.isEmpty())
            return trainWithPoints(points);
        // one model per phase context: the per-phase models' specialisation, with every option of a decision compared
        int offset = CardFeatures.version() >= 2 ? com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.length()
                : STATE_LENGTH;
        int[] phases = null;
        if (BY_PHASE_TRAINING && CardFeatures.version() >= 2) {
            var all = com.gempukku.lotro.common.Phase.values();
            phases = new int[all.length];
            for (int i = 0; i < all.length; i++)
                phases[i] = com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.index("phase_" + all[i].name());
        }
        int contexts = phases != null ? phases.length : CONTEXTS.size();
        List<List<LabeledPoint>> parts = new ArrayList<>();
        for (int c = 0; c <= contexts; c++)
            parts.add(new ArrayList<>());
        for (LabeledPoint p : points)
            parts.get(PerContextModel.contextOf(p.x(), offset, CONTEXTS.size(), phases)).add(p);
        List<SoftClassifier<double[]>> models = new ArrayList<>();
        for (List<LabeledPoint> part : parts)
            models.add(part.size() < 200 || part.stream().mapToInt(LabeledPoint::y).distinct().count() < 2 ? null
                    : trainWithPoints(part));
        SoftClassifier<double[]> all = trainWithPoints(points); // for a context with too few rows of its own
        return new PerContextModel(models, all, offset, CONTEXTS.size(), phases);
    }

    /** The base model with the {@link #REFIT_CONTEXTS} models fit anew, each on {@code points}' decisions of its context. */
    private SoftClassifier<double[]> refitContexts(List<LabeledPoint> points) {
        PerContextModel model = load(REFIT_FROM);
        if (model.phases != null)
            throw new IllegalStateException("anyPhase.refitFrom: a by-phase model has no slots by prompt");
        int offset = CardFeatures.version() >= 2 ? com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.length()
                : STATE_LENGTH;
        // the contexts fit side by side (-Dtrain.threads, 3 by default): one after another, seven of them overran the
        // two hours a background job gets
        java.util.Map<String, java.util.concurrent.Future<SoftClassifier<double[]>>> fits = new java.util.LinkedHashMap<>();
        var pool = java.util.concurrent.Executors.newFixedThreadPool(Integer.getInteger("train.threads", 3));
        try {
            for (String name : REFIT_CONTEXTS) {
                int c = contextIndex(name);
                List<LabeledPoint> part = new ArrayList<>();
                for (LabeledPoint p : points)
                    if (PerContextModel.contextOf(p.x(), offset, CONTEXTS.size(), null) == c)
                        part.add(p);
                if (part.size() < 200) { // too few to fit on (one style's games): the base's model stays
                    System.err.printf("  keeping %-12s %8d rows%n", name, part.size());
                    continue;
                }
                System.out.printf("  refitting %-12s %8d rows%n", name, part.size());
                fits.put(name, pool.submit(() -> trainWithPoints(part)));
            }
            for (var e : fits.entrySet())
                model = model.with(contextIndex(e.getKey()), e.getValue().get());
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("anyPhase.refitContexts", e);
        } finally {
            pool.shutdownNow();
        }
        return model;
    }

    static PerContextModel load(String file) {
        try (var in = new java.io.ObjectInputStream(new java.io.FileInputStream(file))) {
            return (PerContextModel) in.readObject();
        } catch (java.io.IOException | ClassNotFoundException | ClassCastException e) {
            throw new IllegalStateException("no per-context model in " + file, e);
        }
    }

    /**
     * Swap context models between two model files: {@code <base> <donor> <contexts> <out>} writes base with the named
     * contexts' models taken from donor. Which phases a change helps is then a match, not a refit apiece.
     */
    public static void main(String[] args) throws java.io.IOException {
        if (args.length != 4)
            throw new IllegalArgumentException("usage: <base.model> <donor.model> <context,context> <out.model>");
        PerContextModel base = load(args[0]), donor = load(args[1]);
        for (String name : names(args[2]))
            base = base.with(contextIndex(name), donor.models.get(contextIndex(name)));
        try (var out = new java.io.ObjectOutputStream(new java.io.FileOutputStream(args[3]))) {
            out.writeObject(base);
        }
    }

    /** Dispatches each option to its decision's context model; the context is one-hot right after the state. */
    static final class PerContextModel implements SoftClassifier<double[]>, java.io.Serializable {
        @java.io.Serial
        private static final long serialVersionUID = 1L;
        private final List<SoftClassifier<double[]>> models;
        private final SoftClassifier<double[]> fallback;
        private final int offset, contexts;
        /** State indices of the phase one-hot, when the models are by game phase; null when by the prompt. */
        private final int[] phases;

        PerContextModel(List<SoftClassifier<double[]>> models, SoftClassifier<double[]> fallback, int offset, int contexts,
                        int[] phases) {
            this.models = models;
            this.fallback = fallback;
            this.offset = offset;
            this.contexts = contexts;
            this.phases = phases;
        }

        /** This model with context {@code c}'s model replaced. */
        PerContextModel with(int c, SoftClassifier<double[]> model) {
            List<SoftClassifier<double[]>> copy = new ArrayList<>(models);
            copy.set(c, model);
            return new PerContextModel(copy, fallback, offset, contexts, phases);
        }

        static int contextOf(double[] x, int offset, int contexts, int[] phases) {
            if (phases != null) {
                for (int c = 0; c < phases.length; c++)
                    if (x[phases[c]] == 1)
                        return c;
                return phases.length;
            }
            for (int c = 0; c < contexts; c++)
                if (x[offset + c] == 1)
                    return c;
            return contexts;
        }

        private SoftClassifier<double[]> pick(double[] x) {
            SoftClassifier<double[]> m = models.get(contextOf(x, offset, contexts, phases));
            return m != null ? m : fallback;
        }

        @Override
        public int predict(double[] x) {
            return pick(x).predict(x);
        }

        @Override
        public int predict(double[] x, double[] posteriori) {
            return pick(x).predict(x, posteriori);
        }
    }

    private static double[] context(String text) {
        String lower = text.toLowerCase();
        double[] c = new double[CONTEXTS.size()];
        for (int i = 0; i < CONTEXTS.size(); i++)
            if (lower.contains(CONTEXTS.get(i))) {
                c[i] = 1;
                break;
            }
        return c;
    }
}
