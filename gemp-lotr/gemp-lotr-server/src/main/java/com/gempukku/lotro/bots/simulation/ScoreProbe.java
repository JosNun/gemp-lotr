package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer;
import com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction;
import java.io.*; import java.lang.reflect.Method; import java.nio.file.*; import java.util.*;

/**
 * Holds a model set's card-action model (AnyPhaseCardActionTrainer) against human decisions written by HumanReplay
 * --steps. {@code <models> <blueprint> <jsonl>...}: for the decisions where that card was offered, how often people
 * took it and how often the model ranks it first or above passing. {@code <models> all <jsonl>...}: top-1 agreement
 * with people over every card-action decision. Run with the jar's classpath from gemp-lotr/.
 */
public class ScoreProbe {
    public static void main(String[] a) throws Exception {
        PrintStream out = System.out; System.setOut(new PrintStream(OutputStream.nullOutputStream()));
        BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        var reg = BotVsBot.loadModels(Path.of(a[0]));
        CardFeatures.setVersion(reg.getFeatureVersion());
        var model = reg.getModel(AnyPhaseCardActionTrainer.class);
        Method vector = AnyPhaseCardActionTrainer.class.getDeclaredMethod("vector", double[].class, double[].class, String.class, String.class, int.class);
        Method context = AnyPhaseCardActionTrainer.class.getDeclaredMethod("context", String.class);
        vector.setAccessible(true); context.setAccessible(true);
        String target = a[1];
        if (target.equals("all")) { agreement(a, model, vector, context, out); System.exit(0); }
        int groups = 0, humanTook = 0, modelTop = 0, modelBeatsPass = 0; double sumP = 0, sumPass = 0;
        String key = null; List<LearningStep> group = new ArrayList<>();
        List<String> files = Arrays.asList(a).subList(2, a.length);
        for (String f : files) try (BufferedReader r = Files.newBufferedReader(Path.of(f))) {
            for (String line; (line = r.readLine()) != null; ) {
                if (!line.contains("CardActionSelectionDecision")) continue;
                LearningStep s = LearningStep.fromJson(line);
                String k = line.substring(line.indexOf("\"decision\"")); // one decision's rows are written together
                if (!k.equals(key)) {
                    int[] res = eval(group, target, model, vector, context);
                    if (res != null) { groups++; humanTook += res[0]; modelTop += res[1]; modelBeatsPass += res[2]; }
                    group = new ArrayList<>(); key = k;
                }
                group.add(s);
            }
        }
        out.printf("%s in %s: %d decisions; people took it %.1f%%; model ranks it first %.1f%%, above pass %.1f%%%n",
                target, a[0], groups, 100.0 * humanTook / groups, 100.0 * modelTop / groups, 100.0 * modelBeatsPass / groups);
        System.exit(0);
    }

    static void agreement(String[] a, smile.classification.SoftClassifier<double[]> model, Method vector, Method context, PrintStream out) throws Exception {
        long n = 0, agree = 0, nPlay = 0, agreePlay = 0, nPass = 0, agreePass = 0, botPass = 0;
        String key = null; List<LearningStep> group = new ArrayList<>();
        for (String f : Arrays.asList(a).subList(2, a.length)) try (BufferedReader r = Files.newBufferedReader(Path.of(f))) {
            for (String line; ; ) {
                line = r.readLine();
                String k = line == null ? null : line.contains("CardActionSelectionDecision") ? line.substring(line.indexOf("\"decision\"")) : "";
                if (line != null && k.isEmpty()) continue;
                if (line == null || !k.equals(key)) {
                    if (!group.isEmpty()) {
                        String human = null, bot = null; double best = -1;
                        for (LearningStep s : group) {
                            var c = (CardActionChoiceAction) s.action;
                            String bp = c.getSourceBlueprint() != null ? c.getSourceBlueprint() : CardFeatures.PASS;
                            String id = bp + "|" + c.getActionText();
                            if (s.reward > 0) human = id;
                            try {
                                double[] x = (double[]) vector.invoke(null, s.state, context.invoke(null, s.decision.getText()),
                                        c.getSourceBlueprint() != null ? c.getActionText() : null, bp, c.getWoundsOnSource());
                                double[] p = new double[2]; model.predict(x, p);
                                if (p[1] > best) { best = p[1]; bot = id; }
                            } catch (Exception e) { }
                        }
                        if (human != null && bot != null && group.size() > 1) {
                            n++; boolean ok = human.equals(bot); if (ok) agree++;
                            if (human.startsWith(CardFeatures.PASS + "|")) { nPass++; if (ok) agreePass++; } else { nPlay++; if (ok) agreePlay++; }
                            if (bot.startsWith(CardFeatures.PASS + "|")) botPass++;
                        }
                    }
                    group = new ArrayList<>(); key = k;
                    if (line == null) break;
                }
                try { LearningStep s = LearningStep.fromJson(line); if (s.action instanceof CardActionChoiceAction) group.add(s); } catch (Exception e) { }
            }
        }
        out.printf("%-34s %7d decisions: agree %.1f%%; people played %.1f%% (agree %.1f%%), passed (agree %.1f%%); model passes %.1f%%%n",
                a[0], n, 100.0 * agree / n, 100.0 * nPlay / n, 100.0 * agreePlay / nPlay, 100.0 * agreePass / nPass, 100.0 * botPass / n);
    }

    static int[] eval(List<LearningStep> g, String target, smile.classification.SoftClassifier<double[]> model, Method vector, Method context) throws Exception {
        boolean offered = false, took = false; double best = -1, tScore = -1, pass = -1; String bestBp = null;
        for (LearningStep s : g) {
            if (!(s.action instanceof CardActionChoiceAction c)) return null;
            String bp = c.getSourceBlueprint() != null ? c.getSourceBlueprint() : CardFeatures.PASS;
            double[] x;
            try {
                x = (double[]) vector.invoke(null, s.state, context.invoke(null, s.decision.getText()),
                        c.getSourceBlueprint() != null ? c.getActionText() : null, bp, c.getWoundsOnSource());
            } catch (Exception e) { continue; }
            double[] p = new double[2]; model.predict(x, p);
            if (bp.equals(target)) { offered = true; took |= s.reward > 0; tScore = Math.max(tScore, p[1]); }
            if (bp.equals(CardFeatures.PASS)) pass = p[1];
            if (p[1] > best) { best = p[1]; bestBp = bp; }
        }
        if (!offered) return null;
        return new int[]{took ? 1 : 0, target.equals(bestBp) ? 1 : 0, tScore > pass ? 1 : 0};
    }
}
