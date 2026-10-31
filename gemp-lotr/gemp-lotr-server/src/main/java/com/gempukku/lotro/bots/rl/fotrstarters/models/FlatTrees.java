package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.base.cart.InternalNode;
import smile.base.cart.Node;
import smile.base.cart.OrdinalNode;
import smile.base.cart.RegressionNode;
import smile.classification.GradientTreeBoost;
import smile.regression.RegressionTree;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Smile's regression trees as flat arrays, walked on the feature vector itself. Smile predicts through a Tuple: every
 * prediction rebuilds the predictors by column name (formula.x) and every split reads its feature through the schema --
 * the tree models were 30% of a pure-network bot's play-out time, a quarter of it in OrdinalNode.predict alone (JFR,
 * 2026-10-30). Same splits, same leaves, same sums: {@code FlatTrees parity <model dir>} checks every tree model in a
 * directory against Smile. {@code -Dtrees.flat=false} goes back to Smile.
 */
public final class FlatTrees {
    public static final boolean ENABLED = !"false".equals(System.getProperty("trees.flat"));

    /** One tree: node i splits on feature[i] (x <= threshold[i] goes to left[i]), or is a leaf when feature[i] < 0. */
    static final class Tree {
        final int[] feature, left, right;
        final double[] threshold, leaf;

        Tree(int[] feature, int[] left, int[] right, double[] threshold, double[] leaf) {
            this.feature = feature;
            this.left = left;
            this.right = right;
            this.threshold = threshold;
            this.leaf = leaf;
        }

        double predict(double[] x) {
            int i = 0;
            while (feature[i] >= 0)
                i = x[feature[i]] <= threshold[i] ? left[i] : right[i];
            return leaf[i];
        }
    }

    private static final Field TRUE_CHILD, FALSE_CHILD, FEATURE, VALUE, B, SHRINKAGE, K;

    static {
        try {
            TRUE_CHILD = InternalNode.class.getDeclaredField("trueChild");
            FALSE_CHILD = InternalNode.class.getDeclaredField("falseChild");
            FEATURE = InternalNode.class.getDeclaredField("feature");
            VALUE = OrdinalNode.class.getDeclaredField("value");
            B = GradientTreeBoost.class.getDeclaredField("b");
            SHRINKAGE = GradientTreeBoost.class.getDeclaredField("shrinkage");
            K = GradientTreeBoost.class.getDeclaredField("k");
            for (Field f : new Field[]{TRUE_CHILD, FALSE_CHILD, FEATURE, VALUE, B, SHRINKAGE, K})
                f.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /** The tree as arrays; throws if it has a split this cannot represent (a nominal one -- the bot's are all ordinal). */
    static Tree flatten(RegressionTree tree) {
        List<Node> nodes = new ArrayList<>();
        nodes.add(tree.root());
        List<int[]> links = new ArrayList<>();
        int[] feature, left, right;
        double[] threshold, leaf;
        try {
            for (int i = 0; i < nodes.size(); i++) { // breadth first, children appended as met
                Node n = nodes.get(i);
                if (n instanceof OrdinalNode o) {
                    nodes.add((Node) TRUE_CHILD.get(o));
                    int l = nodes.size() - 1;
                    nodes.add((Node) FALSE_CHILD.get(o));
                    links.add(new int[]{l, nodes.size() - 1});
                } else if (n instanceof RegressionNode) {
                    links.add(null);
                } else
                    throw new IllegalArgumentException("unsupported node " + n.getClass().getSimpleName());
            }
            int size = nodes.size();
            feature = new int[size];
            left = new int[size];
            right = new int[size];
            threshold = new double[size];
            leaf = new double[size];
            for (int i = 0; i < size; i++) {
                Node n = nodes.get(i);
                if (n instanceof OrdinalNode o) {
                    feature[i] = FEATURE.getInt(o);
                    threshold[i] = VALUE.getDouble(o);
                    left[i] = links.get(i)[0];
                    right[i] = links.get(i)[1];
                } else {
                    feature[i] = -1;
                    leaf[i] = ((RegressionNode) n).output();
                }
            }
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        return new Tree(feature, left, right, threshold, leaf);
    }

    static Tree[] flatten(RegressionTree[] trees) {
        Tree[] out = new Tree[trees.length];
        for (int i = 0; i < trees.length; i++)
            out[i] = flatten(trees[i]);
        return out;
    }

    /** A binary GradientTreeBoost: intercept, shrinkage and trees. */
    static final class Boost {
        final double b, shrinkage;
        final Tree[] trees;

        Boost(GradientTreeBoost model) {
            try {
                if (K.getInt(model) != 2)
                    throw new IllegalArgumentException("not a binary model");
                b = B.getDouble(model);
                shrinkage = SHRINKAGE.getDouble(model);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
            trees = flatten(model.trees());
        }

        /** As GradientTreeBoost.predict(Tuple, double[]) for two classes. */
        int predict(double[] x, double[] posteriori) {
            double y = b;
            for (Tree t : trees)
                y += shrinkage * t.predict(x);
            if (posteriori != null) {
                posteriori[0] = 1.0 / (1.0 + Math.exp(2 * y));
                posteriori[1] = 1.0 - posteriori[0];
            }
            return y > 0 ? 1 : 0;
        }
    }

    static double sum(Tree[] trees, double shrinkage, double[] x) {
        double s = 0;
        for (Tree t : trees)
            s += shrinkage * t.predict(x);
        return s;
    }

    /** Every TreeModel and ListwiseTreeModel reachable from {@code o} through the bot's own classes and lists. */
    private static void collect(Object o, java.util.Set<Object> seen, List<Object> out) {
        if (o == null || !seen.add(o))
            return;
        if (o instanceof TreeModel || o instanceof ListwiseTreeModel) {
            out.add(o);
            return;
        }
        if (o instanceof Iterable<?> it) {
            for (Object x : it)
                collect(x, seen, out);
            return;
        }
        if (o instanceof java.util.Map<?, ?> map) {
            for (Object x : map.values())
                collect(x, seen, out);
            return;
        }
        if (o instanceof Object[] arr) {
            for (Object x : arr)
                collect(x, seen, out);
            return;
        }
        if (!o.getClass().getName().startsWith("com.gempukku"))
            return;
        for (Class<?> c = o.getClass(); c != null && c != Object.class; c = c.getSuperclass())
            for (Field f : c.getDeclaredFields()) {
                if (java.lang.reflect.Modifier.isStatic(f.getModifiers()) || f.getType().isPrimitive())
                    continue;
                try {
                    f.setAccessible(true);
                    collect(f.get(o), seen, out);
                } catch (ReflectiveOperationException | RuntimeException ignored) {
                }
            }
    }

    /**
     * {@code parity <model dir>}: every tree model there, Smile against the flat trees on random vectors (and the
     * zero vector): the largest difference in the posterior, and the share of different classes.
     */
    public static void main(String[] args) throws Exception {
        java.nio.file.Path dir = java.nio.file.Path.of(args[1]);
        java.util.Random rnd = new java.util.Random(1);
        int models = 0;
        double worst = 0;
        long differ = 0, total = 0;
        try (var files = java.nio.file.Files.list(dir)) {
            for (java.nio.file.Path f : files.sorted().toList()) {
                if (!f.toString().endsWith(".model"))
                    continue;
                Object m;
                try (var in = new java.io.ObjectInputStream(new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(f)))) {
                    m = in.readObject();
                } catch (Exception e) {
                    continue;
                }
                List<Object> inner = new ArrayList<>();
                collect(m, java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>()), inner);
                for (Object o : inner) {
                    if (!(o instanceof TreeModel) && !(o instanceof ListwiseTreeModel))
                        continue;
                    int width = o instanceof TreeModel t ? t.width() : ((ListwiseTreeModel) o).width();
                    models++;
                    for (int i = 0; i < 2000; i++) {
                        double[] x = new double[width];
                        if (i > 0)
                            for (int j = 0; j < width; j++)
                                x[j] = rnd.nextInt(4) == 0 ? 0 : rnd.nextGaussian() * (1 + rnd.nextInt(10));
                        double[] a = new double[2], b = new double[2];
                        int ca, cb;
                        if (o instanceof TreeModel t) {
                            ca = t.predictSmile(x, a);
                            cb = t.predict(x, b);
                        } else {
                            ListwiseTreeModel l = (ListwiseTreeModel) o;
                            ca = l.predictSmile(x, a);
                            cb = l.predict(x, b);
                        }
                        worst = Math.max(worst, Math.abs(a[1] - b[1]));
                        differ += ca != cb ? 1 : 0;
                        total++;
                    }
                }
            }
        }
        System.out.printf("%d tree models, %d predictions: posterior max |diff| %.2e, class differs on %d%n", models,
                total, worst, differ);
        System.exit(0);
    }
}
