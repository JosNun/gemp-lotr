package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.game.DefaultUserFeedback;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import sun.misc.Unsafe;

import java.io.File;
import java.lang.reflect.*;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.jar.JarFile;

/**
 * Copies a live game at a decision point, so a lookahead rollout can play on from a copy instead of replaying the game
 * from its seed (BotTeach --clone). Everything the game owns is copied; everything every game in the JVM shares is not:
 * the card library and formats (every blueprint and all that hangs off them), whatever the engine's classes hold in
 * static fields, and the decks the harness passed in. That line is drawn by identity, not by type -- the shared objects
 * are found once by walking from those roots -- because a game's actions and modifiers point straight at pieces of
 * blueprints (filters, effect appenders), and a copied piece would no longer be the object a blueprint compares
 * against. It is exactly what a replay shares too: every replayed game gets the same library and the same statics.
 * <p>
 * How each object is copied:
 * <ul>
 * <li>Ordinary classes, the engine's and the JDK's plain containers (ArrayList, LinkedList, Stack, TreeMap, EnumMap,
 *     Random...): allocated without a constructor and every field copied, via Unsafe. Their structure, iteration order
 *     and the Random's seed come across bit for bit.</li>
 * <li>HashMap / HashSet / LinkedHashMap / LinkedHashSet: rebuilt through their public API with the original's table
 *     size and load factor, entries inserted in the original's iteration order, so a key whose hash is the same in
 *     the copy (strings, enums, shared objects, cards -- which hash by their creation id) lands where it was and the
 *     copy iterates in the same order now and after further inserts. Raw copying would keep each node's cached hash,
 *     wrong for any key that hashes by identity.</li>
 * <li>Lambdas (hidden classes, whose final fields cannot be written) and records: rebuilt through their constructor
 *     from copies of what they captured; shared as they are if everything captured is shared.</li>
 * <li>Objects with no instance fields at all (non-capturing lambdas, stateless filters, plain lock objects): shared.</li>
 * </ul>
 * Any other JDK class is refused with the path to it, rather than guessed at.
 * <p>
 * Uses sun.misc.Unsafe (jdk.unsupported, open on JDK 21 without flags): it is how fields of JDK collections are read
 * and final fields written without --add-opens.
 */
public final class GameCopier {
    private static final Unsafe U;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            U = (Unsafe) f.get(null);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    // ---- what is shared ----

    /** Objects every game shares: reachable from the library, the formats, the decks, or a static field. */
    private final Set<Object> frozen = Collections.newSetFromMap(new IdentityHashMap<>());
    /** The classes with at least one shared instance. */
    private final Map<Class<?>, Boolean> frozenClasses = new java.util.concurrent.ConcurrentHashMap<>();
    /** Stands for a null replacement in a copy's identity map, so a miss costs one lookup. */
    private static final Object DROPPED = new Object();
    private static final ClassValue<Boolean> LEAF = new ClassValue<>() {
        @Override
        protected Boolean computeValue(Class<?> c) {
            return isLeaf(c);
        }
    };
    private final ReentrantReadWriteLock frozenLock = new ReentrantReadWriteLock();
    /** Engine classes not initialised when last looked; their statics are walked once they are. */
    private final List<Class<?>> uninitialised = new ArrayList<>();

    public GameCopier(LotroCardBlueprintLibrary library, Object... sharedRoots) {
        List<Class<?>> classes = new ArrayList<>();
        for (Class<?> anchor : List.of(LotroCardBlueprintLibrary.class, com.gempukku.lotro.common.Phase.class))
            classes.addAll(classesBesides(anchor));
        uninitialised.addAll(classes);
        frozenLock.writeLock().lock();
        try {
            freeze(library);
            for (Object root : sharedRoots)
                freeze(root);
            freezeNewStatics();
        } finally {
            frozenLock.writeLock().unlock();
        }
    }

    /** How many objects are shared; for the curious. */
    public int sharedCount() {
        return frozen.size();
    }

    /** Engine classes whose statics are still to be walked, once something initialises them. */
    public int classesPending() {
        synchronized (uninitialised) {
            return uninitialised.size();
        }
    }

    /** Adds more objects every game may share, e.g. decks loaded after the copier was made. */
    public void share(Object root) {
        frozenLock.writeLock().lock();
        try {
            freeze(root);
        } finally {
            frozenLock.writeLock().unlock();
        }
    }

    private static List<Class<?>> classesBesides(Class<?> anchor) {
        List<Class<?>> out = new ArrayList<>();
        try {
            URI location = anchor.getProtectionDomain().getCodeSource().getLocation().toURI();
            File file = new File(location);
            List<String> names = new ArrayList<>();
            if (file.isDirectory()) {
                Path root = file.toPath();
                try (var walk = Files.walk(root)) {
                    walk.filter(p -> p.toString().endsWith(".class"))
                            .forEach(p -> names.add(root.relativize(p).toString()));
                }
            } else {
                try (JarFile jar = new JarFile(file)) {
                    jar.stream().map(java.util.zip.ZipEntry::getName).filter(n -> n.endsWith(".class"))
                            .forEach(names::add);
                }
            }
            ClassLoader loader = anchor.getClassLoader();
            for (String n : names) {
                if (n.endsWith("module-info.class") || n.endsWith("package-info.class"))
                    continue;
                String cn = n.substring(0, n.length() - 6).replace('/', '.').replace(File.separatorChar, '.');
                try {
                    out.add(Class.forName(cn, false, loader));
                } catch (Throwable ignored) {
                    // a class that cannot even load holds nothing a game can reach
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException("cannot list the classes beside " + anchor, e);
        }
        return out;
    }

    /** Walks the statics of engine classes initialised since last time. Call with the write lock held. */
    @SuppressWarnings("deprecation")
    private void freezeNewStatics() {
        for (Iterator<Class<?>> it = uninitialised.iterator(); it.hasNext(); ) {
            Class<?> c = it.next();
            boolean pending;
            try {
                pending = U.shouldBeInitialized(c);
            } catch (Throwable t) {
                it.remove();
                continue;
            }
            if (pending)
                continue;
            it.remove();
            Field[] fields;
            try {
                fields = c.getDeclaredFields();
            } catch (Throwable t) {
                continue;
            }
            for (Field f : fields) {
                if (!Modifier.isStatic(f.getModifiers()) || f.getType().isPrimitive())
                    continue;
                try {
                    freeze(U.getObject(U.staticFieldBase(f), U.staticFieldOffset(f)));
                } catch (Throwable ignored) {
                    // a static we cannot read is one no game reaches through us either
                }
            }
        }
    }

    private boolean anyNewlyInitialised() {
        synchronized (uninitialised) {
            for (Class<?> c : uninitialised) {
                try {
                    if (!U.shouldBeInitialized(c))
                        return true;
                } catch (Throwable t) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Marks everything reachable from root as shared. Call with the write lock held. */
    private void freeze(Object root) {
        ArrayDeque<Object> stack = new ArrayDeque<>();
        if (root != null)
            stack.push(root);
        while (!stack.isEmpty()) {
            Object o = stack.pop();
            if (o == null || LEAF.get(o.getClass()) || !frozen.add(o))
                continue;
            Class<?> c = o.getClass();
            frozenClasses.put(c, Boolean.TRUE);
            try {
                if (c.isArray()) {
                    if (!c.getComponentType().isPrimitive())
                        for (Object e : (Object[]) o)
                            if (e != null)
                                stack.push(e);
                } else if (Proxy.isProxyClass(c)) {
                    stack.push(Proxy.getInvocationHandler(o));
                } else if (isJdk(c) && !c.isHidden()) {
                    if (o instanceof Map<?, ?> m) {
                        for (Map.Entry<?, ?> e : m.entrySet()) {
                            if (e.getKey() != null)
                                stack.push(e.getKey());
                            if (e.getValue() != null)
                                stack.push(e.getValue());
                        }
                    } else if (o instanceof Collection<?> col) {
                        for (Object e : col)
                            if (e != null)
                                stack.push(e);
                    } else if (RAW_JDK.contains(c.getName())) {
                        for (Object e : plan(c).referents(o))
                            if (e != null)
                                stack.push(e);
                    }
                    // other JDK objects (files, locks, threads...) are shared and not looked into
                } else if (!isOpaque(c)) {
                    for (Object e : plan(c).referents(o))
                        if (e != null)
                            stack.push(e);
                }
            } catch (RuntimeException e) {
                // a structure that will not be walked is still shared; only what lies beyond it is missed
            }
        }
    }

    // ---- the copy ----

    /**
     * A copy of {@code game}, at the decision it is waiting on, sharing nothing mutable with it. Listeners the harness
     * attached (game results, game-state listeners) are not carried over; the card-creation callback and the deck shuffler
     * are dropped.
     */
    public DefaultLotroGame copy(DefaultLotroGame game) {
        if (anyNewlyInitialised()) {
            frozenLock.writeLock().lock();
            try {
                synchronized (uninitialised) {
                    freezeNewStatics();
                }
            } finally {
                frozenLock.writeLock().unlock();
            }
        }
        frozenLock.readLock().lock();
        try {
            Session s = new Session();
            try {
                s.seed(field(game, DefaultLotroGame.class, "_gameResultListeners"), new HashSet<>());
                GameState state = game.getGameState();
                s.seed(field(state, GameState.class, "_gameStateListeners"), new HashSet<>());
                Object callback = field(state, GameState.class, "_cardCreationCallback");
                if (callback != null)
                    s.seed(callback, null);
                // a harness's hook, like the callback: the copy gets its own if it needs one (HumanReplay)
                Object shuffler = field(state, GameState.class, "_deckShuffler");
                if (shuffler != null)
                    s.seed(shuffler, null);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
            DefaultLotroGame copy = s.copyGraph(game);
            if (histogram != null)
                s.copies.forEach((k, v) -> {
                    if (k != v && v != DROPPED)
                        histogram.merge(k.getClass().getName().replaceAll("[$][$]Lambda.*", "-lambda"), 1,
                                Integer::sum);
                });
            if (copy.getGameState() == state(game) || copy.getUserFeedback() == game.getUserFeedback())
                throw new IllegalStateException("the copy shares its state with the original");
            return copy;
        } finally {
            frozenLock.readLock().unlock();
        }
    }

    private static GameState state(DefaultLotroGame game) {
        return game.getGameState();
    }

    /** The copied game's feedback, which holds the decision it waits on. */
    public static DefaultUserFeedback feedback(DefaultLotroGame game) {
        return (DefaultUserFeedback) game.getUserFeedback();
    }

    private static Object field(Object o, Class<?> c, String name) throws ReflectiveOperationException {
        Field f = c.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(o);
    }

    /** Diagnostics: when set, every copy counts what it copied, by class, into it. */
    volatile Map<String, Integer> histogram;

    /** Diagnostics: copies print the path to whatever they refuse when this is set. */
    static volatile boolean tracePaths = System.getenv("CLONE_TRACE") != null;

    private final class Session {
        private final IdentityHashMap<Object, Object> copies = new IdentityHashMap<>(4096);
        private final ArrayDeque<Object> fill = new ArrayDeque<>(); // orig, copy pairs: fields or elements to fill
        private final ArrayDeque<Object> hashFill = new ArrayDeque<>(); // orig, copy pairs of hash containers
        private final Set<Object> constructing = Collections.newSetFromMap(new IdentityHashMap<>());
        private final IdentityHashMap<Object, String> path = tracePaths ? new IdentityHashMap<>() : null;

        void seed(Object original, Object replacement) {
            if (original != null)
                copies.put(original, replacement == null ? DROPPED : replacement);
        }

        @SuppressWarnings("unchecked")
        <T> T copyGraph(T root) {
            if (path != null)
                path.put(root, root.getClass().getSimpleName());
            T result = (T) copy(root, null, null);
            while (true) {
                drain();
                if (hashFill.isEmpty())
                    break;
                // hash containers last: their keys' hashCodes must see filled objects. Take this round's
                // containers, copy everything they hold, fill all that brought in, then insert.
                List<Object> batch = new ArrayList<>(hashFill);
                hashFill.clear();
                List<Object[]> contents = new ArrayList<>(batch.size() / 2);
                for (int i = 0; i < batch.size(); i += 2)
                    contents.add(copyContents(batch.get(i)));
                drain();
                for (int i = 0; i < batch.size(); i += 2)
                    insert(batch.get(i + 1), contents.get(i / 2));
            }
            return result;
        }

        private void drain() {
            while (!fill.isEmpty()) {
                Object orig = fill.poll(), copy = fill.poll();
                Class<?> c = orig.getClass();
                if (c.isArray()) {
                    Object[] from = (Object[]) orig, to = (Object[]) copy;
                    for (int i = 0; i < from.length; i++)
                        to[i] = copy(from[i], orig, "[" + i + "]");
                } else {
                    plan(c).fill(this, orig, copy);
                }
            }
        }

        Object copy(Object o, Object parent, String via) {
            if (o == null)
                return null;
            Class<?> c = o.getClass();
            if (LEAF.get(c))
                return o;
            Object done = copies.get(o);
            if (done != null)
                return done == DROPPED ? null : done;
            // the shared set is large and lookups into it miss the cache; most game objects are of classes with
            // no shared instance at all, which a small map answers
            if (frozenClasses.containsKey(c) && frozen.contains(o))
                return o;
            if (path != null)
                path.put(o, (parent == null ? "" : path.getOrDefault(parent, "?")) + "." + via
                        + "<" + c.getName() + ">");
            if (c.isArray()) {
                if (c.getComponentType().isPrimitive()) {
                    Object a = primitiveArrayClone(o);
                    copies.put(o, a);
                    return a;
                }
                Object a = Array.newInstance(c.getComponentType(), Array.getLength(o));
                copies.put(o, a);
                fill.add(o);
                fill.add(a);
                return a;
            }
            Plan plan = plan(c);
            return plan.copy(this, o);
        }

        String where(Object o) {
            if (path == null)
                return "(set CLONE_TRACE=1 for the path)";
            return "at " + path.getOrDefault(o, "?");
        }

        Object[] copyContents(Object orig) {
            if (orig instanceof Map<?, ?> m) {
                Object[] kv = new Object[m.size() * 2];
                int i = 0;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    kv[i] = copy(e.getKey(), orig, "key");
                    kv[i + 1] = copy(e.getValue(), orig, "value");
                    i += 2;
                }
                return kv;
            }
            Collection<?> col = (Collection<?>) orig;
            Object[] es = new Object[col.size()];
            int i = 0;
            for (Object e : col)
                es[i++] = copy(e, orig, "element");
            return es;
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        void insert(Object copy, Object[] contents) {
            if (copy instanceof Map m) {
                for (int i = 0; i < contents.length; i += 2)
                    m.put(contents[i], contents[i + 1]);
                if (m.size() * 2 != contents.length)
                    throw new IllegalStateException("keys collided rebuilding a " + copy.getClass().getName());
            } else {
                Collection col = (Collection) copy;
                Collections.addAll(col, contents);
                if (col.size() != contents.length)
                    throw new IllegalStateException("elements collided rebuilding a " + copy.getClass().getName());
            }
        }
    }

    private static Object primitiveArrayClone(Object a) {
        if (a instanceof int[] x) return x.clone();
        if (a instanceof long[] x) return x.clone();
        if (a instanceof boolean[] x) return x.clone();
        if (a instanceof double[] x) return x.clone();
        if (a instanceof byte[] x) return x.clone();
        if (a instanceof char[] x) return x.clone();
        if (a instanceof float[] x) return x.clone();
        if (a instanceof short[] x) return x.clone();
        throw new IllegalArgumentException(a.getClass().getName());
    }

    // ---- classification ----

    private static boolean isJdk(Class<?> c) {
        ClassLoader l = c.getClassLoader();
        return l == null || l == ClassLoader.getPlatformClassLoader();
    }

    /** Immutable values and singletons: never copied, never walked. */
    private static boolean isLeaf(Class<?> c) {
        return c == String.class || c == Integer.class || c == Long.class || c == Boolean.class
                || c == Double.class || c == Float.class || c == Short.class || c == Byte.class
                || c == Character.class || c == Class.class || Enum.class.isAssignableFrom(c)
                || c == java.math.BigInteger.class || c == java.math.BigDecimal.class || c == UUID.class
                || c == File.class || Path.class.isAssignableFrom(c) || c == java.util.regex.Pattern.class
                || c == Locale.class || java.nio.charset.Charset.class.isAssignableFrom(c)
                || c == Object.class;
    }

    /** Non-JDK classes that are not looked into while walking what is shared (and are shared if met). */
    private static boolean isOpaque(Class<?> c) {
        String n = c.getName();
        return n.startsWith("org.apache.logging.") || n.startsWith("org.slf4j.");
    }

    /** JDK classes copied field by field: no hashing inside, no native state. Their inner nodes too. */
    private static final Set<String> RAW_JDK = Set.of(
            "java.util.ArrayList", "java.util.LinkedList", "java.util.LinkedList$Node", "java.util.ArrayDeque",
            "java.util.Vector", "java.util.Stack", "java.util.TreeMap", "java.util.TreeMap$Entry",
            "java.util.TreeSet", "java.util.EnumMap", "java.util.EnumSet", "java.util.RegularEnumSet",
            "java.util.JumboEnumSet",
            "java.util.Random", "java.util.concurrent.atomic.AtomicInteger",
            "java.util.concurrent.atomic.AtomicLong", "java.util.concurrent.atomic.AtomicBoolean",
            "java.util.concurrent.atomic.AtomicReference", "java.util.BitSet", "java.lang.StringBuilder",
            "java.util.Arrays$ArrayList", "java.util.Collections$EmptyList",
            // views: their fields point at the list they view, which is copied with them, so the copy views the copy
            "java.util.AbstractList$SubList", "java.util.AbstractList$RandomAccessSubList",
            "java.util.Optional", "java.util.AbstractMap$SimpleEntry",
            "java.util.AbstractMap$SimpleImmutableEntry", "java.util.KeyValueHolder",
            "java.util.Collections$UnmodifiableCollection", "java.util.Collections$UnmodifiableList",
            "java.util.Collections$UnmodifiableRandomAccessList", "java.util.Collections$UnmodifiableSet",
            "java.util.Collections$UnmodifiableSortedSet", "java.util.Collections$UnmodifiableMap",
            "java.util.Collections$UnmodifiableSortedMap", "java.util.Collections$SingletonList",
            "java.util.Collections$SingletonSet", "java.util.Collections$SingletonMap",
            "java.util.ImmutableCollections$ListN", "java.util.ImmutableCollections$List12",
            "java.util.ImmutableCollections$Set12", "java.util.ImmutableCollections$Map1",
            "java.util.ArrayList$SubList", "java.util.ArrayList$Itr", "java.util.ArrayList$ListItr",
            "java.util.LinkedList$ListItr", "java.util.AbstractMap$1", "java.util.AbstractMap$2",
            "java.util.HashMap$KeySet", "java.util.HashMap$Values", "java.util.HashMap$EntrySet",
            "java.util.LinkedHashMap$LinkedKeySet", "java.util.LinkedHashMap$LinkedValues",
            "java.util.LinkedHashMap$LinkedEntrySet", "java.util.TreeMap$KeySet", "java.util.TreeMap$Values",
            "java.util.TreeMap$EntrySet");

    private static final Set<String> HASHED_IMMUTABLE = Set.of(
            "java.util.ImmutableCollections$SetN", "java.util.ImmutableCollections$MapN");

    private static final ClassValue<Plan> PLANS = new ClassValue<>() {
        @Override
        protected Plan computeValue(Class<?> c) {
            return makePlan(c);
        }
    };

    private static Plan plan(Class<?> c) {
        return PLANS.get(c);
    }

    private static Plan makePlan(Class<?> c) {
        if (c.isHidden())
            return new ConstructedPlan(c, false);
        if (c.isRecord())
            return new ConstructedPlan(c, true);
        if (c == HashMap.class || c == LinkedHashMap.class || c == HashSet.class || c == LinkedHashSet.class)
            return new HashPlan(c);
        if (HASHED_IMMUTABLE.contains(c.getName()))
            return new ImmutableHashPlan();
        if (isJdk(c) && !RAW_JDK.contains(c.getName())) {
            boolean stateless = true; // Collections.emptySet() and the like: nothing to copy, and == keeps working
            for (Class<?> k = c; k != null && k != Object.class; k = k.getSuperclass())
                stateless &= !hasInstanceFields(k);
            return stateless ? new SharedPlan(null) : new RefusedPlan(c, "a JDK class with no known way to copy it");
        }
        for (Class<?> s = c.getSuperclass(); s != null && s != Object.class; s = s.getSuperclass())
            if (isJdk(s) && !RAW_JDK.contains(s.getName()) && !s.getName().startsWith("java.util.Abstract")
                    && hasInstanceFields(s))
                return new RefusedPlan(c, "extends " + s.getName());
        RawPlan raw = new RawPlan(c);
        return raw.objectOffsets.length == 0 && raw.primitiveOffsets.length == 0 ? new SharedPlan(raw) : raw;
    }

    private static boolean hasInstanceFields(Class<?> c) {
        for (Field f : c.getDeclaredFields())
            if (!Modifier.isStatic(f.getModifiers()))
                return true;
        return false;
    }

    private abstract static class Plan {
        abstract Object copy(Session s, Object o);

        void fill(Session s, Object orig, Object copy) {
            throw new IllegalStateException();
        }

        /** What an object of this class points to, for walking what is shared. */
        abstract List<Object> referents(Object o);
    }

    /** Field by field, through Unsafe. */
    private static class RawPlan extends Plan {
        final Class<?> type;
        final long[] objectOffsets;
        final String[] objectNames;
        final long[] primitiveOffsets;
        final char[] primitiveKinds;

        RawPlan(Class<?> c) {
            type = c;
            List<Field> objects = new ArrayList<>(), primitives = new ArrayList<>();
            for (Class<?> k = c; k != null && k != Object.class; k = k.getSuperclass())
                for (Field f : k.getDeclaredFields()) {
                    if (Modifier.isStatic(f.getModifiers()))
                        continue;
                    (f.getType().isPrimitive() ? primitives : objects).add(f);
                }
            objectOffsets = new long[objects.size()];
            objectNames = new String[objects.size()];
            for (int i = 0; i < objects.size(); i++) {
                objectOffsets[i] = U.objectFieldOffset(objects.get(i));
                objectNames[i] = objects.get(i).getName();
            }
            primitiveOffsets = new long[primitives.size()];
            primitiveKinds = new char[primitives.size()];
            for (int i = 0; i < primitives.size(); i++) {
                primitiveOffsets[i] = U.objectFieldOffset(primitives.get(i));
                Class<?> t = primitives.get(i).getType();
                primitiveKinds[i] = t == long.class ? 'J' : t == double.class ? 'D' : t == int.class ? 'I'
                        : t == float.class ? 'F' : t == boolean.class ? 'Z' : t == byte.class ? 'B'
                        : t == char.class ? 'C' : 'S';
            }
        }

        @Override
        Object copy(Session s, Object o) {
            Object c;
            try {
                c = U.allocateInstance(type);
            } catch (InstantiationException e) {
                throw new IllegalStateException(e);
            }
            s.copies.put(o, c);
            s.fill.add(o);
            s.fill.add(c);
            return c;
        }

        @Override
        void fill(Session s, Object orig, Object copy) {
            for (int i = 0; i < primitiveOffsets.length; i++) {
                long off = primitiveOffsets[i];
                switch (primitiveKinds[i]) {
                    case 'J' -> U.putLong(copy, off, U.getLong(orig, off));
                    case 'D' -> U.putDouble(copy, off, U.getDouble(orig, off));
                    case 'I' -> U.putInt(copy, off, U.getInt(orig, off));
                    case 'F' -> U.putFloat(copy, off, U.getFloat(orig, off));
                    case 'Z' -> U.putBoolean(copy, off, U.getBoolean(orig, off));
                    case 'B' -> U.putByte(copy, off, U.getByte(orig, off));
                    case 'C' -> U.putChar(copy, off, U.getChar(orig, off));
                    default -> U.putShort(copy, off, U.getShort(orig, off));
                }
            }
            for (int i = 0; i < objectOffsets.length; i++) {
                long off = objectOffsets[i];
                U.putObject(copy, off, s.copy(U.getObject(orig, off), orig, objectNames[i]));
            }
            U.fullFence();
        }

        @Override
        List<Object> referents(Object o) {
            List<Object> out = new ArrayList<>(objectOffsets.length);
            for (long off : objectOffsets)
                out.add(U.getObject(o, off));
            return out;
        }
    }

    /** No instance state at all: the object is its class, and sharing it keeps == working. */
    private static final class SharedPlan extends Plan {
        final RawPlan raw;

        SharedPlan(RawPlan raw) {
            this.raw = raw;
        }

        @Override
        Object copy(Session s, Object o) {
            return o;
        }

        @Override
        List<Object> referents(Object o) {
            return List.of();
        }
    }

    private static final class RefusedPlan extends Plan {
        final Class<?> type;
        final String why;

        RefusedPlan(Class<?> type, String why) {
            this.type = type;
            this.why = why;
        }

        @Override
        Object copy(Session s, Object o) {
            throw new UnsupportedOperationException("cannot copy " + type.getName() + " (" + why + ") "
                    + s.where(o));
        }

        @Override
        List<Object> referents(Object o) {
            return List.of();
        }
    }

    /** Lambdas and records: final fields that only a constructor can set. */
    private static final class ConstructedPlan extends Plan {
        final Class<?> type;
        final Constructor<?> ctor;
        final Field[] fields;
        final String refused;
        final boolean stateless;

        ConstructedPlan(Class<?> c, boolean record) {
            type = c;
            stateless = !hasInstanceFields(c);
            Constructor<?> k = null;
            Field[] fs = new Field[0];
            String why = null;
            try {
                if (record) {
                    RecordComponent[] rc = c.getRecordComponents();
                    Class<?>[] types = new Class<?>[rc.length];
                    fs = new Field[rc.length];
                    for (int i = 0; i < rc.length; i++) {
                        types[i] = rc[i].getType();
                        fs[i] = c.getDeclaredField(rc[i].getName());
                    }
                    k = c.getDeclaredConstructor(types);
                } else {
                    List<Field> list = new ArrayList<>();
                    for (Field f : c.getDeclaredFields())
                        if (!Modifier.isStatic(f.getModifiers()))
                            list.add(f);
                    // lambda forms capture into arg$1..arg$n, which the one constructor takes in that order
                    list.sort(Comparator.comparingInt(f -> lambdaIndex(f.getName())));
                    fs = list.toArray(new Field[0]);
                    Constructor<?>[] all = c.getDeclaredConstructors();
                    if (all.length != 1)
                        throw new IllegalStateException(all.length + " constructors");
                    k = all[0];
                    if (k.getParameterCount() != fs.length)
                        throw new IllegalStateException("constructor takes " + k.getParameterCount() + ", "
                                + fs.length + " fields");
                }
                for (Field f : fs)
                    f.setAccessible(true);
                k.setAccessible(true);
            } catch (Exception e) {
                why = e.toString();
            }
            ctor = k;
            fields = fs;
            refused = why;
        }

        private static int lambdaIndex(String name) {
            int at = name.lastIndexOf('$');
            try {
                return Integer.parseInt(name.substring(at + 1));
            } catch (NumberFormatException e) {
                return Integer.MAX_VALUE;
            }
        }

        @Override
        Object copy(Session s, Object o) {
            if (stateless)
                return o; // a non-capturing lambda: one instance per call site, and == must keep working
            if (refused != null) {
                throw new UnsupportedOperationException("cannot copy " + type.getName() + ": " + refused + " "
                        + s.where(o));
            }
            if (!s.constructing.add(o))
                throw new UnsupportedOperationException("a cycle runs back through " + type.getName()
                        + ", which only a constructor can build " + s.where(o));
            try {
                Object[] args = new Object[fields.length];
                boolean same = true;
                for (int i = 0; i < fields.length; i++) {
                    Object v = fields[i].get(o);
                    args[i] = s.copy(v, o, fields[i].getName());
                    same &= args[i] == v;
                }
                Object c = same ? o : ctor.newInstance(args);
                s.copies.put(o, c);
                return c;
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("rebuilding " + type.getName(), e);
            } finally {
                s.constructing.remove(o);
            }
        }

        @Override
        List<Object> referents(Object o) {
            List<Object> out = new ArrayList<>(fields.length);
            if (refused == null)
                for (Field f : fields)
                    try {
                        out.add(f.get(o));
                    } catch (IllegalAccessException e) {
                        throw new IllegalStateException(e);
                    }
            return out;
        }
    }

    /** HashMap and friends, rebuilt with the original's capacity so the copy iterates the same way. */
    private static final class HashPlan extends Plan {
        static final long TABLE, THRESHOLD, LOAD_FACTOR, ACCESS_ORDER, SET_MAP;

        static {
            try {
                TABLE = U.objectFieldOffset(HashMap.class.getDeclaredField("table"));
                THRESHOLD = U.objectFieldOffset(HashMap.class.getDeclaredField("threshold"));
                LOAD_FACTOR = U.objectFieldOffset(HashMap.class.getDeclaredField("loadFactor"));
                ACCESS_ORDER = U.objectFieldOffset(LinkedHashMap.class.getDeclaredField("accessOrder"));
                SET_MAP = U.objectFieldOffset(HashSet.class.getDeclaredField("map"));
            } catch (NoSuchFieldException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        final Class<?> type;

        HashPlan(Class<?> type) {
            this.type = type;
        }

        /** The capacity a new map needs to have the same table as this one, or 0 for the default. */
        private static int capacity(HashMap<?, ?> m) {
            Object[] table = (Object[]) U.getObject(m, TABLE);
            if (table != null)
                return table.length;
            return U.getInt(m, THRESHOLD); // not yet allocated: threshold holds the initial capacity (0 = default)
        }

        @Override
        Object copy(Session s, Object o) {
            Object c;
            if (type == HashMap.class || type == LinkedHashMap.class) {
                HashMap<?, ?> m = (HashMap<?, ?>) o;
                int cap = capacity(m);
                float lf = U.getFloat(m, LOAD_FACTOR);
                if (type == LinkedHashMap.class)
                    c = new LinkedHashMap<>(cap == 0 ? 16 : cap, lf, U.getBoolean(m, ACCESS_ORDER));
                else
                    c = cap == 0 ? new HashMap<>(16, lf) : new HashMap<>(cap, lf);
            } else {
                HashMap<?, ?> inner = (HashMap<?, ?>) U.getObject(o, SET_MAP);
                int cap = capacity(inner);
                float lf = U.getFloat(inner, LOAD_FACTOR);
                c = type == LinkedHashSet.class ? new LinkedHashSet<>(cap == 0 ? 16 : cap, lf)
                        : new HashSet<>(cap == 0 ? 16 : cap, lf);
            }
            s.copies.put(o, c);
            s.hashFill.add(o);
            s.hashFill.add(c);
            return c;
        }

        @Override
        List<Object> referents(Object o) {
            throw new IllegalStateException("walked through the API");
        }
    }

    /** Set.of / Map.of with more than two entries: their layout hangs on hashes. Shared if what they hold is. */
    private static final class ImmutableHashPlan extends Plan {
        @Override
        Object copy(Session s, Object o) {
            boolean same = true;
            if (o instanceof Map<?, ?> m) {
                List<Map.Entry<Object, Object>> entries = new ArrayList<>();
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    Object k = s.copy(e.getKey(), o, "key"), v = s.copy(e.getValue(), o, "value");
                    same &= k == e.getKey() && v == e.getValue();
                    entries.add(Map.entry(k, v));
                }
                if (same) {
                    s.copies.put(o, o);
                    return o;
                }
                s.drain();
                @SuppressWarnings("unchecked")
                Map.Entry<Object, Object>[] arr = entries.toArray(new Map.Entry[0]);
                Object c = Map.ofEntries(arr);
                s.copies.put(o, c);
                return c;
            }
            List<Object> elements = new ArrayList<>();
            for (Object e : (Set<?>) o) {
                Object k = s.copy(e, o, "element");
                same &= k == e;
                elements.add(k);
            }
            if (same) {
                s.copies.put(o, o);
                return o;
            }
            s.drain();
            Object c = Set.of(elements.toArray());
            s.copies.put(o, c);
            return c;
        }

        @Override
        List<Object> referents(Object o) {
            throw new IllegalStateException("walked through the API");
        }
    }
}
