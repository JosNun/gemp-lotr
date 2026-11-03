package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSONObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * A dashboard feed (scripts/feed.py): one JSON record a line appended to labels/feed/&lt;stream&gt;.jsonl, stamped
 * with "t"; the dashboard works out how to draw a stream from its records' fields. Numbers and bot, deck or month
 * names only: the page is served on the LAN.
 */
public final class Feed {
    static final Path DIR = Path.of(System.getProperty("user.home"), "dev", "gemp-replays", "labels", "feed");

    private Feed() {
    }

    public static void emit(String stream, Map<String, ?> fields) {
        if (stream == null || !stream.matches("[A-Za-z0-9._-]+"))
            return;
        JSONObject rec = new JSONObject();
        rec.put("t", System.currentTimeMillis() / 1000.0);
        rec.putAll(fields);
        try {
            Files.createDirectories(DIR);
            synchronized (Feed.class) {
                Files.writeString(DIR.resolve(stream + ".jsonl"), rec + "\n", StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            }
        } catch (IOException ignored) {
            // a feed is for watching; a job does not stop for it
        }
    }
}
