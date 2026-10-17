#!/bin/sh
# Headless bot-vs-bot: ./bot-vs-bot.sh aragorn gandalf --games 100
# Rebuilds gemp-lotr-server first if any of its sources are newer than the jar. See BotVsBot.java for options.
set -e
cd "$(dirname "$0")"

if [ -z "$JAVA_HOME" ] && [ -d /opt/homebrew/opt/openjdk@21 ]; then
    JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi
export JAVA_HOME

jar=gemp-lotr-server/target/gemp-lotr-server-1.0-SNAPSHOT.jar
cp=gemp-lotr-server/target/runtime-classpath.txt
if [ ! -f "$jar" ] || [ -n "$(find gemp-lotr-server/src/main gemp-lotr-logic/src/main gemp-lotr-common/src/main -newer "$jar" -name '*.java' | head -1)" ]; then
    echo "building..." >&2
    mvn -B -q install -DskipTests -pl gemp-lotr-server -am >&2
fi
if [ ! -f "$cp" ] || [ gemp-lotr-server/pom.xml -nt "$cp" ]; then
    mvn -B -q dependency:build-classpath -pl gemp-lotr-server -Dmdep.outputFile="$PWD/$cp" >&2
fi

exec "$JAVA_HOME/bin/java" $JAVA_OPTS -cp "$jar:$(cat "$cp")" "${BOT_MAIN:-com.gempukku.lotro.bots.simulation.BotVsBot}" "$@"
