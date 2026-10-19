#!/bin/sh
# Re-create recorded human games in the engine: ./replay.sh <main class simple name> args...
# Builds with `mvn package` (never install: other checkouts load the gemp jars from ~/.m2) and runs with this
# checkout's four gemp jars ahead of the shared runtime classpath, the ~/.m2 gemp entries dropped.
set -e
cd "$(dirname "$0")"
if [ -z "$JAVA_HOME" ] && [ -d /opt/homebrew/opt/openjdk@21 ]; then
    JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi
export JAVA_HOME
jar=gemp-lotr-server/target/gemp-lotr-server-1.0-SNAPSHOT.jar
if [ ! -f "$jar" ] || [ -n "$(find gemp-lotr-server/src/main gemp-lotr-logic/src/main gemp-lotr-common/src/main -newer "$jar" -name '*.java' | head -1)" ]; then
    echo "building..." >&2
    mvn -B -q package -DskipTests -pl gemp-lotr-server -am >&2
fi
base=${GEMP_CLASSPATH:-$HOME/dev/gemp-lotr/gemp-lotr/gemp-lotr-server/target/runtime-classpath.txt}
rest=$(tr ':' '\n' < "$base" | grep -v '/com/gempukku/lotro/gemp-lotr-' | paste -sd: -)
cp="$jar:gemp-lotr-logic/target/gemp-lotr-logic-1.0-SNAPSHOT.jar:gemp-lotr-common/target/gemp-lotr-common-1.0-SNAPSHOT.jar:gemp-lotr-cards/target/gemp-lotr-cards-1.0-SNAPSHOT.jar:$rest"
main=$1; shift
case "$main" in *.*) ;; *) main=com.gempukku.lotro.bots.simulation.$main ;; esac
exec "$JAVA_HOME/bin/java" $JAVA_OPTS -cp "$cp" "$main" "$@"
