# Run the comparison script for all the benchmarks.
echo "["
first=true
# Read bench.txt and run the comparison script for each line.
for f in $(cat bench.txt); do
    # Emit it as json.
    # only emit the file name, not the full path.
    JSON=$(sh compare2.sh $1 $2 $f)
    # Skip if exit code is not 0.
    if [ $? -ne 0 ]; then
        continue
    fi
    if [ $first = false ]; then
        echo ","
    fi
    first=false
    echo "{\"file\": \"$(basename $f)\", \"json\": $JSON}"
done
echo "]"
