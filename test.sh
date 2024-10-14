# Run the comparison script for all the benchmarks.
echo "["
first=true
for f in $(ls $1/*.fir); do
    # Emit it as json.
    # only emit the file name, not the full path.
    JSON=$(sh compare.sh $f)
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