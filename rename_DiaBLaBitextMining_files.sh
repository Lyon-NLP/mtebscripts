for f in results/*/*/DiaBLa*; do
    mv "$f" "${f/DiaBLa/DiaBla}"
done   

for f in results/*/DiaBLa*; do
    mv "$f" "${f/DiaBLa/DiaBla}"
done  