remote=$1
git clone --depth=1 $remote treelstm-tmp
cd treelstm-tmp
current_commit="$(git revt rev-parse HEAD)"
last_commit="$(cat ../HEAD)"

if [[ $current_commit == $last_commit ]]; then
    echo "updating ...."
    comment=git log $(last_commit)..
    rm -rf .git
    cd ..
    rm -rf treelstm
    mv treelstm-tmp treelstm
    echo "current_commit" > HEAD
    git add .
    git commit -m $comment
else
    echo "no updates"
fi

