#!/bin/bash
python -m pip install --no-build-isolation --no-deps . -vvv
CARGO_DIR="$PREFIX/info/${PKG_NAME}_cargo"
mkdir -p $CARGO_DIR
python $RECIPE_DIR/reqs.py . ${CARGO_DIR}/requirements.txt

if [ -f $RECIPE_DIR/post-install.sh ]; then
cp $RECIPE_DIR/post-install.sh $CARGO_DIR
fi

if [ -d $RECIPE_DIR/script ]; then
cp -r $RECIPE_DIR/script $CARGO_DIR
fi

cat << EOF > $PREFIX/bin/.${PKG_NAME}-post-link.sh
#/bin/bash
pip install -r ${CARGO_DIR}/requirements.txt
if [ -f ${CARGO_DIR}/post-install.sh ]; then
   bash ${CARGO_DIR}/post-install.sh  $PREFIX
fi
EOF