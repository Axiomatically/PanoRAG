## Remember to set {cls} before indexing ~~~
## export cls=agriculture ##

echo "Running context_insert.py ..."
python context_insert.py

echo "construct_topic_tree.py ..."
python construct_topic_tree.py #Generating the topic-tree

echo "build_topic_tree_vdb.py ..."
python build_topic_tree_vdb.py #Embedding for dense search

echo "topics_insert.py ..."
python topics_insert.py