from treelib import Node, Tree

def represent_tree(list):
    root, *tail = list
    tree = Tree()
    node = Node(root)
    tree.add_node(node)

    queue = [[node, *tail]]

    while queue:
        parent, *children = queue.pop()

        for child in children:
            if instance(child, list):
                head, *tail = child
                node = tree.create_node(head, parent=parent)
                queue.append([node, *tail])
            else:
                tree.create_node(child, parent=parent)

    tree.show()
