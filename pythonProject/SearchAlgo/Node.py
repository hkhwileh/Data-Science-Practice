class Node(object):
    ''' initiate the class Node with root, left , right '''
    def __init__(self, object):
        self.value = object
        self.left = None
        self.right = None


class BinaryTree(object):
    ''' initiate the class Binary Treewith the @Node class'''
    def __init__(self):
        self.root = Node(object)


''' this is implementation for Binary Tree , its define the following tree
                
                    1
                   / \
                  2   3
                 / \  / \
                4   5 6   7           

'''

tree =BinaryTree(1)
tree.root.left =Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)
tree.root.right.left = Node(6)
tree.root.right.right = Node(7)
