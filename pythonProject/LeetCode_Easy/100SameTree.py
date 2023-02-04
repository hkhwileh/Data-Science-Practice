class Solution(object):

p=[1,2,3]
q= [1,2,3]


def isSameTree(self, p, q):
    if not p and not q:
        return True
    if (not p or not q) or (p.val != q.val):
        return False
    else:
        return (self.isSameTree(p.left, q.left) and
                self.isSameTree(p.right, q.right))


print(isSameTree(p,q))