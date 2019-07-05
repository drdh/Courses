
#from anytree import AnyNode, RenderTree

from anytree import NodeMixin, RenderTree
class MyBaseClass(object):
    foo = 4
class MyClass(MyBaseClass, NodeMixin):  # Add Node feature
    def __init__(self, name, length, width, parent=None):
        super(MyClass, self).__init__()
        self.name = name
        self.length = length
        self.width = width
        self.parent = parent
        
def test():
    my0 = MyClass('my0', 0, 0)
    my1 = MyClass('my1', 1, 0, parent=my0)
    my2 = MyClass('my2', 0, 2, parent=my0)
    print(RenderTree(my0))


if __name__ == '__main__':
	test()