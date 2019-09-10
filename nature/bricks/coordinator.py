from nature import ConcatCoords1D, ConcatCoords2D

def Coordinator(shape):
    if len(shape) is 3:
        return ConcatCoords1D()
    else:
        return ConcatCoords2D()
