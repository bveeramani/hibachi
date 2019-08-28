
def lattice(d):

    def increment(x, b=2):
        if not x:
            return []

        assert 0 <= x[0] < b
        if x[0]:
            return [0] + increment(x[1:])
        else:
            return [1] + x[1:]

    x = [0 for _ in range(d)]
    for _ in range(pow(2, d)):
        yield torch.tensor(x)
        x = increment(x)
