# tilecoding
A python implementation of tile coding using numpy.

The theano code will most likely have better performance on the CPU. I suspect the indexing used in this implementation does not play nicely with the GPUs memory access. Also, the theano code has had less testing and could still contain some bugs.

# Usage examples for a 3D state space

First, we consider the necessary arguments to get a simple 10x10x10 grid discretization (i.e.,one layer)
over all 3 dimensions This gives us a feature representation which, if combined with a linear function, would
give us a piece-wise constant approximation.
```python
state_range = [[min_d0, min_d1, min_d2], [max_d0, max_d1, max_d2]]
tc = representation.TileCoding(input_indices = [np.arange(3)],
						ntiles = [10],
						ntilings = [1],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())
```
or, equivalently,
```python
tc = representation.TileCoding(input_indices = [np.arange(3)],
						ntiles = [np.array([10,10,10], dtype='int')],
						ntilings = [1],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())

```
It is important for the first three arguments to be lists with the same number of entries, if offsets are included 
it must be a list of the same size as well. The reason for this will be more apparent later.

In this previous example, input_indices tells us that we want one set of tilings over the first three input
dimensions. The argument ntiles tells us we want 10 discretization per inputs (for a 10x10x10 grid) and ntilings 
tells us that we only want one layer in our set of tilings. The resulting representation will always ouput only
one index corresponding to the tile containing the given state since there is only one layer.

The following example shows how we could add several overlapping 10x10x10 grids to increase the expressiveness of
our representation without making tiles smaller. 
(smaller tiles \-\> less generalization, larger tiles \-\> more generalization)
```python
tc = representation.TileCoding(input_indices = [np.arange(3)],
						ntiles = [10],
						ntilings = [5],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())
```
This gives us 5 overlapping set of tilings, each 10x10x10 grids. By default, tilings generated this way will be
given a uniform random offset based on the width a tile for the a given layer. In some cases, we might want
to specify the offsets ourselves, either to randomize the representation in a specific way or add domain knowledge. Random offsets can
be created manually with the offset argument as such:
```python
ntiles = [np.array([10,10,10], dtype='int')]
ntilings = [5]
random_offsets = [-1.0/num_tiles[:,None] * np.random.rand(num_tiles.shape[0], num_tilings) 
					for num_tiles, num_tilings in zip(ntiles, ntilings)]
```
This way of creating random offsets should allow you to easily create random offsets for various ntiles and ntilings
as long as ntiles does not use the int shortcut and only contains arrays of int. Specifying the offsets manually removes
the requirement for a random stream. The final constructor then looks like this:
```python
tc = representation.TileCoding(input_indices = [np.arange(3)],
						ntiles = ntiles,
						ntilings = ntilings,
						hashing = None,
						offset = random_offsets,
						state_range = state_range)
```
The offset argument must provide an offset for each input dimension and each layer of tilings, which is why the 
offsets provided have shape = (3,5). To ensure proper coverage of the the tilings over the state space, offsets
for a given input dimension d should be negative and contained in [0, -1.0/ntiles[d]], where ntiles[d] is the number
of discretization along dimension d.

Up to now, we've only considered simple sets of tilings over all inputs but the machinery provided can do a lot
more to conveniently build complex sets of tilings. Here is an example where we only want tilings over the
second dimension, corresponding to 5 stacked discritizations of size 10:
```python
tc1 = representation.TileCoding(input_indices = [[1]],
						ntiles = [10],
						ntilings = [5],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())
```
We might also want to add another 1D discritization on a different dimension which is a little coarser but with
a few more layers, for example:
```python
tc2 = representation.TileCoding(input_indices = [[2]],
						ntiles = [5],
						ntilings = [7],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())
```
Combining these together is tedious as we have to properly keep track of index offsets. Instead, TileCoding offers 
this automatically if these two sets of tilings are built together in the following way:
```python
tc = representation.TileCoding(input_indices = [[1],[2]],
						ntiles = [10, 5],
						ntilings = [5, 7],
						hashing = None,
						state_range = state_range,
						rnd_stream = np.random.RandomState())
```
From there, we can build complex sets of tilings with relative ease. Suppose we wanted a set of 1D tilings for each
dimension as well as another set of tilings over all three dimensions, all with random offsets. We can achieve this with
the following lines:
```python
# define the input dimensions for the different sets
input_indices = [np.arange(3), [0], [1], [2]]

# define how many discritization for each dimension each sets should use
ntiles = [[10,10,10], [5], [5], [5]]
ntiles = [np.array(x, dtype='int') for x in ntiles]

# how many different layers should the different sets use
ntilings = [8, 4, 4, 4]

# get random offsets for all tilings
random_offsets = [-1.0/num_tiles[:,None] * np.random.rand(num_tiles.shape[0], num_tilings) 
					for num_tiles, num_tilings in zip(ntiles, ntilings)]

tc = representation.TileCoding(input_indices = input_indices,
						ntiles = ntiles,
						ntilings = ntilings,
						hashing = None,
						offsets = random_offsets,
						state_range = state_range)
```